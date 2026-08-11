"""Adapters for upstream policies that predict joint-action chunks.

The public contract is intentionally model-agnostic: a predictor receives one
DM Control observation dictionary and returns one or more 14D joint actions.
``ChunkPredictorAdapter`` handles common NumPy, PyTorch, tuple, and dictionary
outputs, while ``ChunkedPolicyRunner`` turns those chunks into individual
simulator actions at a parameterized execution speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from ee_sim_env import make_ee_sim_env
from scripted_policy import make_scripted_policy
from sim_env import make_sim_env
from sim_tasks import get_task_spec, normalize_task_name


def _to_numpy(value: Any) -> np.ndarray:
    """Convert common model outputs without importing a framework eagerly."""

    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if numpy is not None:
        value = numpy()
    return np.asarray(value, dtype=np.float64)


def as_action_chunk(output: Any) -> np.ndarray:
    """Normalize an upstream policy output to a finite ``[time, 14]`` array."""

    if isinstance(output, dict):
        for key in ("actions", "action", "chunk"):
            if key in output:
                output = output[key]
                break
        else:
            raise ValueError(
                "Chunk dictionaries must contain an 'actions', 'action', or 'chunk' key"
            )
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("A chunked policy returned an empty sequence")
        output = output[0]

    actions = _to_numpy(output)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1 and actions.shape[0] == 14:
        actions = actions[None]
    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) == 0:
        raise ValueError("Chunked policy output must have shape [time, 14]")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Chunked policy produced a non-finite action")
    return actions


class ChunkPredictorAdapter:
    """Adapt an arbitrary upstream chunk predictor to the public contract.

    ``predictor`` can be callable or expose ``predict_chunk(observation)``.
    Optional adapters make it possible to translate repository-specific
    observations and outputs without changing the simulator integration.
    """

    def __init__(
        self,
        predictor: Any,
        observation_adapter: Callable[[dict], Any] | None = None,
        output_adapter: Callable[[Any], Any] | None = None,
    ):
        predict = getattr(predictor, "predict_chunk", None)
        if predict is None and callable(predictor):
            predict = predictor
        if predict is None:
            raise TypeError("predictor must be callable or define predict_chunk()")
        self.predictor = predictor
        self._predict = predict
        self.observation_adapter = observation_adapter or (lambda observation: observation)
        self.output_adapter = output_adapter or (lambda output: output)

    def reset(self):
        reset = getattr(self.predictor, "reset", None)
        if reset is not None:
            reset()

    def advance(self, nominal_steps):
        """Notify predictors that explicitly track nominal demonstration time."""

        advance = getattr(self.predictor, "advance", None)
        if advance is not None:
            advance(float(nominal_steps))

    def __call__(self, observation):
        model_input = self.observation_adapter(observation)
        output = self.output_adapter(self._predict(model_input))
        return as_action_chunk(output)


def interpolate_action_chunk(actions, speed=1.0):
    """Resample a ``[time, 14]`` action chunk at a new execution speed."""

    actions = as_action_chunk(actions)
    if speed <= 0:
        raise ValueError("speed must be positive")
    sample_times = np.arange(0.0, len(actions), speed)
    lower = np.floor(sample_times).astype(int)
    upper = np.minimum(lower + 1, len(actions) - 1)
    fraction = (sample_times - lower)[:, None]
    return actions[lower] + fraction * (actions[upper] - actions[lower])


class ChunkedPolicyRunner:
    """Turn a chunk-predicting policy into one joint action per simulator step.

    Speed is expressed in nominal policy timesteps per physics step. It may be
    fixed at construction or supplied for every call, allowing a learned speed
    policy to change acceleration online without modifying the upstream model.
    """

    def __init__(self, predictor, speed=1.0):
        self.predictor = (
            predictor
            if isinstance(predictor, ChunkPredictorAdapter)
            else ChunkPredictorAdapter(predictor)
        )
        self.speed = self._validate_speed(speed)
        self._chunk = None
        self._chunk_index = 0.0

    @staticmethod
    def _validate_speed(speed):
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be a finite positive value")
        return speed

    def reset(self):
        self._chunk = None
        self._chunk_index = 0.0
        reset = getattr(self.predictor, "reset", None)
        if reset is not None:
            reset()

    def begin_decision(self, observation, speed=None):
        """Predict a fresh receding-horizon chunk for one speed decision."""

        speed = self.speed if speed is None else self._validate_speed(speed)
        self._chunk = self.predictor(observation)
        self._chunk_index = 0.0
        self._decision_speed = speed
        return self._chunk

    def action(self, observation, speed=None):
        speed = self.speed if speed is None else self._validate_speed(speed)
        if self._chunk is None or self._chunk_index >= len(self._chunk):
            self.begin_decision(observation, speed=speed)

        lower = int(np.floor(self._chunk_index))
        upper = min(lower + 1, len(self._chunk) - 1)
        fraction = self._chunk_index - lower
        action = self._chunk[lower] + fraction * (
            self._chunk[upper] - self._chunk[lower]
        )
        self._chunk_index += speed
        self.predictor.advance(speed)
        return action.copy()


class TorchChunkPredictor:
    """Pre/post-processing adapter for ACT-compatible PyTorch policies.

    The wrapped model must accept normalized ``qpos`` with shape ``[1, 14]`` and
    RGB images with shape ``[1, cameras, 3, height, width]``, and return an action
    tensor with shape ``[1, chunk, 14]``.
    """

    def __init__(
        self,
        model,
        camera_names,
        qpos_mean,
        qpos_std,
        action_mean,
        action_std,
        device=None,
    ):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Learned policies require: uv sync --extra learned") from exc
        self.torch = torch
        self.model = model
        self.camera_names = tuple(camera_names)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.qpos_mean = np.asarray(qpos_mean)
        self.qpos_std = np.maximum(np.asarray(qpos_std), 1e-6)
        self.action_mean = np.asarray(action_mean)
        self.action_std = np.asarray(action_std)
        for value, name in (
            (self.qpos_mean, "qpos_mean"),
            (self.qpos_std, "qpos_std"),
            (self.action_mean, "action_mean"),
            (self.action_std, "action_std"),
        ):
            if value.shape != (14,):
                raise ValueError(f"{name} must have shape (14,)")
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, observation):
        torch = self.torch
        if "images" not in observation:
            raise ValueError("The environment must be created with render_images=True")
        qpos = (np.asarray(observation["qpos"]) - self.qpos_mean) / self.qpos_std
        images = np.stack(
            [observation["images"][name] for name in self.camera_names], axis=0
        ).transpose(0, 3, 1, 2)
        qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)[None]
        image_tensor = torch.as_tensor(
            images / 255.0, dtype=torch.float32, device=self.device
        )[None]
        with torch.inference_mode():
            output = self.model(qpos_tensor, image_tensor)
        chunk = as_action_chunk(output)
        return chunk * self.action_std + self.action_mean


@dataclass
class JointDemonstration:
    actions: np.ndarray
    object_pose: np.ndarray


def collect_scripted_joint_demonstration(task_name, seed=0):
    """Convert the retained EE scripted rollout into joint commands."""

    task_name = normalize_task_name(task_name)
    spec = get_task_spec(task_name)
    env = make_ee_sim_env(task_name, render_images=False, seed=seed)
    timestep = env.reset()
    episode = [timestep]
    policy = make_scripted_policy(task_name)
    for _ in range(spec.episode_len):
        timestep = env.step(policy(timestep))
        episode.append(timestep)

    actions = []
    for item in episode:
        action = item.observation["qpos"].copy()
        gripper_ctrl = item.observation["gripper_ctrl"]
        action[6] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(gripper_ctrl[0])
        action[13] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(gripper_ctrl[2])
        actions.append(action)
    return JointDemonstration(
        actions=np.asarray(actions),
        object_pose=episode[0].observation["env_state"].copy(),
    )


class RecordedChunkPredictor:
    """Deterministic chunk oracle used to validate the learned-policy contract."""

    def __init__(self, actions, chunk_size):
        self.actions = np.asarray(actions)
        self.chunk_size = chunk_size
        self.cursor = 0.0

    def reset(self):
        self.cursor = 0.0

    def advance(self, nominal_steps):
        self.cursor = min(self.cursor + float(nominal_steps), len(self.actions) - 1)

    def __call__(self, observation):
        del observation
        start = int(np.floor(self.cursor))
        end = min(start + self.chunk_size, len(self.actions))
        chunk = self.actions[start:end]
        if len(chunk) < self.chunk_size:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], self.chunk_size - len(chunk), axis=0)]
            )
        return chunk


def replay_recorded_chunks(task_name, chunk_size=25, seed=0):
    """Validate chunked-policy replay through the joint-control environment."""

    task_name = normalize_task_name(task_name)
    demonstration = collect_scripted_joint_demonstration(task_name, seed=seed)
    env = make_sim_env(
        task_name,
        render_images=False,
        seed=seed,
        object_pose=demonstration.object_pose,
    )
    timestep = env.reset()
    predictor = RecordedChunkPredictor(demonstration.actions, chunk_size)
    runner = ChunkedPolicyRunner(predictor)
    rewards = []
    for _ in range(len(demonstration.actions)):
        timestep = env.step(runner.action(timestep.observation))
        rewards.append(int(timestep.reward or 0))
    return {
        "task": task_name,
        "success": max(rewards, default=0) == env.task.max_reward,
        "max_reward": max(rewards, default=0),
        "target_reward": env.task.max_reward,
        "chunk_size": chunk_size,
        "steps": len(demonstration.actions),
    }
