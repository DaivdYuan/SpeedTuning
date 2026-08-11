"""Model-agnostic speed-policy adapters and rollout utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class SpeedContext:
    """Episode metadata passed to a speed policy at each decision."""

    policy_time: float
    physics_steps: int
    episode_len: int
    speed_values: tuple[float, ...]


class SpeedPolicyAdapter:
    """Adapt a callable or ``select_speed`` object to the speed contract.

    A public speed policy receives ``(observation, context)`` and returns a
    finite positive multiplier such as ``1.0`` or ``1.5``.
    """

    def __init__(self, policy: Any):
        select = getattr(policy, "select_speed", None)
        if select is None and callable(policy):
            select = policy
        if select is None:
            raise TypeError("speed policy must be callable or define select_speed()")
        self.policy = policy
        self._select = select

    def reset(self):
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            reset()

    def __call__(self, observation, context):
        speed = float(self._select(observation, context))
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("A speed policy must return a finite positive multiplier")
        return speed


class FixedSpeedPolicy:
    """Always choose one physical speed multiplier."""

    def __init__(self, speed=1.0):
        self.speed = float(speed)
        if not np.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("speed must be finite and positive")

    def select_speed(self, observation, context):
        del observation, context
        return self.speed


class SpeedProfilePolicy:
    """Select piecewise-constant speeds over normalized nominal policy time."""

    def __init__(self, speeds: Sequence[float]):
        values = np.asarray(speeds, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("speeds must be a non-empty one-dimensional sequence")
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Every profile speed must be finite and positive")
        self.speeds = tuple(float(value) for value in values)

    def select_speed(self, observation, context):
        del observation
        fraction = min(max(context.policy_time / context.episode_len, 0.0), 1.0)
        index = min(int(fraction * len(self.speeds)), len(self.speeds) - 1)
        return self.speeds[index]


class CallableSpeedPolicy:
    """Give a descriptive wrapper to a user-provided speed function."""

    def __init__(self, function: Callable[[np.ndarray, SpeedContext], float]):
        self.function = function

    def select_speed(self, observation, context):
        return self.function(observation, context)


class RainbowSpeedPolicy:
    """Inference-only speed policy loaded from a public training checkpoint."""

    def __init__(
        self,
        network,
        speed_values,
        device="cpu",
        frame_skip=10,
        observation_spec=None,
        environment_spec=None,
        observation_encoder_state_dict=None,
        checkpoint_metadata=None,
    ):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()
        self.speed_values = tuple(float(value) for value in speed_values)
        self.observation_dim = int(network.in_dim)
        self.frame_skip = int(frame_skip)
        self.observation_spec = observation_spec
        self.environment_spec = environment_spec
        self.observation_encoder_state_dict = observation_encoder_state_dict
        self.checkpoint_metadata = dict(checkpoint_metadata or {})

    @classmethod
    def load(cls, checkpoint_path, device="cpu"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Rainbow evaluation requires: uv sync --extra rl") from exc
        from rl.rainbowDQN.network import Network

        checkpoint_path = Path(checkpoint_path)
        # Public speed checkpoints contain tensors and primitive metadata only,
        # so use PyTorch's restricted unpickler for downloaded artifacts.
        payload = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        required = {
            "model_state_dict",
            "observation_dim",
            "speed_values",
            "atom_size",
            "v_min",
            "v_max",
            "hidden_dim",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"Speed checkpoint is missing keys: {', '.join(missing)}")
        support = torch.linspace(
            float(payload["v_min"]),
            float(payload["v_max"]),
            int(payload["atom_size"]),
        ).to(device)
        network = Network(
            int(payload["observation_dim"]),
            len(payload["speed_values"]),
            int(payload["atom_size"]),
            support,
            hidden_dim=int(payload["hidden_dim"]),
        )
        network.load_state_dict(payload["model_state_dict"])
        training_config = payload.get("training_config", {})
        return cls(
            network,
            payload["speed_values"],
            device=device,
            frame_skip=payload.get(
                "decision_frame_skip", training_config.get("frame_skip", 10)
            ),
            observation_spec=payload.get("observation_spec"),
            environment_spec=payload.get("environment_spec"),
            observation_encoder_state_dict=payload.get(
                "observation_encoder_state_dict"
            ),
            checkpoint_metadata=payload.get("metadata"),
        )

    def select_action(self, observation):
        tensor = self.torch.as_tensor(
            observation, dtype=self.torch.float32, device=self.device
        ).unsqueeze(0)
        with self.torch.inference_mode():
            action = int(self.network(tensor).argmax(dim=1).item())
        return action

    def select_speed(self, observation, context):
        del context
        return self.speed_values[self.select_action(observation)]

    def configure_environment(self, env):
        """Restore and validate observation preprocessing for evaluation."""

        env.load_observation_encoder_state_dict(
            self.observation_encoder_state_dict
        )
        observation = env.reset()
        if self.observation_dim != observation.size:
            raise ValueError(
                "Checkpoint observation size does not match the environment: "
                f"{self.observation_dim} != {observation.size}"
            )
        if tuple(self.speed_values) != tuple(env.speed_values):
            raise ValueError(
                "Checkpoint speed_values do not match the environment: "
                f"{self.speed_values} != {env.speed_values}"
            )
        if self.observation_spec is not None:
            actual_spec = env.observation_spec()
            if self.observation_spec != actual_spec:
                raise ValueError(
                    "Checkpoint observation preprocessing does not match the environment: "
                    f"{self.observation_spec!r} != {actual_spec!r}"
                )
        if self.environment_spec is not None:
            actual_environment = env.environment_spec()
            if self.environment_spec != actual_environment:
                raise ValueError(
                    "Checkpoint environment does not match evaluation: "
                    f"{self.environment_spec!r} != {actual_environment!r}"
                )
        return observation


def rollout_speed_policy(env, speed_policy, capture_speeds=False, frame_skip=None):
    """Pair any conforming speed policy with a configured speed environment."""

    raw_policy = speed_policy
    prepared_observation = None
    if isinstance(raw_policy, RainbowSpeedPolicy):
        prepared_observation = raw_policy.configure_environment(env)
    policy = (
        speed_policy
        if isinstance(speed_policy, SpeedPolicyAdapter)
        else SpeedPolicyAdapter(speed_policy)
    )
    policy.reset()
    observation = (
        env.reset() if prepared_observation is None else prepared_observation
    )
    decision_frame_skip = int(
        frame_skip
        if frame_skip is not None
        else getattr(raw_policy, "frame_skip", env.decision_frame_skip)
    )
    if decision_frame_skip <= 0:
        raise ValueError("frame_skip must be positive")
    done = False
    total_reward = 0.0
    info = {"success": False}
    speeds = []
    decisions = 0
    while not done:
        context = SpeedContext(
            policy_time=env.policy_time,
            physics_steps=env.physics_steps,
            episode_len=env.episode_len,
            speed_values=env.speed_values,
        )
        speed = policy(observation, context)
        observation, reward, done, info = env.step_decision(
            speed,
            frame_skip=decision_frame_skip,
            quantized=False,
        )
        total_reward += reward
        decisions += 1
        if capture_speeds:
            speeds.append(speed)

    acceleration = float(env.episode_len / max(info["physics_steps"], 1))
    result = {
        "success": bool(info["success"]),
        "return": float(total_reward),
        "physics_steps": int(info["physics_steps"]),
        "policy_time": float(info["policy_time"]),
        "mean_speed": float(np.mean(env.speed_list)),
        "max_speed": float(np.max(env.speed_list)),
        "acceleration": acceleration,
        "successful_acceleration": acceleration if info["success"] else None,
        "decisions": decisions,
        "decision_frame_skip": decision_frame_skip,
        "duration_seconds": float(info["physics_steps"] / 50.0),
        "nominal_duration_seconds": float(env.episode_len / 50.0),
    }
    if capture_speeds:
        result["speeds"] = speeds
    return result


def summarize_rollouts(rollouts):
    """Aggregate paper-facing success and physical-acceleration metrics."""

    rollouts = list(rollouts)
    if not rollouts:
        raise ValueError("At least one rollout is required")
    successes = np.asarray([item["success"] for item in rollouts], dtype=np.float64)
    accelerations = np.asarray(
        [item["acceleration"] for item in rollouts], dtype=np.float64
    )
    successful = accelerations[successes.astype(bool)]
    physics_steps = np.asarray(
        [item["physics_steps"] for item in rollouts], dtype=np.float64
    )
    mean_speeds = np.asarray(
        [item["mean_speed"] for item in rollouts], dtype=np.float64
    )
    return {
        "episodes": len(rollouts),
        "successes": int(successes.sum()),
        "success_rate": float(successes.mean()),
        "success_standard_error": float(
            np.sqrt(successes.mean() * (1.0 - successes.mean()) / len(successes))
        ),
        "mean_acceleration": float(accelerations.mean()),
        "median_acceleration": float(np.median(accelerations)),
        "acceleration_standard_deviation": float(accelerations.std()),
        "acceleration_25th_percentile": float(np.percentile(accelerations, 25)),
        "acceleration_75th_percentile": float(np.percentile(accelerations, 75)),
        "mean_successful_acceleration": (
            None if successful.size == 0 else float(successful.mean())
        ),
        "successful_acceleration_standard_deviation": (
            None if successful.size == 0 else float(successful.std())
        ),
        "mean_physics_steps": float(physics_steps.mean()),
        "mean_commanded_speed": float(mean_speeds.mean()),
    }
