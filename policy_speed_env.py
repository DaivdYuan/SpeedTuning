"""Speed-control environments for scripted and chunked robot policies.

The speed agent never produces robot actions. It observes simulator state and
chooses how quickly an independent base policy advances through nominal policy
time. This separation lets the same speed learner wrap retained waypoint
controllers, ACT, or another upstream policy that emits 14D action chunks.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from dm_control.rl import control

from chunked_policy import (
    ChunkedPolicyRunner,
    RecordedChunkPredictor,
    collect_scripted_joint_demonstration,
)
from ee_sim_env import make_ee_sim_env
from scripted_policy import make_scripted_policy
from sim_env import make_sim_env
from sim_tasks import get_task_spec, normalize_task_name
from speed_observation import StateObservationEncoder, encoder_spec


DEFAULT_SPEEDS = (1.0, 1.5, 2.0, 2.5, 3.0)


def terminal_success_reward(speed, done, success):
    """Sparse objective used by the initial SpeedTuning experiments."""

    del speed
    return 100.0 if done and success else 0.0


def make_speed_reward(success_bonus=100.0, speed_weight=0.01, speed_power=2.0):
    """Build a serializable-style speed objective for public experiments."""

    if speed_power < 0:
        raise ValueError("speed_power must be non-negative")

    def reward_fn(speed, done, success):
        reward = speed_weight * float(speed) ** speed_power
        if done and success:
            reward += success_bonus
        return reward

    return reward_fn


def encode_state_observation(observation):
    """Flatten task state, robot position, and robot velocity for a speed agent."""

    return StateObservationEncoder()(observation)


class WaypointActionSource:
    """Advance an end-effector waypoint controller by a requested speed."""

    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            reset()

    def begin_decision(self, timestep, speed):
        del timestep, speed

    def action(self, timestep, speed):
        return self.policy(timestep, step_inc=speed)


class ChunkedActionSource:
    """Advance an arbitrary joint-action chunk predictor by a requested speed."""

    def __init__(self, predictor):
        self.runner = ChunkedPolicyRunner(predictor)

    def reset(self):
        self.runner.reset()

    def begin_decision(self, timestep, speed):
        self.runner.begin_decision(timestep.observation, speed=speed)

    def action(self, timestep, speed):
        return self.runner.action(timestep.observation, speed=speed)


class SpeedPolicyEnv:
    """A small RL environment whose actions select base-policy speed.

    The interface follows the classic ``reset() -> observation`` and
    ``step(action) -> observation, reward, done, info`` convention used by the
    included Rainbow implementation. Discrete action ``i`` maps to
    ``speed_values[i]``; callers may also pass a physical speed with
    ``quantized=False``.
    """

    def __init__(
        self,
        env,
        action_source,
        episode_len,
        reward_fn=None,
        speed_values: Sequence[float] = DEFAULT_SPEEDS,
        observation_encoder: Callable[[dict], np.ndarray] | None = None,
        frame_stack=1,
        decision_frame_skip=10,
        save_video=False,
        onscreen_render=False,
        video_path="output_video.mp4",
        max_physics_steps=None,
        terminate_on_success=False,
        environment_metadata=None,
    ):
        self.env = env
        self.action_source = action_source
        self.reward_fn = reward_fn or terminal_success_reward
        self.episode_len = int(episode_len)
        self.observation_encoder = observation_encoder or StateObservationEncoder()
        self.frame_stack = int(frame_stack)
        self.decision_frame_skip = int(decision_frame_skip)
        if self.frame_stack <= 0 or self.decision_frame_skip <= 0:
            raise ValueError("frame_stack and decision_frame_skip must be positive")
        self.save_video = bool(save_video)
        self.onscreen_render = bool(onscreen_render)
        self.video_path = Path(video_path)
        self.terminate_on_success = bool(terminate_on_success)
        self._environment_metadata = dict(environment_metadata or {})

        values = np.asarray(speed_values, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("speed_values must be a non-empty one-dimensional sequence")
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Every speed value must be finite and positive")
        self.speed_values = tuple(float(value) for value in values)
        self.action_space = len(self.speed_values)
        self.max_physics_steps = int(
            max_physics_steps
            if max_physics_steps is not None
            else np.ceil(self.episode_len / min(self.speed_values))
        )
        if self.max_physics_steps <= 0:
            raise ValueError("max_physics_steps must be positive")

        env_state_dim = int(self.env.physics.data.qpos[16:].size)
        output_dim = getattr(self.observation_encoder, "output_dim", None)
        try:
            single_observation_dim = int(output_dim(env_state_dim))
        except (AttributeError, TypeError):
            single_observation_dim = env_state_dim + 28
        self.obs_space = single_observation_dim * self.frame_stack
        self.cur_ts = None
        self.cur_success = False
        self.policy_time = 0.0
        self.physics_steps = 0
        self.speed_list = []
        self.image_list = []
        self._observation_stack = deque(maxlen=self.frame_stack)
        self._figure = None
        self._plot_image = None

    def reset(self):
        self.cur_ts = self.env.reset()
        self.action_source.reset()
        reset_encoder = getattr(self.observation_encoder, "reset", None)
        if reset_encoder is not None:
            reset_encoder()
        self.cur_success = False
        self.policy_time = 0.0
        self.physics_steps = 0
        self.speed_list = []
        self.image_list = []
        self._observation_stack.clear()

        if self.onscreen_render:
            import matplotlib.pyplot as plt

            self._figure, axis = plt.subplots()
            self._plot_image = axis.imshow(self.cur_ts.observation["images"]["angle"])
            plt.ion()
        encoded = self._encode_observation(self.cur_ts.observation)
        for _ in range(self.frame_stack):
            self._observation_stack.append(encoded.copy())
        observation = self.get_obs()
        self.obs_space = int(observation.size)
        return observation

    def speed_from_action(self, action):
        if not isinstance(action, (int, np.integer)):
            raise ValueError("A discrete speed action must be an integer index")
        action = int(action)
        if not 0 <= action < self.action_space:
            raise ValueError(f"Speed action must be in [0, {self.action_space - 1}]")
        return self.speed_values[action]

    @staticmethod
    def _validate_continuous_speed(speed):
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Continuous speed must be finite and positive")
        return speed

    def _resolve_speed(self, speed, quantized):
        return (
            self.speed_from_action(speed)
            if quantized
            else self._validate_continuous_speed(speed)
        )

    def begin_decision(self, speed, quantized=True):
        """Select one speed and force a fresh receding-horizon task chunk."""

        if self.cur_ts is None:
            raise RuntimeError("Call reset() before beginning a decision")
        resolved_speed = self._resolve_speed(speed, quantized)
        begin = getattr(self.action_source, "begin_decision", None)
        if begin is not None:
            begin(self.cur_ts, resolved_speed)
        return resolved_speed

    def step(self, speed, quantized=True):
        """Advance one physics step, retaining legacy low-level behavior."""

        return self._step_physics(self._resolve_speed(speed, quantized))

    def step_decision(self, speed, frame_skip=None, quantized=True):
        """Execute one speed-policy action for a fixed block of physics steps.

        A fresh task-policy chunk is predicted at the decision boundary. Rewards
        inside the block are summed without discounting and one transition is
        returned to the speed learner.
        """

        repeats = self.decision_frame_skip if frame_skip is None else int(frame_skip)
        if repeats <= 0:
            raise ValueError("frame_skip must be positive")
        resolved_speed = self.begin_decision(speed, quantized=quantized)
        total_reward = 0.0
        executed = 0
        info = {"success": False}
        for _ in range(repeats):
            observation, reward, done, info = self._step_physics(resolved_speed)
            total_reward += float(reward)
            executed += 1
            if done:
                break
        info = dict(info)
        info.update(
            decision_frame_skip=repeats,
            decision_physics_steps=executed,
            reward_aggregation="undiscounted_sum",
        )
        return observation, total_reward, done, info

    def _step_physics(self, speed):
        if self.cur_ts is None:
            raise RuntimeError("Call reset() before step()")
        if self.policy_time >= self.episode_len or self.physics_steps >= self.max_physics_steps:
            raise RuntimeError("The episode has already finished")

        action = self.action_source.action(self.cur_ts, speed)
        try:
            next_timestep = self.env.step(action)
        except control.PhysicsError as exc:
            # Aggressive speed exploration can occasionally drive MuJoCo into an
            # invalid state. Treat that rollout as a failed terminal transition
            # so one unstable sample cannot abort a long reinforcement-learning
            # run. The following reset restores the simulator state.
            self.policy_time += speed
            self.physics_steps += 1
            self.speed_list.append(speed)
            done = True
            reward = float(self.reward_fn(speed, done, False))
            info = {
                "success": False,
                "speed": speed,
                "policy_time": self.policy_time,
                "physics_steps": self.physics_steps,
                "task_reward": 0.0,
                "target_reward": self.env.task.max_reward,
                "physics_error": str(exc),
            }
            return self.get_obs(), reward, done, info
        self.cur_ts = next_timestep
        self.policy_time += speed
        self.physics_steps += 1
        self.speed_list.append(speed)
        self._observation_stack.append(
            self._encode_observation(self.cur_ts.observation)
        )

        if self.onscreen_render:
            import matplotlib.pyplot as plt

            self._plot_image.set_data(self.cur_ts.observation["images"]["angle"])
            plt.pause(0.002)
        if self.save_video:
            self.image_list.append(self.cur_ts.observation["images"]["angle"])

        task_reward = float(self.cur_ts.reward or 0)
        if task_reward >= self.env.task.max_reward:
            self.cur_success = True
        timed_out = (
            self.policy_time >= self.episode_len
            or self.physics_steps >= self.max_physics_steps
        )
        done = timed_out or (self.terminate_on_success and self.cur_success)
        reward = float(self.reward_fn(speed, done, self.cur_success))

        if done and self.save_video:
            self._save_video()
        info = {
            "success": self.cur_success,
            "speed": speed,
            "policy_time": self.policy_time,
            "physics_steps": self.physics_steps,
            "task_reward": task_reward,
            "target_reward": self.env.task.max_reward,
        }
        return self.get_obs(), reward, done, info

    def _encode_observation(self, observation_dict):
        observation = np.asarray(
            self.observation_encoder(observation_dict), dtype=np.float32
        )
        if observation.ndim != 1 or not np.all(np.isfinite(observation)):
            raise ValueError("The speed-policy observation must be a finite 1D array")
        return observation

    def get_obs(self):
        if self.cur_ts is None or not self._observation_stack:
            raise RuntimeError("Call reset() before requesting an observation")
        return np.concatenate(tuple(self._observation_stack)).astype(
            np.float32, copy=False
        )

    def observation_spec(self):
        return {
            "encoder": encoder_spec(self.observation_encoder),
            "frame_stack": self.frame_stack,
            "padding": "repeat_initial",
            "observation_dim": int(self.obs_space),
        }

    def environment_spec(self):
        return dict(self._environment_metadata)

    def observation_encoder_state_dict(self):
        state_dict = getattr(self.observation_encoder, "state_dict", None)
        return None if state_dict is None else state_dict()

    def load_observation_encoder_state_dict(self, state_dict):
        if state_dict is None:
            return
        load = getattr(self.observation_encoder, "load_state_dict", None)
        if load is None:
            raise ValueError("Configured observation encoder cannot load checkpoint state")
        load(state_dict)

    def _save_video(self):
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("Video export requires: uv sync --extra video") from exc
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(self.video_path, self.image_list, fps=50)

    def close(self):
        if self._figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self._figure)
            self._figure = None
        close_encoder = getattr(self.observation_encoder, "close", None)
        if close_encoder is not None:
            close_encoder()


def create_speed_env(
    task_name="tea_bag",
    reward_fn=None,
    chunk_predictor=None,
    object_pose=None,
    onscreen_render=False,
    save_video=False,
    render_images=None,
    seed=None,
    speed_values=DEFAULT_SPEEDS,
    video_path="output_video.mp4",
    observation_encoder=None,
    frame_stack=1,
    decision_frame_skip=10,
    terminate_on_success=False,
    randomize_object_pose=False,
):
    """Create a speed environment around a scripted or chunked base policy.

    With no ``chunk_predictor`` this uses the retained end-effector waypoint
    controller. Supplying any callable chunk predictor switches to the joint
    simulator and the generic ``[time, 14]`` action-chunk contract.
    """

    task_name = normalize_task_name(task_name)
    spec = get_task_spec(task_name)
    observation_encoder = observation_encoder or StateObservationEncoder()
    if render_images is None:
        render_images = bool(
            onscreen_render
            or save_video
            or chunk_predictor is not None
            or getattr(observation_encoder, "requires_images", False)
        )
    if getattr(observation_encoder, "requires_images", False) and not render_images:
        raise ValueError("The configured speed observation encoder requires images")

    if chunk_predictor is None:
        env = make_ee_sim_env(
            task_name,
            render_images=render_images,
            seed=seed,
            object_pose=object_pose,
            randomize_object_pose=randomize_object_pose,
        )
        action_source = WaypointActionSource(make_scripted_policy(task_name))
    else:
        env = make_sim_env(
            task_name,
            render_images=render_images,
            seed=seed,
            object_pose=object_pose,
            randomize_object_pose=randomize_object_pose,
        )
        action_source = ChunkedActionSource(chunk_predictor)

    return SpeedPolicyEnv(
        env=env,
        action_source=action_source,
        reward_fn=reward_fn,
        episode_len=spec.episode_len,
        speed_values=speed_values,
        observation_encoder=observation_encoder,
        frame_stack=frame_stack,
        decision_frame_skip=decision_frame_skip,
        onscreen_render=onscreen_render,
        save_video=save_video,
        video_path=video_path,
        terminate_on_success=terminate_on_success,
        environment_metadata={
            "task": task_name,
            "base_policy": (
                "scripted" if chunk_predictor is None else "chunked"
            ),
            "randomize_object_pose": bool(randomize_object_pose),
        },
    )


def create_recorded_chunk_speed_env(
    task_name="tea_bag",
    chunk_size=25,
    seed=0,
    render_images=False,
    **kwargs,
):
    """Create a checkpoint-free chunk environment for integration testing."""

    demonstration = collect_scripted_joint_demonstration(task_name, seed=seed)
    predictor = RecordedChunkPredictor(demonstration.actions, chunk_size=chunk_size)
    return create_speed_env(
        task_name=task_name,
        chunk_predictor=predictor,
        object_pose=demonstration.object_pose,
        render_images=render_images,
        seed=seed,
        **kwargs,
    )


def test_speed_env(
    task_name="tea_bag",
    speed_func=None,
    seed=0,
    chunk_predictor=None,
):
    """Run a continuous speed function and return the maximum wrapper reward."""

    speed_env = create_speed_env(
        task_name=task_name,
        chunk_predictor=chunk_predictor,
        seed=seed,
    )
    observation = speed_env.reset()
    done = False
    rewards = []
    while not done:
        speed = 1.0 if speed_func is None else speed_func(
            observation=observation,
            policy_time=speed_env.policy_time,
        )
        observation, reward, done, _ = speed_env.step(speed, quantized=False)
        rewards.append(reward)
    speed_env.close()
    return max(rewards, default=0.0)


if __name__ == "__main__":
    for _task_name in ("pick_and_place", "insertion", "tea_bag"):
        print(_task_name, test_speed_env(_task_name))
