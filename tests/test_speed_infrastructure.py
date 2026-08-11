from pathlib import Path

import numpy as np
import pytest
from dm_control.rl import control

from policy_speed_env import create_recorded_chunk_speed_env, create_speed_env
from speed_policy import (
    FixedSpeedPolicy,
    SpeedContext,
    SpeedPolicyAdapter,
    SpeedProfilePolicy,
    rollout_speed_policy,
)


def test_speed_policy_adapter_validates_external_output():
    context = SpeedContext(0.0, 0, 100, (1.0, 1.5))
    policy = SpeedPolicyAdapter(lambda observation, metadata: 1.5)
    assert policy(np.zeros(3), context) == 1.5

    invalid = SpeedPolicyAdapter(lambda observation, metadata: 0.0)
    with pytest.raises(ValueError, match="positive"):
        invalid(np.zeros(3), context)


def test_profile_policy_segments_nominal_time():
    policy = SpeedProfilePolicy([1.0, 2.0, 3.0])
    assert policy.select_speed(None, SpeedContext(0, 0, 90, (1.0,))) == 1.0
    assert policy.select_speed(None, SpeedContext(40, 0, 90, (1.0,))) == 2.0
    assert policy.select_speed(None, SpeedContext(89, 0, 90, (1.0,))) == 3.0


def test_fixed_speed_policy_pairs_with_scripted_environment():
    env = create_speed_env("tea_bag", seed=0)
    result = rollout_speed_policy(env, FixedSpeedPolicy(1.5))
    assert result["success"]
    assert result["physics_steps"] < env.episode_len


def test_physics_instability_becomes_failed_terminal_transition(monkeypatch):
    env = create_speed_env("tea_bag", seed=0)
    env.reset()

    def unstable_step(action):
        del action
        raise control.PhysicsError("invalid simulated state")

    monkeypatch.setattr(env.env, "step", unstable_step)
    observation, reward, done, info = env.step(3.0, quantized=False)

    assert done
    assert not info["success"]
    assert "invalid simulated state" in info["physics_error"]
    assert observation.shape == (env.obs_space,)
    assert np.isfinite(reward)


def test_recorded_chunk_policy_pairs_with_speed_environment():
    env = create_recorded_chunk_speed_env("tea_bag", chunk_size=25, seed=0)
    result = rollout_speed_policy(env, FixedSpeedPolicy(1.0))
    assert result["success"]
    assert result["mean_speed"] == 1.0


@pytest.mark.rl
def test_prioritized_replay_samples_newest_transition():
    pytest.importorskip("torch")
    from rl.rainbowDQN.replayBuffer import PrioritizedReplayBuffer

    replay = PrioritizedReplayBuffer(
        obs_dim=1,
        size=8,
        batch_size=4,
        alpha=1.0,
    )
    for value in range(4):
        replay.store(
            np.array([value], dtype=np.float32),
            value,
            0.0,
            np.array([value + 1], dtype=np.float32),
            False,
        )
    replay.update_priorities(
        np.arange(4),
        np.array([1e-6, 1e-6, 1e-6, 100.0]),
    )

    assert 3 in replay.sample_batch(beta=0.4)["indices"]


@pytest.mark.rl
def test_public_rainbow_training_checkpoint_round_trip(tmp_path: Path):
    pytest.importorskip("torch")
    from speed_policy import RainbowSpeedPolicy
    from speed_training import (
        RainbowTrainingConfig,
        evaluate_rainbow_speed_policy,
        train_rainbow_speed_policy,
    )

    env = create_speed_env("tea_bag", speed_values=(1.0,), seed=0)
    config = RainbowTrainingConfig(
        decisions=12,
        memory_size=64,
        batch_size=4,
        learning_starts=4,
        frame_skip=50,
        gradient_steps=1,
        train_interval=1,
        target_update=2,
        norm_update_interval=2,
        exploration_steps=20,
        atom_size=11,
        n_step=3,
        hidden_dim=32,
        update_schedule="episode",
        checkpoint_interval=6,
    )
    checkpoint = tmp_path / "speed.pt"
    result = train_rainbow_speed_policy(
        env,
        checkpoint,
        config=config,
        seed=0,
        device="cpu",
        progress=False,
    )
    assert checkpoint.exists()
    assert len(result["numbered_checkpoints"]) == 2
    assert all(Path(path).exists() for path in result["numbered_checkpoints"])
    assert result["updates"] > 0
    assert result["losses_finite"]

    observation = env.reset()
    policy = RainbowSpeedPolicy.load(checkpoint)
    assert policy.frame_skip == 50
    assert policy.observation_spec == env.observation_spec()
    assert policy.environment_spec == env.environment_spec()
    assert policy.select_speed(
        observation,
        SpeedContext(0, 0, env.episode_len, env.speed_values),
    ) in env.speed_values
    evaluation = evaluate_rainbow_speed_policy(
        env, checkpoint, episodes=1, device="cpu"
    )
    assert evaluation["episodes"] == 1
    assert "mean_acceleration" in evaluation

    randomized_env = create_speed_env(
        "tea_bag",
        speed_values=(1.0,),
        randomize_object_pose=True,
        seed=0,
    )
    with pytest.raises(ValueError, match="Checkpoint environment"):
        rollout_speed_policy(randomized_env, policy)
