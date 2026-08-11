"""Training and evaluation loops for the included Rainbow speed policy."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from speed_policy import RainbowSpeedPolicy, rollout_speed_policy, summarize_rollouts


@dataclass(frozen=True)
class RainbowTrainingConfig:
    """Practical defaults for speed-policy experiments, not paper reproduction."""

    decisions: int = 5_000
    memory_size: int = 100_000
    batch_size: int = 128
    learning_starts: int = 512
    frame_skip: int = 10
    gradient_steps: int = 4
    train_interval: int = 1
    target_update: int = 50
    norm_update_interval: int = 100
    learning_rate: float = 1e-4
    gamma: float = 0.97
    tau: float = 0.5
    epsilon: float = 1.0
    epsilon_decay: float = 0.999
    min_epsilon: float = 0.1
    exploration_steps: int = 2_000
    alpha: float = 0.2
    beta: float = 0.6
    beta_schedule: str = "linear"
    atom_size: int = 121
    v_min: float = 0.0
    v_max: float = 120.0
    n_step: int = 3
    hidden_dim: int = 256
    update_schedule: str = "decision"
    checkpoint_interval: int = 0

    def validate(self):
        positive_ints = (
            "decisions",
            "memory_size",
            "batch_size",
            "learning_starts",
            "frame_skip",
            "gradient_steps",
            "train_interval",
            "target_update",
            "norm_update_interval",
            "atom_size",
            "n_step",
            "hidden_dim",
        )
        for name in positive_ints:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.memory_size < self.batch_size:
            raise ValueError("memory_size must be at least batch_size")
        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be at least batch_size")
        if self.atom_size < 2 or self.v_max <= self.v_min:
            raise ValueError("Categorical support requires atom_size >= 2 and v_max > v_min")
        if self.update_schedule not in {"decision", "episode"}:
            raise ValueError("update_schedule must be 'decision' or 'episode'")
        if self.beta_schedule not in {"linear", "legacy"}:
            raise ValueError("beta_schedule must be 'linear' or 'legacy'")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval cannot be negative")


def _normalization_stats(states):
    values = np.asarray(states, dtype=np.float32)
    return {
        "states_mean": values.mean(axis=0),
        "states_std": np.maximum(values.std(axis=0), 1e-6),
    }


def _checkpoint_payload(agent, env, config, seed, metadata, completed_decisions):
    return {
        "format_version": 2,
        "algorithm": "rainbow_dqn",
        "model_state_dict": agent.dqn.state_dict(),
        "observation_dim": int(env.obs_space),
        "speed_values": list(env.speed_values),
        "atom_size": config.atom_size,
        "v_min": config.v_min,
        "v_max": config.v_max,
        "hidden_dim": config.hidden_dim,
        "seed": int(seed),
        "training_config": asdict(config),
        "completed_decisions": int(completed_decisions),
        "decision_frame_skip": config.frame_skip,
        "reward_aggregation": "undiscounted_sum_per_decision",
        "observation_spec": env.observation_spec(),
        "environment_spec": env.environment_spec(),
        "observation_encoder_state_dict": env.observation_encoder_state_dict(),
        "metric_spec": {
            "acceleration": "episode_len / physics_steps",
            "control_frequency_hz": 50,
        },
        "metadata": dict(metadata or {}),
    }


def _numbered_checkpoint_path(checkpoint_path, decision):
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}.decision-{decision}{checkpoint_path.suffix}"
    )


def train_rainbow_speed_policy(
    env,
    checkpoint_path,
    config=None,
    seed=0,
    device=None,
    metadata=None,
    progress=True,
):
    """Train Rainbow against any ``SpeedPolicyEnv`` and save one checkpoint."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Rainbow training requires: uv sync --extra rl") from exc
    from rl.rainbowDQN.dqnAgent import DQNAgent

    config = config or RainbowTrainingConfig()
    config.validate()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state = env.reset()
    agent = DQNAgent(
        env,
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        target_update=config.target_update,
        seed=seed,
        lr=config.learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        frame_skip=config.frame_skip,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay,
        min_epsilon=config.min_epsilon,
        exploration_steps=config.exploration_steps,
        alpha=config.alpha,
        beta=config.beta,
        atom_size=config.atom_size,
        v_min=config.v_min,
        v_max=config.v_max,
        n_step=config.n_step,
        hidden_dim=config.hidden_dim,
        device=device,
        log_dir=None,
    )

    state_history = deque(maxlen=min(config.memory_size, 100_000))
    state_history.append(state.copy())
    episode_return = 0.0
    episode_decisions = 0
    episode_index = 0
    update_count = 0
    losses = []
    episodes = []
    numbered_checkpoints = []

    def update_network(number_of_updates):
        nonlocal update_count
        for _ in range(number_of_updates):
            losses.append(float(agent.update_model()))
            update_count += 1
            if update_count % config.target_update == 0:
                agent._target_soft_update()

    for decision in range(1, config.decisions + 1):
        action = agent.select_action(state)
        next_state, reward, done, info = agent.step(action, config.frame_skip)
        state_history.append(next_state.copy())
        episode_return += float(reward)
        episode_decisions += 1

        progress_fraction = min(decision / config.decisions, 1.0)
        if config.beta_schedule == "legacy":
            # Retained trainer behavior: repeatedly close the remaining gap.
            agent.beta += progress_fraction * (1.0 - agent.beta)
        else:
            agent.beta = config.beta + progress_fraction * (1.0 - config.beta)
        agent.decay_epsilon(decision)

        ready = len(agent.memory) >= max(config.batch_size, config.learning_starts)
        if (
            config.update_schedule == "decision"
            and ready
            and decision % config.train_interval == 0
        ):
            if decision % config.norm_update_interval == 0 or update_count == 0:
                stats = _normalization_stats(state_history)
                agent.dqn.update_norm_stats(stats)
                agent.dqn_target.update_norm_stats(stats)
            update_network(config.gradient_steps)

        state = next_state
        if done:
            episode_index += 1
            record = {
                "episode": episode_index,
                "decision": decision,
                "return": episode_return,
                "decisions": episode_decisions,
                "physics_steps": int(info["physics_steps"]),
                "mean_speed": float(np.mean(env.speed_list)),
                "acceleration": float(
                    env.episode_len / max(int(info["physics_steps"]), 1)
                ),
                "success": bool(info["success"]),
            }
            episodes.append(record)
            if progress:
                print(
                    "episode={episode} decision={decision} success={success} "
                    "return={return:.3f} mean_speed={mean_speed:.3f}".format(**record)
                )
            if config.update_schedule == "episode" and ready:
                stats = _normalization_stats(state_history)
                agent.dqn.update_norm_stats(stats)
                agent.dqn_target.update_norm_stats(stats)
                # The retained SpeedTuning trainer optimized once per decision,
                # batching those updates at the end of each episode.
                update_network(episode_decisions * config.gradient_steps)
            state = env.reset()
            state_history.append(state.copy())
            episode_return = 0.0
            episode_decisions = 0

        if (
            config.checkpoint_interval
            and decision % config.checkpoint_interval == 0
        ):
            numbered_path = _numbered_checkpoint_path(checkpoint_path, decision)
            numbered_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                _checkpoint_payload(
                    agent, env, config, seed, metadata, completed_decisions=decision
                ),
                numbered_path,
            )
            numbered_checkpoints.append(str(numbered_path))

    if len(state_history) >= 2:
        stats = _normalization_stats(state_history)
        agent.dqn.update_norm_stats(stats)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        agent,
        env,
        config,
        seed,
        metadata,
        completed_decisions=config.decisions,
    )
    torch.save(payload, checkpoint_path)

    finite_losses = bool(np.isfinite(losses).all()) if losses else True
    return {
        "checkpoint": str(checkpoint_path),
        "decisions": config.decisions,
        "episodes": len(episodes),
        "successes": sum(int(item["success"]) for item in episodes),
        "updates": update_count,
        "losses_finite": finite_losses,
        "loss_last": losses[-1] if losses else None,
        "numbered_checkpoints": numbered_checkpoints,
        "episode_history": episodes,
    }


def evaluate_rainbow_speed_policy(env, checkpoint_path, episodes=10, device="cpu"):
    """Evaluate a saved speed policy against the supplied base-policy wrapper."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    policy = RainbowSpeedPolicy.load(checkpoint_path, device=device)
    results = [rollout_speed_policy(env, policy) for _ in range(episodes)]
    return {**summarize_rollouts(results), "rollouts": results}
