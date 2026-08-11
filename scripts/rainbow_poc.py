#!/usr/bin/env python3
"""Run a small Rainbow DQN optimization proof on the tea-bag speed task."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError as exc:
    raise SystemExit("Rainbow POC requires: uv sync --extra rl") from exc

from policy_speed_env import create_speed_env  # noqa: E402
from rl.rainbowDQN.dqnAgent import DQNAgent  # noqa: E402


def speed_reward(speed, done, success):
    reward = speed**2 / 100.0
    if done and success:
        reward += 100.0
    return reward


def run_poc(seed=0, min_transitions=64, updates=8):
    """Collect real simulator transitions and perform Rainbow optimizer steps."""

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env = create_speed_env(
        task_name="tea_bag",
        reward_fn=speed_reward,
        seed=seed,
    )
    agent = DQNAgent(
        env,
        memory_size=2048,
        batch_size=16,
        target_update=4,
        seed=seed,
        lr=3e-4,
        gamma=0.97,
        frame_skip=5,
        epsilon=1.0,
        exploration_steps=1000,
        min_epsilon=0.1,
        atom_size=51,
        v_min=0.0,
        v_max=120.0,
        n_step=3,
        hidden_dim=64,
        device="cpu",
        log_dir=None,
    )

    episodes = 0
    successes = 0
    actions_seen = set()
    rng = np.random.RandomState(seed)
    decision_index = 0
    while len(agent.memory) < min_transitions:
        state = env.reset()
        done = False
        info = {"success": False}
        while not done:
            # Exercise every action during the initial approach, then collect a
            # predominantly safe 1.0x/1.5x rollout so terminal success is present.
            if decision_index < env.action_space:
                action = decision_index
            else:
                action = int(rng.choice(2, p=[0.8, 0.2]))
            decision_index += 1
            actions_seen.add(action)
            agent.transition = [state, action]
            state, _, done, info = agent.step(action, frame_skip=agent.frame_skip)
        episodes += 1
        successes += int(info["success"])

    states = agent.memory.obs_buf[: len(agent.memory)]
    norm_stats = {
        "states_mean": states.mean(axis=0),
        "states_std": states.std(axis=0),
    }
    agent.dqn.update_norm_stats(norm_stats)
    agent.dqn_target.update_norm_stats(norm_stats)

    parameters_before = [parameter.detach().clone() for parameter in agent.dqn.parameters()]
    losses = []
    for update in range(updates):
        loss = agent.update_model()
        losses.append(loss)
        if (update + 1) % agent.target_update == 0:
            agent._target_soft_update()

    parameter_delta = sum(
        (before - after.detach()).abs().sum().item()
        for before, after in zip(parameters_before, agent.dqn.parameters())
    )
    probe = torch.as_tensor(states[:4], dtype=torch.float32)
    with torch.inference_mode():
        q_values = agent.dqn(probe)
        distributions = agent.dqn.dist(probe)

    result = {
        "task": "tea_bag",
        "episodes": episodes,
        "successful_episodes": successes,
        "replay_transitions": len(agent.memory),
        "actions_seen": sorted(actions_seen),
        "updates": updates,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "losses_finite": bool(np.isfinite(losses).all()),
        "parameter_delta": parameter_delta,
        "q_shape": list(q_values.shape),
        "distributions_normalized": bool(
            torch.allclose(
                distributions.sum(dim=-1),
                torch.ones_like(distributions.sum(dim=-1)),
                atol=1e-5,
            )
        ),
    }
    result["passed"] = bool(
        result["losses_finite"]
        and parameter_delta > 0
        and result["distributions_normalized"]
        and len(agent.memory) >= min_transitions
    )
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-transitions", type=int, default=64)
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()
    result = run_poc(args.seed, args.min_transitions, args.updates)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
