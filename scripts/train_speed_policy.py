#!/usr/bin/env python3
"""Train a Rainbow speed policy around a scripted or external chunked policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import defaults_from_argv  # noqa: E402
from policy_speed_env import make_speed_reward  # noqa: E402
from scripts.policy_cli import (  # noqa: E402
    add_base_policy_arguments,
    add_observation_arguments,
    build_speed_env,
    comma_floats,
)
from speed_training import (  # noqa: E402
    RainbowTrainingConfig,
    evaluate_rainbow_speed_policy,
    train_rainbow_speed_policy,
)


def parse_args():
    config_defaults, config_metadata = defaults_from_argv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="JSON experiment manifest or named preset such as paper-sim.",
    )
    parser.add_argument("--task", choices=("pick_and_place", "insertion", "tea_bag"), default="tea_bag")
    parser.add_argument("--output", type=Path, default=Path("outputs/speed_policy.pt"))
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON sidecar containing the summary and episode history.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speed-values", type=comma_floats, default=(1.0, 1.5, 2.0, 2.5, 3.0))
    parser.add_argument("--decisions", type=int, default=5_000)
    parser.add_argument("--memory-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--train-interval", type=int, default=1)
    parser.add_argument(
        "--update-schedule",
        choices=("decision", "episode"),
        default="decision",
        help="Optimize after each decision or batch equivalent updates at episode end.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Also save a numbered checkpoint every N decisions (0 disables it).",
    )
    parser.add_argument("--target-update", type=int, default=50)
    parser.add_argument("--norm-update-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--min-epsilon", type=float, default=0.1)
    parser.add_argument("--exploration-steps", type=int, default=2000)
    parser.add_argument("--per-alpha", type=float, default=0.2)
    parser.add_argument("--per-beta", type=float, default=0.6)
    parser.add_argument(
        "--beta-schedule",
        choices=("linear", "legacy"),
        default="linear",
        help="Importance-sampling schedule; legacy matches the retained trainer.",
    )
    parser.add_argument("--atom-size", type=int, default=121)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=120.0)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--success-bonus", type=float, default=100.0)
    parser.add_argument("--speed-weight", type=float, default=0.01)
    parser.add_argument("--speed-power", type=float, default=2.0)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode progress while retaining the final JSON summary.",
    )
    add_base_policy_arguments(parser)
    add_observation_arguments(parser)
    parser.set_defaults(**config_defaults)
    return parser.parse_args(), config_metadata


def main():
    args, config_metadata = parse_args()
    reward_fn = make_speed_reward(
        success_bonus=args.success_bonus,
        speed_weight=args.speed_weight,
        speed_power=args.speed_power,
    )
    try:
        env = build_speed_env(args, reward_fn=reward_fn)
        config = RainbowTrainingConfig(
            decisions=args.decisions,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            frame_skip=args.frame_skip,
            gradient_steps=args.gradient_steps,
            hidden_dim=args.hidden_dim,
            train_interval=args.train_interval,
            update_schedule=args.update_schedule,
            checkpoint_interval=args.checkpoint_interval,
            target_update=args.target_update,
            norm_update_interval=args.norm_update_interval,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            min_epsilon=args.min_epsilon,
            exploration_steps=args.exploration_steps,
            alpha=args.per_alpha,
            beta=args.per_beta,
            beta_schedule=args.beta_schedule,
            atom_size=args.atom_size,
            v_min=args.v_min,
            v_max=args.v_max,
            n_step=args.n_step,
        )
        result = train_rainbow_speed_policy(
            env,
            args.output,
            config=config,
            seed=args.seed,
            device=args.device,
            progress=not args.quiet,
            metadata={
                "task": args.task,
                "base_policy": args.base_policy,
                "randomize_object_pose": args.randomize_object_pose,
                "experiment_config": config_metadata,
            },
        )
        summary = {key: value for key, value in result.items() if key != "episode_history"}
        if args.eval_episodes:
            summary["evaluation"] = evaluate_rainbow_speed_policy(
                env,
                args.output,
                episodes=args.eval_episodes,
                device=args.device,
            )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(
                    {
                        "summary": summary,
                        "episode_history": result["episode_history"],
                        "experiment_config": config_metadata,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            summary["report"] = str(args.report)
        print(json.dumps(summary, sort_keys=True))
        return 0 if result["losses_finite"] else 1
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if "env" in locals():
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
