#!/usr/bin/env python3
"""Run the paper-style fixed-speed sweep and optional adaptive-policy evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import defaults_from_argv  # noqa: E402
from policy_loader import load_speed_policy  # noqa: E402
from policy_speed_env import make_speed_reward  # noqa: E402
from scripts.policy_cli import (  # noqa: E402
    add_base_policy_arguments,
    add_observation_arguments,
    build_speed_env,
    comma_floats,
    comma_ints,
    json_object,
)
from speed_evaluation import (  # noqa: E402
    evaluate_fixed_speed_sweep,
    evaluate_seeded_policy,
    plot_speed_success_tradeoff,
    speed_grid,
)
from speed_policy import RainbowSpeedPolicy, SpeedProfilePolicy  # noqa: E402


def parse_args():
    defaults, config_metadata = defaults_from_argv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON manifest or named preset such as paper-sim.")
    parser.add_argument("--task", choices=("pick_and_place", "insertion", "tea_bag"), default="tea_bag")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=comma_ints)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--success-bonus", type=float, default=100.0)
    parser.add_argument("--speed-weight", type=float, default=0.01)
    parser.add_argument("--speed-power", type=float, default=2.0)
    parser.add_argument("--speed-values", type=comma_floats, default=(1.0, 1.5, 2.0, 2.5, 3.0))
    parser.add_argument("--speed-start", type=float, default=1.0)
    parser.add_argument("--speed-stop", type=float, default=4.5)
    parser.add_argument("--speed-step", type=float, default=0.1)
    parser.add_argument("--episodes-per-speed", type=int, default=100)
    parser.add_argument("--adaptive-episodes", type=int, default=0)
    parser.add_argument(
        "--adaptive-policy",
        choices=("rainbow", "external", "profile"),
    )
    parser.add_argument("--profile", type=comma_floats)
    parser.add_argument("--speed-checkpoint", type=Path)
    parser.add_argument("--speed-policy-loader")
    parser.add_argument("--speed-factory-kwargs", type=json_object, default={})
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    add_base_policy_arguments(parser)
    add_observation_arguments(parser)
    parser.set_defaults(**defaults)
    return parser.parse_args(), config_metadata


def build_adaptive_policy(args):
    if args.adaptive_policy is None:
        return None
    if args.adaptive_policy == "rainbow":
        if args.speed_checkpoint is None:
            raise ValueError("--speed-checkpoint is required for Rainbow evaluation")
        return RainbowSpeedPolicy.load(args.speed_checkpoint, device=args.device)
    if args.adaptive_policy == "profile":
        if args.profile is None:
            raise ValueError("--profile is required for profile evaluation")
        return SpeedProfilePolicy(args.profile)
    if not args.speed_policy_loader:
        raise ValueError("--speed-policy-loader is required for external evaluation")
    return load_speed_policy(
        args.speed_policy_loader,
        checkpoint=args.speed_checkpoint,
        device=args.device,
        factory_kwargs=args.speed_factory_kwargs,
    )


def main():
    args, config_metadata = parse_args()
    try:
        if args.episodes_per_speed <= 0 or args.adaptive_episodes < 0:
            raise ValueError("Episode counts must be positive (or zero for adaptive)")
        adaptive_policy = build_adaptive_policy(args)
        reward_fn = make_speed_reward(
            success_bonus=args.success_bonus,
            speed_weight=args.speed_weight,
            speed_power=args.speed_power,
        )
        args.restore_observation_encoder = isinstance(
            adaptive_policy, RainbowSpeedPolicy
        )
        fixed_seeds = args.seeds or tuple(
            range(args.seed, args.seed + args.episodes_per_speed)
        )

        def env_factory(seed):
            return build_speed_env(args, reward_fn=reward_fn, seed=seed)

        report = {
            "task": args.task,
            "base_policy": args.base_policy,
            "frame_skip": args.frame_skip,
            "metric": "physical_acceleration=episode_len/physics_steps",
            "experiment_config": config_metadata,
            "fixed_speed_sweep": evaluate_fixed_speed_sweep(
                env_factory,
                speed_grid(args.speed_start, args.speed_stop, args.speed_step),
                fixed_seeds,
                frame_skip=args.frame_skip,
            ),
        }
        if adaptive_policy is not None:
            count = args.adaptive_episodes or 2000
            adaptive_seeds = tuple(range(args.seed, args.seed + count))
            report["adaptive_policy"] = evaluate_seeded_policy(
                env_factory,
                adaptive_policy,
                adaptive_seeds,
                frame_skip=(
                    adaptive_policy.frame_skip
                    if isinstance(adaptive_policy, RainbowSpeedPolicy)
                    else args.frame_skip
                ),
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if args.plot is not None:
            plot_speed_success_tradeoff(report, args.plot)
        summary = {
            "output": str(args.output),
            "plot": None if args.plot is None else str(args.plot),
            "fixed_speed_points": len(report["fixed_speed_sweep"]),
            "adaptive_episodes": (
                0 if "adaptive_policy" not in report else report["adaptive_policy"]["episodes"]
            ),
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
