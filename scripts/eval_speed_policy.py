#!/usr/bin/env python3
"""Evaluate fixed, profiled, Rainbow, or external speed policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import defaults_from_argv  # noqa: E402
from policy_speed_env import make_speed_reward  # noqa: E402
from policy_loader import load_speed_policy  # noqa: E402
from scripts.policy_cli import (  # noqa: E402
    add_base_policy_arguments,
    add_observation_arguments,
    build_speed_env,
    comma_floats,
    comma_ints,
    json_object,
)
from speed_policy import (  # noqa: E402
    FixedSpeedPolicy,
    RainbowSpeedPolicy,
    SpeedProfilePolicy,
    rollout_speed_policy,
    summarize_rollouts,
)


def parse_args():
    config_defaults, config_metadata = defaults_from_argv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON manifest or named preset such as paper-sim.")
    parser.add_argument("--task", choices=("pick_and_place", "insertion", "tea_bag"), default="tea_bag")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--seeds",
        type=comma_ints,
        help="Explicit episode seeds; when set, one rollout is run per seed.",
    )
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--success-bonus", type=float, default=100.0)
    parser.add_argument("--speed-weight", type=float, default=0.01)
    parser.add_argument("--speed-power", type=float, default=2.0)
    parser.add_argument("--speed-values", type=comma_floats, default=(1.0, 1.5, 2.0, 2.5, 3.0))
    parser.add_argument(
        "--speed-policy",
        choices=("fixed", "profile", "rainbow", "external"),
        default="fixed",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--profile", type=comma_floats)
    parser.add_argument("--speed-checkpoint", type=Path)
    parser.add_argument("--speed-policy-loader", help="External speed factory as module:attribute.")
    parser.add_argument("--speed-factory-kwargs", type=json_object, default={})
    parser.add_argument("--video", type=Path)
    add_base_policy_arguments(parser)
    add_observation_arguments(parser)
    parser.set_defaults(**config_defaults)
    return parser.parse_args(), config_metadata


def build_speed_policy(args):
    if args.speed_policy == "fixed":
        return FixedSpeedPolicy(args.speed)
    if args.speed_policy == "profile":
        if args.profile is None:
            raise ValueError("--profile is required for a profile speed policy")
        return SpeedProfilePolicy(args.profile)
    if args.speed_policy == "rainbow":
        if args.speed_checkpoint is None:
            raise ValueError("--speed-checkpoint is required for a Rainbow speed policy")
        return RainbowSpeedPolicy.load(args.speed_checkpoint, device=args.device)
    if not args.speed_policy_loader:
        raise ValueError("--speed-policy-loader is required for an external speed policy")
    return load_speed_policy(
        args.speed_policy_loader,
        checkpoint=args.speed_checkpoint,
        device=args.device,
        factory_kwargs=args.speed_factory_kwargs,
    )


def main():
    args, config_metadata = parse_args()
    if args.episodes <= 0:
        print("error: --episodes must be positive", file=sys.stderr)
        return 2
    try:
        policy = build_speed_policy(args)
        reward_fn = make_speed_reward(
            success_bonus=args.success_bonus,
            speed_weight=args.speed_weight,
            speed_power=args.speed_power,
        )
        args.restore_observation_encoder = isinstance(policy, RainbowSpeedPolicy)
        seeds = args.seeds or tuple(range(args.seed, args.seed + args.episodes))
        rollouts = []
        for index, seed in enumerate(seeds):
            env = build_speed_env(
                args,
                reward_fn=reward_fn,
                video_path=args.video if index == 0 else None,
                seed=seed,
            )
            try:
                rollout = rollout_speed_policy(
                    env,
                    policy,
                    frame_skip=(
                        policy.frame_skip
                        if isinstance(policy, RainbowSpeedPolicy)
                        else args.frame_skip
                    ),
                )
                rollout["seed"] = seed
                rollouts.append(rollout)
            finally:
                env.close()
        result = {
            "task": args.task,
            "base_policy": args.base_policy,
            "speed_policy": args.speed_policy,
            "seeds": list(seeds),
            "experiment_config": config_metadata,
            **summarize_rollouts(rollouts),
            "rollouts": rollouts,
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
