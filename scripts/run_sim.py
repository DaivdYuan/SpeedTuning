#!/usr/bin/env python3
"""Run the recovered SpeedTuning scripted simulation tasks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ee_sim_env import make_ee_sim_env  # noqa: E402
from scripted_policy import make_scripted_policy  # noqa: E402
from sim_tasks import TASK_SPECS, normalize_task_name  # noqa: E402


def run_task(task_name, speed=1.0, seed=0, video_path=None):
    """Run one open-loop scripted rollout and return a JSON-safe summary."""

    task_name = normalize_task_name(task_name)
    spec = TASK_SPECS[task_name]
    render_images = video_path is not None
    env = make_ee_sim_env(task_name, render_images=render_images, seed=seed)
    timestep = env.reset()
    policy = make_scripted_policy(task_name)
    rewards = []
    frames = []

    num_steps = math.ceil(spec.episode_len / speed)
    for _ in range(num_steps):
        timestep = env.step(policy(timestep, step_inc=speed))
        rewards.append(int(timestep.reward or 0))
        if render_images:
            frames.append(timestep.observation["images"]["angle"])

    max_reward = max(rewards, default=0)
    result = {
        "task": task_name,
        "success": max_reward == env.task.max_reward,
        "max_reward": max_reward,
        "target_reward": env.task.max_reward,
        "return": sum(rewards),
        "steps": num_steps,
        "speed": speed,
        "seed": seed,
    }

    if video_path is not None:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "Video export requires: uv sync --extra video"
            ) from exc
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=max(1, round(50 / speed)))
        result["video"] = str(video_path)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run recovered pick-and-place, insertion, and tea-bag simulations."
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Task to run; repeat the flag for multiple tasks (default: all).",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Waypoint time increment.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Optional directory for angle-camera MP4 rollouts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")
    tasks = args.task or list(TASK_SPECS)
    results = []
    for requested_task in tasks:
        task_name = normalize_task_name(requested_task)
        video_path = None
        if args.video_dir is not None:
            video_path = args.video_dir / f"{task_name}.mp4"
        result = run_task(
            task_name,
            speed=args.speed,
            seed=args.seed,
            video_path=video_path,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True))
    return 0 if all(result["success"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
