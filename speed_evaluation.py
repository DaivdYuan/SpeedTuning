"""Deterministic evaluation and plotting helpers for SpeedTuning experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from speed_policy import FixedSpeedPolicy, rollout_speed_policy, summarize_rollouts


def speed_grid(start, stop, step):
    start, stop, step = float(start), float(stop), float(step)
    if step <= 0 or stop < start:
        raise ValueError("Speed grid requires step > 0 and stop >= start")
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    values = start + np.arange(count, dtype=np.float64) * step
    if values[-1] < stop - 1e-9:
        values = np.append(values, stop)
    return tuple(float(np.round(value, 10)) for value in values)


def evaluate_seeded_policy(env_factory, policy, seeds, frame_skip=10):
    rollouts = []
    for seed in seeds:
        env = env_factory(int(seed))
        try:
            result = rollout_speed_policy(
                env,
                policy,
                frame_skip=frame_skip,
            )
            result["seed"] = int(seed)
            rollouts.append(result)
        finally:
            env.close()
    return {**summarize_rollouts(rollouts), "seeds": list(seeds), "rollouts": rollouts}


def evaluate_fixed_speed_sweep(env_factory, speeds, seeds, frame_skip=10):
    points = []
    for speed in speeds:
        result = evaluate_seeded_policy(
            env_factory,
            FixedSpeedPolicy(speed),
            seeds,
            frame_skip=frame_skip,
        )
        points.append({"fixed_speed": float(speed), **result})
    return points


def plot_speed_success_tradeoff(report, output_path):
    """Write a compact success-versus-physical-acceleration plot."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires: uv sync --extra evaluation") from exc

    baseline = report.get("fixed_speed_sweep", [])
    if not baseline:
        raise ValueError("Report has no fixed-speed sweep to plot")
    figure, axis = plt.subplots(figsize=(4.4, 3.2))
    axis.plot(
        [point["mean_acceleration"] for point in baseline],
        [point["success_rate"] for point in baseline],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Fixed speed",
    )
    adaptive = report.get("adaptive_policy")
    if adaptive is not None:
        axis.scatter(
            [adaptive["mean_acceleration"]],
            [adaptive["success_rate"]],
            marker="*",
            s=90,
            label="Speed policy",
            zorder=3,
        )
    axis.set_xlabel("Physical acceleration")
    axis.set_ylabel("Success rate")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path
