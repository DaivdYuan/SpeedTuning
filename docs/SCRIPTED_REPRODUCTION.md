# Scripted-policy simulation reproduction

This guide reproduces the simulation methodology from *SpeedTuning: Speeding Up
Policy Execution with Lightweight Reinforcement Learning* using the three task
policies included in the repository.

No pretrained model is required. Rainbow DQN learns the execution-speed policy
from scratch, and all generated checkpoints and reports stay under the ignored
`outputs/` directory.

## Method

The public presets use the retained scripted-policy training recipe:

- state observations containing object state, joint position, and joint velocity;
- one speed decision every 10 MuJoCo physics steps;
- reward `0.01 * speed**2` per physics step and 100 for terminal success;
- categorical dueling Double DQN with NoisyNet, prioritized replay, and 3-step
  returns;
- one optimizer update per speed decision, applied at episode boundaries;
- 100,000 speed decisions with periodic local checkpoints.

Task-specific presets expose appropriate discrete speed ranges:

| Task | Preset | Speed range |
| --- | --- | --- |
| Pick-and-place | `scripted-pick-and-place` | 1.0x-4.5x |
| Insertion | `scripted-insertion` | 1.0x-3.0x |
| Tea bag | `scripted-tea-bag` | 1.0x-3.0x |
| Tea bag, randomized poses | `scripted-tea-bag-randomized` | 1.0x-3.0x |

The complete hyperparameters are versioned in `configs/`. Command-line options
can override them for ablations or short smoke runs.

## Install and verify

```bash
uv sync --extra rl --extra test
MUJOCO_GL=egl uv run pytest -q
MUJOCO_GL=egl uv run speedtuning-sim
```

`MUJOCO_GL=egl` enables headless rendering on Linux. Depending on the platform,
DM Control may select a suitable backend without it.

## Train

Run each task separately because every task has its own speed policy:

```bash
MUJOCO_GL=egl uv run speedtuning-train-speed \
  --config scripted-pick-and-place --task pick_and_place \
  --output outputs/pick_and_place_speed.pt \
  --report outputs/pick_and_place_speed.training.json --quiet

MUJOCO_GL=egl uv run speedtuning-train-speed \
  --config scripted-insertion --task insertion \
  --output outputs/insertion_speed.pt \
  --report outputs/insertion_speed.training.json --quiet

MUJOCO_GL=egl uv run speedtuning-train-speed \
  --config scripted-tea-bag --task tea_bag \
  --output outputs/tea_bag_speed.pt \
  --report outputs/tea_bag_speed.training.json --quiet
```

Training defaults to CPU. Add `--device cuda` on a CUDA machine. Hardware changes
runtime, but it does not change the simulator protocol or reported acceleration
metric.

The presets write a snapshot every 10,000 decisions. Checkpoint metadata records
the task, pose protocol, speed actions, observation preprocessing, and training
configuration. Evaluation rejects an incompatible task or protocol.

## Evaluate

Evaluate held-out initial poses by choosing a starting seed and episode count:

```bash
MUJOCO_GL=egl uv run speedtuning-eval-speed \
  --config scripted-pick-and-place --task pick_and_place \
  --speed-policy rainbow \
  --speed-checkpoint outputs/pick_and_place_speed.pt \
  --seed 100 --episodes 100
```

Compare against a fixed speed on the same seeds:

```bash
MUJOCO_GL=egl uv run speedtuning-eval-speed \
  --config scripted-pick-and-place --task pick_and_place \
  --speed-policy fixed --speed 3.856 \
  --seed 100 --episodes 100
```

For a complete fixed-speed frontier:

```bash
MUJOCO_GL=egl uv run speedtuning-sweep \
  --config scripted-pick-and-place --task pick_and_place \
  --speed-start 1.0 --speed-stop 4.5 --speed-step 0.25 \
  --episodes-per-speed 100 \
  --output outputs/pick_and_place_sweep.json
```

Physical acceleration is the nominal task horizon divided by executed MuJoCo
steps. Failed rollouts remain in the success-rate and acceleration summaries.

## Tea-bag protocols

The retained tea-bag environment has one fixed initial object pose. It is useful
for matching the historical scripted-policy experiment, but repeated seeds are
intentionally identical.

Use `scripted-tea-bag-randomized` when a distribution of initial poses is needed:

```bash
MUJOCO_GL=egl uv run speedtuning-train-speed \
  --config scripted-tea-bag-randomized --task tea_bag --seed 1 \
  --output outputs/tea_bag_randomized_speed.pt --quiet

MUJOCO_GL=egl uv run speedtuning-eval-speed \
  --config scripted-tea-bag-randomized --task tea_bag \
  --speed-policy rainbow \
  --speed-checkpoint outputs/tea_bag_randomized_speed.pt \
  --seed 100 --episodes 100
```

Do not evaluate a fixed-pose checkpoint under the randomized protocol or vice
versa; the observation distribution and normalization differ.

## Reference results

One seeded run using the protocol above produced:

| Protocol | Learned speed | Matched fixed speed |
| --- | --- | --- |
| Pick-and-place, seeds 100-199 | 98% at 3.856x | 66% at 3.846x |
| Insertion, seeds 100-199 | 97% at 2.387x | 52% at 2.381x |
| Tea bag, randomized seeds 100-199 | 78% at 2.077x | 24% at 2.075x |

The machine-readable record is
[`benchmarks/scripted_results.json`](../benchmarks/scripted_results.json).
Reinforcement learning is stochastic, so compare reruns at the level of success
and acceleration trends rather than exact decimal equality.

## Scope

This workflow reproduces speed-policy learning, temporal acceleration, sparse
success, fixed-speed baselines, and seeded simulation evaluation. It does not
include real-robot execution, datasets, or learned task-policy checkpoints.

See [External task-policy integration](EXTERNAL_POLICIES.md) to wrap ACT or
another action-chunk policy.
