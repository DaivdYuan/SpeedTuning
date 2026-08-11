# External task-policy integration

SpeedTuning treats the task policy and speed policy as independent components.
An external task policy supplies robot action chunks; the speed policy chooses
how quickly those chunks are executed.

This integration is optional. The bundled scripted policies are sufficient for
the from-scratch simulation reproduction.

## Python interface

A task policy must be callable, or define `predict_chunk(observation)`, and
return joint actions with shape `[time, 14]`. Outputs shaped `[1, time, 14]`,
PyTorch tensors, and dictionaries containing `actions`, `action`, or `chunk` are
normalized automatically.

```python
from policy_speed_env import create_speed_env
from speed_policy import FixedSpeedPolicy, rollout_speed_policy


class MyChunkPolicy:
    def reset(self):
        pass

    def predict_chunk(self, observation):
        return model(observation)


env = create_speed_env(
    "insertion",
    chunk_predictor=MyChunkPolicy(),
    seed=0,
)
result = rollout_speed_policy(env, FixedSpeedPolicy(1.5))
```

At each speed-policy decision, the environment requests a fresh receding-horizon
chunk, interpolates it at the selected speed, and holds that speed for the
configured frame-skip block. A short chunk is safely replanned as needed.

`TorchChunkPredictor` in `chunked_policy.py` provides normalization and tensor
handling for ACT-style models.

## CLI factory

Expose a factory in the external policy package:

```python
# my_policy/integration.py
def build_policy(task_name, checkpoint, device):
    return MyChunkPolicy.load(checkpoint, task=task_name, device=device)
```

Install that package in the SpeedTuning environment, then reference the factory
as `module:attribute`:

```bash
uv run speedtuning-eval-speed \
  --task insertion \
  --base-policy external-chunk \
  --chunk-policy my_policy.integration:build_policy \
  --upstream-checkpoint /path/to/upstream.pt \
  --speed-policy fixed --speed 1.5 \
  --episodes 10
```

The factory may accept any subset of `task_name`, `checkpoint`, `device`, and
keys supplied through `--factory-kwargs`.

Training uses the same task-policy flags:

```bash
uv run speedtuning-train-speed \
  --task insertion \
  --base-policy external-chunk \
  --chunk-policy my_policy.integration:build_policy \
  --upstream-checkpoint /path/to/upstream.pt \
  --output outputs/insertion_speed.pt
```

## Retained ACT integration

Install the learned-policy dependencies:

```bash
uv sync --extra rl --extra learned
```

The retained ACT loader accepts checkpoints containing `model_state_dict`,
`policy_config`, and normalization arrays under `stats`. The required statistics
are `qpos_mean`, `qpos_std`, `action_mean`, and `action_std`. They may instead be
provided in an `.npz`, JSON, or legacy pickle file.

```bash
uv run speedtuning-eval-speed \
  --task insertion \
  --base-policy external-chunk \
  --chunk-policy act_integration:build_act_chunk_predictor \
  --upstream-checkpoint /path/to/act.pt \
  --factory-kwargs '{"stats_path":"/path/to/dataset_stats.npz"}' \
  --speed-policy fixed --speed 1.5 \
  --episodes 10
```

ACT checkpoints and legacy pickle statistics require Python pickle loading. Only
load files created locally or obtained from a trusted source. Rainbow speed
checkpoints produced by this repository use PyTorch's restricted weight loader.

## Speed-policy observations

The speed learner supports:

- `--speed-observation state` for selected simulator and proprioceptive fields;
- `--speed-observation visual` for joint state plus encoded camera images;
- `--speed-observation external` for a custom feature encoder.

Use `--no-env-state` to remove privileged simulator state. Visual runs support
pretrained or randomly initialized ResNet-18 encoders, configurable cameras, and
frame stacking. Encoder state and preprocessing metadata are saved in the local
Rainbow checkpoint so evaluation cannot silently change them.

The retained ACT backbone can be used through:

```text
--speed-observation external \
--observation-encoder-loader act_integration:build_act_observation_encoder
```

## Archival paper configuration

The `paper-sim` preset records the visual observation and speed-action settings
recoverable from the paper and retained experiment names. It requires an
external task-policy checkpoint and is provided as an archival starting point,
not an exact numerical reproduction claim.

```bash
uv run speedtuning-train-speed \
  --config paper-sim \
  --task tea_bag \
  --chunk-policy my_policy.integration:build_policy \
  --upstream-checkpoint /path/to/task_policy.pt \
  --output outputs/tea_bag_speed.pt
```

The JSON manifests under `configs/ablations/` cover the observation, reward,
frame-skip, image-encoder, and task-policy ablations retained for research use.
