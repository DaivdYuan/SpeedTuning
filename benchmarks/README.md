# Reference results

[`scripted_results.json`](scripted_results.json) records one seeded run of the
from-scratch scripted-policy protocol. The record is hardware-neutral: it
reports simulator success and physical acceleration, not training time.

| Protocol | Learned speed | Matched fixed speed |
| --- | --- | --- |
| Pick-and-place | 98% at 3.856x | 66% at 3.846x |
| Insertion | 97% at 2.387x | 52% at 2.381x |
| Tea bag, randomized poses | 78% at 2.077x | 24% at 2.075x |

The JSON includes the preset, training seed, and episode count needed to
interpret each result. Training is stochastic, so these values
are reference points rather than exact-decimal guarantees.

See the [reproduction guide](../docs/SCRIPTED_REPRODUCTION.md) for commands and
protocol details. No trained checkpoint is distributed with the repository.
