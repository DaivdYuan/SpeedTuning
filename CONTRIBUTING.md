# Contributing

Bug reports and focused pull requests for the supported simulation surface are
welcome. Before opening a pull request:

1. Install Python 3.10 and the development extras with
   `uv sync --extra rl --extra learned --extra test`.
2. Run `uv run pytest -q`.
3. Run `uv run speedtuning-sim` and `uv run speedtuning-check-chunks` when
   changing tasks, policies, interpolation, or physics assets.
4. Run `uv run speedtuning-rainbow-poc` when changing speed-policy learning.
5. Run a two-point `speedtuning-sweep` smoke test when changing metrics,
   decision timing, or experiment manifests.

Please keep real-robot dependencies, private checkpoints, datasets, machine-local
paths, and generated outputs outside this repository. New external policy support
should use the public adapters instead of adding a dependency on another research
repository to the core environment.

By contributing, you agree that your contribution may be distributed under the
license applicable to the directory you modify.
