"""JSON experiment manifests shared by training, evaluation, and sweeps."""

from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path


CONFIG_ROOT = Path(str(resources.files("configs")))
NAMED_CONFIGS = {
    "paper-sim": CONFIG_ROOT / "paper_sim.json",
    "scripted-tea-bag": CONFIG_ROOT / "scripted_tea_bag.json",
    "scripted-tea-bag-randomized": (
        CONFIG_ROOT / "scripted_tea_bag_randomized.json"
    ),
    "scripted-pick-and-place": (
        CONFIG_ROOT / "scripted_pick_and_place.json"
    ),
    "scripted-insertion": CONFIG_ROOT / "scripted_insertion.json",
}
CONFIG_SECTIONS = (
    "base_policy",
    "environment",
    "observation",
    "reward",
    "training",
    "evaluation",
)


def resolve_config_path(value):
    if value is None:
        return None
    if value in NAMED_CONFIGS:
        return NAMED_CONFIGS[value]
    return Path(value)


def load_experiment_config(value):
    path = resolve_config_path(value)
    if path is None:
        return {}, None
    if not path.exists():
        raise ValueError(f"Experiment config does not exist: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must contain a JSON object")
    inherited_defaults = {}
    inherited_manifest = None
    if payload.get("inherits"):
        inherited_defaults, inherited_manifest = load_experiment_config(
            payload["inherits"]
        )
    defaults = dict(inherited_defaults)
    for section in CONFIG_SECTIONS:
        values = payload.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Experiment config section {section!r} must be an object")
        defaults.update(values)
    return defaults, {
        "path": str(path),
        "manifest": payload,
        "inherits": inherited_manifest,
    }


def defaults_from_argv(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    known, _ = parser.parse_known_args(argv)
    return load_experiment_config(known.config)
