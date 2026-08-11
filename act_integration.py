"""Load retained ACT checkpoints through the public chunk-policy interface."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from chunked_policy import TorchChunkPredictor


REQUIRED_STATS = ("qpos_mean", "qpos_std", "action_mean", "action_std")


def _load_mapping(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as values:
            return {key: values[key] for key in values.files}
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as stream:
            return pickle.load(stream)
    raise ValueError("ACT stats must use .npz, .json, .pkl, or .pickle")


def _checkpoint_parts(checkpoint, device):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("ACT integration requires: uv sync --extra learned") from exc
    if checkpoint is None:
        raise ValueError("An ACT checkpoint path is required")
    # ACT checkpoints may include NumPy normalization arrays and legacy config
    # objects. They therefore require pickle loading and must come from a
    # trusted source (normally the user's own task-policy training run).
    payload = torch.load(
        Path(checkpoint), map_location=device, weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError("ACT checkpoint must contain a state dictionary or payload")
    for key in ("model_state_dict", "policy_state_dict", "state_dict"):
        if key in payload:
            return payload, payload[key]
    # A raw torch state dictionary maps names to tensors.
    if payload and all(isinstance(key, str) for key in payload):
        return {}, payload
    raise ValueError("ACT checkpoint does not contain recognizable model weights")


def _resolve_config(payload, policy_config, camera_names, device):
    config = dict(payload.get("policy_config", {}))
    config.update(policy_config or {})
    if camera_names is not None:
        config["camera_names"] = list(camera_names)
    if not config.get("camera_names"):
        raise ValueError("ACT policy_config must provide camera_names")
    required = ("num_queries", "hidden_dim", "dim_feedforward", "enc_layers", "dec_layers", "nheads")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"ACT policy_config is missing: {', '.join(missing)}")
    config.setdefault("lr", 1e-4)
    config.setdefault("lr_backbone", 0.0)
    config.setdefault("kl_weight", 10.0)
    config.setdefault("backbone", "resnet18")
    config.setdefault("pretrained_backbone", False)
    config["device"] = device
    return config


def _resolve_stats(payload, stats_path):
    stats = dict(payload.get("stats", {}))
    if stats_path is not None:
        stats.update(_load_mapping(stats_path))
    missing = [key for key in REQUIRED_STATS if key not in stats]
    if missing:
        raise ValueError(
            "ACT normalization stats are missing: " + ", ".join(missing)
        )
    return {key: np.asarray(stats[key], dtype=np.float32) for key in REQUIRED_STATS}


def load_act_policy(
    checkpoint,
    device="cpu",
    stats_path=None,
    policy_config=None,
    camera_names=None,
    strict=True,
):
    """Return the ACT module, resolved configuration, and normalization stats."""

    from policy import ACTPolicy

    payload, state_dict = _checkpoint_parts(checkpoint, device)
    config = _resolve_config(payload, policy_config, camera_names, device)
    stats = _resolve_stats(payload, stats_path)
    model = ACTPolicy(config)
    incompatible = model.load_state_dict(state_dict, strict=bool(strict))
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        # Keep the information available to callers without printing during imports.
        model.checkpoint_incompatibilities = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
    model.eval()
    return model, config, stats


def build_act_chunk_predictor(
    task_name,
    checkpoint,
    device="cpu",
    stats_path=None,
    policy_config=None,
    camera_names=None,
    strict=True,
):
    """Factory usable as ``act_integration:build_act_chunk_predictor``."""

    del task_name
    model, config, stats = load_act_policy(
        checkpoint=checkpoint,
        device=device,
        stats_path=stats_path,
        policy_config=policy_config,
        camera_names=camera_names,
        strict=strict,
    )
    return TorchChunkPredictor(
        model=model,
        camera_names=config["camera_names"],
        qpos_mean=stats["qpos_mean"],
        qpos_std=stats["qpos_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
        device=device,
    )


class ACTBackboneObservationEncoder:
    """Use a supplied ACT task policy's ResNet backbone for speed features."""

    requires_images = True

    def __init__(self, model, camera_names, include_qvel=True, device="cpu"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.camera_names = tuple(camera_names)
        self.include_qvel = bool(include_qvel)
        self.backbone = model.model.backbones[0]
        self.backbone.to(self.device).eval()
        self.feature_dim = int(self.backbone.num_channels)
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

    def reset(self):
        return None

    def __call__(self, observation):
        torch = self.torch
        if "images" not in observation:
            raise ValueError("ACT-backbone speed observations require images")
        images = np.stack(
            [observation["images"][name] for name in self.camera_names]
        ).transpose(0, 3, 1, 2)
        tensor = torch.as_tensor(images, dtype=torch.float32, device=self.device) / 255.0
        tensor = (tensor - self.mean) / self.std
        features = []
        with torch.inference_mode():
            for image in tensor:
                backbone_features, _ = self.backbone(image.unsqueeze(0))
                feature_map = backbone_features[-1]
                features.append(feature_map.mean(dim=(2, 3)).squeeze(0))
        proprioception = [np.asarray(observation["qpos"], dtype=np.float32)]
        if self.include_qvel:
            proprioception.append(np.asarray(observation["qvel"], dtype=np.float32))
        return np.concatenate(
            proprioception
            + [torch.cat(features).detach().cpu().numpy().astype(np.float32)]
        )

    def output_dim(self, env_state_dim):
        del env_state_dim
        return 14 + (14 if self.include_qvel else 0) + len(self.camera_names) * self.feature_dim

    def spec(self):
        return {
            "type": "act_backbone",
            "camera_names": list(self.camera_names),
            "include_qpos": True,
            "include_qvel": self.include_qvel,
            "include_env_state": False,
            "feature_dim": self.feature_dim,
        }

    def state_dict(self):
        return {
            key: value.detach().cpu()
            for key, value in self.backbone.state_dict().items()
        }

    def load_state_dict(self, state_dict):
        self.backbone.load_state_dict(state_dict)


def build_act_observation_encoder(
    task_name,
    checkpoint,
    device="cpu",
    stats_path=None,
    policy_config=None,
    camera_names=None,
    include_qvel=True,
    strict=True,
):
    """Factory for the task-policy image-encoder ablation."""

    del task_name
    model, config, _ = load_act_policy(
        checkpoint=checkpoint,
        device=device,
        stats_path=stats_path,
        policy_config=policy_config,
        camera_names=camera_names,
        strict=strict,
    )
    return ACTBackboneObservationEncoder(
        model,
        config["camera_names"],
        include_qvel=include_qvel,
        device=device,
    )
