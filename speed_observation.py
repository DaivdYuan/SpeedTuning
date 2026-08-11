"""Observation encoders for SpeedTuning speed policies.

The speed learner consumes finite one-dimensional feature vectors.  Encoders in
this module turn simulator observation dictionaries into those vectors while
keeping image preprocessing independent from the task policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _finite_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must produce a non-empty finite feature vector")
    return vector


@dataclass
class StateObservationEncoder:
    """Encode proprioception and, optionally, privileged simulator state."""

    include_qpos: bool = True
    include_qvel: bool = True
    include_env_state: bool = True
    requires_images: bool = False

    def __post_init__(self):
        if not (self.include_qpos or self.include_qvel or self.include_env_state):
            raise ValueError("At least one state observation field must be enabled")

    def reset(self):
        return None

    def __call__(self, observation: Mapping[str, Any]) -> np.ndarray:
        fields = []
        for enabled, name in (
            (self.include_env_state, "env_state"),
            (self.include_qpos, "qpos"),
            (self.include_qvel, "qvel"),
        ):
            if enabled:
                if name not in observation:
                    raise ValueError(f"Observation is missing required field {name!r}")
                fields.append(_finite_vector(observation[name], name))
        return np.concatenate(fields).astype(np.float32, copy=False)

    def output_dim(self, env_state_dim: int) -> int:
        return (
            (14 if self.include_qpos else 0)
            + (14 if self.include_qvel else 0)
            + (int(env_state_dim) if self.include_env_state else 0)
        )

    def spec(self) -> dict[str, Any]:
        return {
            "type": "state",
            "include_qpos": self.include_qpos,
            "include_qvel": self.include_qvel,
            "include_env_state": self.include_env_state,
        }


class ResNet18ImageEncoder:
    """Frozen torchvision ResNet-18 image features for one or more cameras."""

    feature_dim = 512

    def __init__(
        self,
        pretrained=True,
        image_size=224,
        device=None,
        initialize_pretrained=True,
    ):
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError(
                "Visual speed observations require: uv sync --extra learned"
            ) from exc

        self.torch = torch
        self.pretrained = bool(pretrained)
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        weights = (
            ResNet18_Weights.DEFAULT
            if self.pretrained and initialize_pretrained
            else None
        )
        self.model = resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        torch = self.torch
        values = np.asarray(images)
        if values.ndim != 4 or values.shape[-1] != 3:
            raise ValueError("Camera images must have shape [camera, height, width, 3]")
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        tensor = tensor.permute(0, 3, 1, 2) / 255.0
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        tensor = (tensor - self.mean) / self.std
        with torch.inference_mode():
            features = self.model(tensor)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def spec(self) -> dict[str, Any]:
        return {
            "type": "resnet18",
            "pretrained": self.pretrained,
            "image_size": self.image_size,
            "feature_dim": self.feature_dim,
        }

    def state_dict(self):
        return {
            key: value.detach().cpu()
            for key, value in self.model.state_dict().items()
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class VisualObservationEncoder:
    """Fuse independent image embeddings with proprioceptive features."""

    requires_images = True

    def __init__(
        self,
        camera_names: Sequence[str] = ("top", "angle", "vis"),
        image_encoder=None,
        pretrained=True,
        image_size=224,
        device=None,
        include_qpos=True,
        include_qvel=True,
        include_env_state=False,
        initialize_pretrained=True,
    ):
        self.camera_names = tuple(camera_names)
        if not self.camera_names:
            raise ValueError("At least one camera is required for visual observations")
        self.image_encoder = image_encoder or ResNet18ImageEncoder(
            pretrained=pretrained,
            image_size=image_size,
            device=device,
            initialize_pretrained=initialize_pretrained,
        )
        self.state_encoder = StateObservationEncoder(
            include_qpos=include_qpos,
            include_qvel=include_qvel,
            include_env_state=include_env_state,
        )

    def reset(self):
        reset = getattr(self.image_encoder, "reset", None)
        if reset is not None:
            reset()

    def __call__(self, observation: Mapping[str, Any]) -> np.ndarray:
        if "images" not in observation:
            raise ValueError("Visual observations require the simulator images field")
        missing = [name for name in self.camera_names if name not in observation["images"]]
        if missing:
            raise ValueError(f"Observation is missing cameras: {', '.join(missing)}")
        images = np.stack(
            [np.asarray(observation["images"][name]) for name in self.camera_names]
        )
        encode = getattr(self.image_encoder, "encode", None) or self.image_encoder
        image_features = _finite_vector(encode(images), "image encoder")
        state_features = self.state_encoder(observation)
        return np.concatenate([state_features, image_features]).astype(
            np.float32, copy=False
        )

    def output_dim(self, env_state_dim: int) -> int:
        feature_dim = getattr(self.image_encoder, "feature_dim", None)
        if feature_dim is None:
            raise AttributeError("External image encoder does not declare feature_dim")
        return self.state_encoder.output_dim(env_state_dim) + len(self.camera_names) * int(
            feature_dim
        )

    def spec(self) -> dict[str, Any]:
        image_spec = getattr(self.image_encoder, "spec", None)
        if image_spec is None:
            image_spec = {"type": type(self.image_encoder).__name__}
        else:
            image_spec = image_spec()
        return {
            "type": "visual",
            "camera_names": list(self.camera_names),
            "state": self.state_encoder.spec(),
            "image_encoder": image_spec,
        }

    def state_dict(self):
        state_dict = getattr(self.image_encoder, "state_dict", None)
        return None if state_dict is None else state_dict()

    def load_state_dict(self, state_dict):
        load = getattr(self.image_encoder, "load_state_dict", None)
        if load is None:
            if state_dict:
                raise ValueError("The configured image encoder cannot load checkpoint state")
            return
        load(state_dict)


class ObservationEncoderAdapter:
    """Validate a user-supplied observation encoder without constraining its model."""

    def __init__(self, encoder):
        if not callable(encoder):
            raise TypeError("An observation encoder must be callable")
        self.encoder = encoder
        self.requires_images = bool(getattr(encoder, "requires_images", False))

    def reset(self):
        reset = getattr(self.encoder, "reset", None)
        if reset is not None:
            reset()

    def __call__(self, observation):
        return _finite_vector(self.encoder(observation), "observation encoder")

    def output_dim(self, env_state_dim):
        output_dim = getattr(self.encoder, "output_dim", None)
        if output_dim is None:
            raise AttributeError("External observation encoder does not declare output_dim")
        return int(output_dim(env_state_dim) if callable(output_dim) else output_dim)

    def spec(self):
        spec = getattr(self.encoder, "spec", None)
        return (
            {"type": type(self.encoder).__name__}
            if spec is None
            else dict(spec() if callable(spec) else spec)
        )

    def state_dict(self):
        state_dict = getattr(self.encoder, "state_dict", None)
        return None if state_dict is None else state_dict()

    def load_state_dict(self, state_dict):
        load = getattr(self.encoder, "load_state_dict", None)
        if load is None:
            if state_dict:
                raise ValueError("External observation encoder cannot load checkpoint state")
            return
        load(state_dict)


def encoder_spec(encoder) -> dict[str, Any]:
    spec = getattr(encoder, "spec", None)
    if spec is None:
        return {"type": type(encoder).__name__}
    return dict(spec() if callable(spec) else spec)
