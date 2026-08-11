"""Small entry-point loader for integrating external policy repositories."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from chunked_policy import ChunkPredictorAdapter
from speed_policy import SpeedPolicyAdapter
from speed_observation import ObservationEncoderAdapter


def load_entrypoint(spec: str):
    """Load ``module.submodule:attribute`` without modifying ``sys.path``."""

    if ":" not in spec:
        raise ValueError("An entry point must use the form 'module.submodule:attribute'")
    module_name, attribute_name = spec.rsplit(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("An entry point must use the form 'module.submodule:attribute'")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as exc:
        raise ValueError(f"{module_name!r} has no attribute {attribute_name!r}") from exc


def _call_factory(factory, available_kwargs):
    if not callable(factory):
        return factory
    signature = inspect.signature(factory)
    accepts_extra = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = {
        name: value
        for name, value in available_kwargs.items()
        if accepts_extra or name in signature.parameters
    }
    return factory(**kwargs)


def load_chunk_predictor(
    entrypoint,
    task_name,
    checkpoint=None,
    device="cpu",
    factory_kwargs=None,
):
    """Instantiate and validate an external joint-action chunk predictor.

    The factory may accept any subset of ``task_name``, ``checkpoint``,
    ``device``, and the keys in ``factory_kwargs``. Its result must be callable
    or define ``predict_chunk(observation)``.
    """

    factory = load_entrypoint(entrypoint)
    available = {
        "task_name": task_name,
        "checkpoint": None if checkpoint is None else Path(checkpoint),
        "device": device,
        **(factory_kwargs or {}),
    }
    return ChunkPredictorAdapter(_call_factory(factory, available))


def load_speed_policy(entrypoint, checkpoint=None, device="cpu", factory_kwargs=None):
    """Instantiate and validate an external physical-speed policy."""

    factory = load_entrypoint(entrypoint)
    available = {
        "checkpoint": None if checkpoint is None else Path(checkpoint),
        "device": device,
        **(factory_kwargs or {}),
    }
    return SpeedPolicyAdapter(_call_factory(factory, available))


def load_observation_encoder(
    entrypoint,
    task_name,
    checkpoint=None,
    device="cpu",
    factory_kwargs=None,
):
    """Instantiate an external speed-observation encoder."""

    factory = load_entrypoint(entrypoint)
    available = {
        "task_name": task_name,
        "checkpoint": None if checkpoint is None else Path(checkpoint),
        "device": device,
        **(factory_kwargs or {}),
    }
    return ObservationEncoderAdapter(_call_factory(factory, available))
