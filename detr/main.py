"""Construct the retained ACT/DETR models from a dictionary configuration."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from .models import build_ACT_model, build_CNNMLP_model


MODEL_DEFAULTS = {
    "lr": 1e-4,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "backbone": "resnet18",
    "pretrained_backbone": False,
    "dilation": False,
    "position_embedding": "sine",
    "camera_names": [],
    "enc_layers": 4,
    "dec_layers": 6,
    "dim_feedforward": 2048,
    "hidden_dim": 256,
    "dropout": 0.1,
    "nheads": 8,
    "num_queries": 400,
    "pre_norm": False,
    "masks": False,
}


def _model_args(overrides):
    values = {**MODEL_DEFAULTS, **overrides}
    return SimpleNamespace(**values)


def _optimizer(model, args):
    parameter_groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if "backbone" not in name and parameter.requires_grad
            ]
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if "backbone" in name and parameter.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    return torch.optim.AdamW(
        parameter_groups, lr=args.lr, weight_decay=args.weight_decay
    )


def _device(args):
    return torch.device(
        getattr(args, "device", None)
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )


def build_ACT_model_and_optimizer(args_override):
    args = _model_args(args_override)
    model = build_ACT_model(args).to(_device(args))
    return model, _optimizer(model, args)


def build_CNNMLP_model_and_optimizer(args_override):
    args = _model_args(args_override)
    model = build_CNNMLP_model(args).to(_device(args))
    return model, _optimizer(model, args)
