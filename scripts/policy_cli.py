"""Shared command-line helpers for base-policy and speed-policy tools."""

from __future__ import annotations

import json

from policy_loader import load_chunk_predictor, load_observation_encoder
from policy_speed_env import create_recorded_chunk_speed_env, create_speed_env
from speed_observation import StateObservationEncoder, VisualObservationEncoder


def comma_floats(value):
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("Expected comma-separated numbers") from exc
    if not values:
        raise ValueError("Expected at least one number")
    return values


def comma_strings(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def comma_ints(value):
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("Expected comma-separated integers") from exc
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def json_object(value):
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Factory kwargs must be a JSON object")
    return parsed


def add_base_policy_arguments(parser):
    parser.add_argument(
        "--base-policy",
        choices=("scripted", "recorded-chunk", "external-chunk"),
        default="scripted",
        help="Robot policy wrapped by the speed controller.",
    )
    parser.add_argument(
        "--chunk-policy",
        help="External chunk factory as module:attribute (required for external-chunk).",
    )
    parser.add_argument("--upstream-checkpoint")
    parser.add_argument("--factory-kwargs", type=json_object, default={})
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument(
        "--no-policy-images",
        action="store_true",
        help="Do not render cameras for an external policy that only uses state.",
    )
    parser.add_argument(
        "--randomize-object-pose",
        action="store_true",
        help="Sample a new tea-bag pose on every reset (cube/insertion already vary).",
    )


def add_observation_arguments(parser):
    parser.add_argument(
        "--speed-observation",
        choices=("state", "visual", "external"),
        default="state",
        help="Input representation used by the speed policy.",
    )
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument(
        "--camera-names",
        type=comma_strings,
        default=("top", "angle", "vis"),
    )
    parser.add_argument(
        "--image-encoder",
        choices=("resnet18-pretrained", "resnet18-random"),
        default="resnet18-pretrained",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--observation-encoder-loader")
    parser.add_argument("--observation-factory-kwargs", type=json_object, default={})
    parser.add_argument(
        "--include-env-state", dest="include_env_state", action="store_true", default=True
    )
    parser.add_argument("--no-env-state", dest="include_env_state", action="store_false")
    parser.add_argument(
        "--include-qpos", dest="include_qpos", action="store_true", default=True
    )
    parser.add_argument("--no-qpos", dest="include_qpos", action="store_false")
    parser.add_argument(
        "--include-qvel", dest="include_qvel", action="store_true", default=True
    )
    parser.add_argument("--no-qvel", dest="include_qvel", action="store_false")


def build_observation_encoder(args):
    if args.speed_observation == "state":
        return StateObservationEncoder(
            include_qpos=args.include_qpos,
            include_qvel=args.include_qvel,
            include_env_state=args.include_env_state,
        )
    if args.speed_observation == "visual":
        return VisualObservationEncoder(
            camera_names=args.camera_names,
            pretrained=args.image_encoder == "resnet18-pretrained",
            image_size=args.image_size,
            device=args.device,
            include_qpos=args.include_qpos,
            include_qvel=args.include_qvel,
            include_env_state=args.include_env_state,
            initialize_pretrained=not getattr(
                args, "restore_observation_encoder", False
            ),
        )
    if not args.observation_encoder_loader:
        raise ValueError(
            "--observation-encoder-loader is required for external observations"
        )
    return load_observation_encoder(
        args.observation_encoder_loader,
        task_name=args.task,
        checkpoint=args.upstream_checkpoint,
        device=args.device,
        factory_kwargs=args.observation_factory_kwargs,
    )


def build_speed_env(args, reward_fn=None, video_path=None, seed=None):
    observation_encoder = build_observation_encoder(args)
    common = {
        "task_name": args.task,
        "reward_fn": reward_fn,
        "seed": args.seed if seed is None else seed,
        "speed_values": args.speed_values,
        "observation_encoder": observation_encoder,
        "frame_stack": args.frame_stack,
        "decision_frame_skip": args.frame_skip,
        "randomize_object_pose": args.randomize_object_pose,
    }
    if video_path is not None:
        common.update(save_video=True, video_path=video_path)

    if args.base_policy == "scripted":
        return create_speed_env(**common)
    if args.base_policy == "recorded-chunk":
        return create_recorded_chunk_speed_env(
            chunk_size=args.chunk_size,
            render_images=getattr(observation_encoder, "requires_images", False),
            **common,
        )
    if not args.chunk_policy:
        raise ValueError("--chunk-policy is required with --base-policy external-chunk")
    predictor = load_chunk_predictor(
        args.chunk_policy,
        task_name=args.task,
        checkpoint=args.upstream_checkpoint,
        device=args.device,
        factory_kwargs=args.factory_kwargs,
    )
    return create_speed_env(
        chunk_predictor=predictor,
        render_images=(
            not args.no_policy_images
            or getattr(observation_encoder, "requires_images", False)
        ),
        **common,
    )
