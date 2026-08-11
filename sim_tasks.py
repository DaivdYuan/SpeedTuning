"""Shared definitions for the reconstructed SpeedTuning simulation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    """Metadata shared by the end-effector and joint-control environments."""

    name: str
    legacy_name: str
    episode_len: int
    ee_xml: str
    joint_xml: str


TASK_SPECS = {
    "pick_and_place": TaskSpec(
        name="pick_and_place",
        legacy_name="sim_transfer_cube_scripted",
        episode_len=400,
        ee_xml="bimanual_viperx_ee_transfer_cube.xml",
        joint_xml="bimanual_viperx_transfer_cube.xml",
    ),
    "insertion": TaskSpec(
        name="insertion",
        legacy_name="sim_insertion_scripted",
        episode_len=400,
        ee_xml="bimanual_viperx_ee_insertion.xml",
        joint_xml="bimanual_viperx_insertion.xml",
    ),
    "tea_bag": TaskSpec(
        name="tea_bag",
        legacy_name="sim_transfer_tea_bag_scripted",
        episode_len=500,
        ee_xml="bimanual_viperx_ee_transfer_tea_bag.xml",
        joint_xml="bimanual_viperx_transfer_tea_bag.xml",
    ),
}

_ALIASES = {
    "pick_and_place": "pick_and_place",
    "pick_place": "pick_and_place",
    "transfer_cube": "pick_and_place",
    "sim_transfer_cube": "pick_and_place",
    "sim_transfer_cube_scripted": "pick_and_place",
    "sim_transfer_cube_human": "pick_and_place",
    "insertion": "insertion",
    "sim_insertion": "insertion",
    "sim_insertion_scripted": "insertion",
    "sim_insertion_human": "insertion",
    "tea_bag": "tea_bag",
    "teabag": "tea_bag",
    "transfer_tea_bag": "tea_bag",
    "sim_transfer_tea_bag": "tea_bag",
    "sim_transfer_tea_bag_scripted": "tea_bag",
}


def normalize_task_name(task_name: str) -> str:
    """Return the public task name while accepting names used by the old code."""

    normalized = task_name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(TASK_SPECS)
        raise ValueError(f"Unknown task {task_name!r}. Supported tasks: {supported}") from exc


def get_task_spec(task_name: str) -> TaskSpec:
    return TASK_SPECS[normalize_task_name(task_name)]


def sample_box_pose(random_state=None) -> np.ndarray:
    """Sample the cube pose from the range used by the original ACT simulator."""

    rng = np.random if random_state is None else random_state
    position = rng.uniform([0.0, 0.4, 0.05], [0.2, 0.6, 0.05])
    return np.concatenate([position, [1.0, 0.0, 0.0, 0.0]])


def sample_insertion_pose(random_state=None) -> tuple[np.ndarray, np.ndarray]:
    """Sample peg and socket poses from the original simulator ranges."""

    rng = np.random if random_state is None else random_state
    peg_position = rng.uniform([0.1, 0.4, 0.05], [0.2, 0.6, 0.05])
    socket_position = rng.uniform([-0.2, 0.4, 0.05], [-0.1, 0.6, 0.05])
    identity_quaternion = [1.0, 0.0, 0.0, 0.0]
    return (
        np.concatenate([peg_position, identity_quaternion]),
        np.concatenate([socket_position, identity_quaternion]),
    )


def contact_pairs(physics) -> set[frozenset[str]]:
    """Collect active contacts without depending on MuJoCo's geom ordering."""

    pairs = set()
    for contact in physics.data.contact[: physics.data.ncon]:
        geom_1 = physics.model.id2name(contact.geom1, "geom")
        geom_2 = physics.model.id2name(contact.geom2, "geom")
        if geom_1 is not None and geom_2 is not None:
            pairs.add(frozenset((geom_1, geom_2)))
    return pairs


def _touching(pairs: set[frozenset[str]], first: str, second: str) -> bool:
    return frozenset((first, second)) in pairs


def _gripper_touch(
    pairs: set[frozenset[str]], object_geom: str, side: str
) -> bool:
    return any(
        _touching(pairs, object_geom, f"vx300s_{side}/10_{finger}_gripper_finger")
        for finger in ("left", "right")
    )


def transfer_cube_reward(physics) -> int:
    pairs = contact_pairs(physics)
    left = _gripper_touch(pairs, "red_box", "left")
    right = _gripper_touch(pairs, "red_box", "right")
    table = _touching(pairs, "red_box", "table")

    if left and not table:
        return 4
    if left:
        return 3
    if right and not table:
        return 2
    if right:
        return 1
    return 0


def insertion_reward(physics) -> int:
    pairs = contact_pairs(physics)
    socket_geoms: Iterable[str] = ("socket-1", "socket-2", "socket-3", "socket-4")
    right = _gripper_touch(pairs, "red_peg", "right")
    left = any(
        _gripper_touch(pairs, socket, "left")
        for socket in socket_geoms
    )
    peg_on_table = _touching(pairs, "red_peg", "table")
    socket_on_table = any(_touching(pairs, socket, "table") for socket in socket_geoms)
    peg_in_socket = any(_touching(pairs, "red_peg", socket) for socket in socket_geoms)
    pin_touched = _touching(pairs, "red_peg", "pin")

    if pin_touched:
        return 4
    if peg_in_socket and not peg_on_table and not socket_on_table:
        return 3
    if left and right and not peg_on_table and not socket_on_table:
        return 2
    if left and right:
        return 1
    return 0


def tea_bag_reward(physics) -> int:
    pairs = contact_pairs(physics)
    right = _gripper_touch(pairs, "red_box", "right")
    on_table = _touching(pairs, "tea_bag", "table") or _touching(
        pairs, "red_box", "table"
    )
    in_cup = _touching(pairs, "cup_base", "tea_bag")

    if in_cup:
        return 3
    if right and not on_table:
        return 2
    if right:
        return 1
    return 0
