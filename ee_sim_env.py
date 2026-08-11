"""MuJoCo environments controlled by left and right end-effector poses."""

from __future__ import annotations

import collections
import os

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import (
    DT,
    PUPPET_GRIPPER_POSITION_CLOSE,
    PUPPET_GRIPPER_POSITION_NORMALIZE_FN,
    PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN,
    PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN,
    START_ARM_POSE,
    XML_DIR,
)
from sim_tasks import (
    get_task_spec,
    insertion_reward,
    normalize_task_name,
    sample_box_pose,
    sample_insertion_pose,
    tea_bag_reward,
    transfer_cube_reward,
)


# Kept for compatibility with the historical scripts. New code should pass
# render_images=False to make_ee_sim_env instead.
DISABLE_RENDER = [False]


def make_ee_sim_env(
    task_name: str,
    render_images: bool = True,
    seed: int | None = None,
    object_pose=None,
    randomize_object_pose: bool = False,
):
    """Create an end-effector-control environment for a SpeedTuning sim task.

    Actions contain two 8D commands: xyz, quaternion, and normalized gripper
    position for the left arm, followed by the same fields for the right arm.
    The factory accepts both the public task names (``pick_and_place``,
    ``insertion``, ``tea_bag``) and the task names used by the historical code.
    """

    task_name = normalize_task_name(task_name)
    spec = get_task_spec(task_name)
    physics = mujoco.Physics.from_xml_path(os.path.join(XML_DIR, spec.ee_xml))
    random_state = np.random.RandomState(seed)
    task_classes = {
        "pick_and_place": TransferCubeEETask,
        "insertion": InsertionEETask,
        "tea_bag": TransferTeaBagEETask,
    }
    task = task_classes[task_name](
        random=random_state,
        render_images=render_images,
        object_pose=object_pose,
        randomize_object_pose=randomize_object_pose,
    )
    return control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )


class BimanualViperXEETask(base.Task):
    def __init__(
        self,
        random=None,
        render_images: bool = True,
        object_pose=None,
        randomize_object_pose: bool = False,
    ):
        super().__init__(random=random)
        self.render_images = render_images
        self.object_pose = (
            None if object_pose is None else np.asarray(object_pose).copy()
        )
        self.randomize_object_pose = bool(randomize_object_pose)

    def before_step(self, action, physics):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (16,) or not np.all(np.isfinite(action)):
            raise ValueError("End-effector actions must be a finite array with shape (16,)")

        action_left = action[:8]
        action_right = action[8:]
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        left_gripper = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        right_gripper = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(
            physics.data.ctrl,
            [left_gripper, -left_gripper, right_gripper, -right_gripper],
        )

    def initialize_robots(self, physics):
        physics.named.data.qpos[:16] = START_ARM_POSE
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        np.copyto(physics.data.mocap_pos[1], [0.31718881, 0.49999888, 0.29525084])
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])
        np.copyto(
            physics.data.ctrl,
            [
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
            ],
        )

    @staticmethod
    def get_qpos(physics):
        qpos = physics.data.qpos.copy()
        left, right = qpos[:8], qpos[8:16]
        return np.concatenate(
            [
                left[:6],
                [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left[6])],
                right[:6],
                [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right[6])],
            ]
        )

    @staticmethod
    def get_qvel(physics):
        qvel = physics.data.qvel.copy()
        left, right = qvel[:8], qvel[8:16]
        return np.concatenate(
            [
                left[:6],
                [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left[6])],
                right[:6],
                [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right[6])],
            ]
        )

    @staticmethod
    def get_env_state(physics):
        return physics.data.qpos.copy()[16:]

    def get_observation(self, physics):
        obs = collections.OrderedDict(
            qpos=self.get_qpos(physics),
            qvel=self.get_qvel(physics),
            env_state=self.get_env_state(physics),
        )
        if self.render_images and not DISABLE_RENDER[0]:
            obs["images"] = {
                "top": physics.render(height=480, width=640, camera_id="top"),
                "angle": physics.render(height=480, width=640, camera_id="angle"),
                "vis": physics.render(height=480, width=640, camera_id="front_close"),
            }
        obs["mocap_pose_left"] = np.concatenate(
            [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]
        ).copy()
        obs["mocap_pose_right"] = np.concatenate(
            [physics.data.mocap_pos[1], physics.data.mocap_quat[1]]
        ).copy()
        obs["gripper_ctrl"] = physics.data.ctrl.copy()
        return obs


class TransferCubeEETask(BimanualViperXEETask):
    max_reward = 4

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        pose = (
            sample_box_pose(self.random)
            if self.object_pose is None
            else self.object_pose
        )
        if np.asarray(pose).shape != (7,):
            raise ValueError("Pick-and-place object_pose must have shape (7,)")
        physics.named.data.qpos["red_box_joint"] = pose
        super().initialize_episode(physics)

    def get_reward(self, physics):
        return transfer_cube_reward(physics)


class InsertionEETask(BimanualViperXEETask):
    max_reward = 4

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        if self.object_pose is None:
            peg_pose, socket_pose = sample_insertion_pose(self.random)
        else:
            if self.object_pose.shape != (14,):
                raise ValueError("Insertion object_pose must have shape (14,)")
            peg_pose, socket_pose = self.object_pose[:7], self.object_pose[7:]
        physics.named.data.qpos["red_peg_joint"] = peg_pose
        physics.named.data.qpos["blue_socket_joint"] = socket_pose
        super().initialize_episode(physics)

    def get_reward(self, physics):
        return insertion_reward(physics)


class TransferTeaBagEETask(BimanualViperXEETask):
    max_reward = 3

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        if self.object_pose is not None:
            pose = self.object_pose
        elif self.randomize_object_pose:
            pose = sample_box_pose(self.random)
        else:
            pose = [0.15, 0.5, 0.05, 1, 0, 0, 0]
        if np.asarray(pose).shape != (7,):
            raise ValueError("Tea-bag object_pose must have shape (7,)")
        physics.named.data.qpos["red_box_joint"] = pose
        super().initialize_episode(physics)

    def get_reward(self, physics):
        return tea_bag_reward(physics)
