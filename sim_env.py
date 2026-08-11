"""MuJoCo environments controlled by bimanual robot joint positions."""

from __future__ import annotations

import collections
import os

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import (
    DT,
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


# Historical replay scripts set this after constructing the environment and before
# reset. New code may pass object_pose directly to make_sim_env.
BOX_POSE = [None]


def make_sim_env(
    task_name: str,
    render_images: bool = True,
    seed: int | None = None,
    object_pose=None,
    randomize_object_pose: bool = False,
):
    """Create a joint-control environment for any reconstructed sim task."""

    task_name = normalize_task_name(task_name)
    spec = get_task_spec(task_name)
    physics = mujoco.Physics.from_xml_path(os.path.join(XML_DIR, spec.joint_xml))
    random_state = np.random.RandomState(seed)
    task_classes = {
        "pick_and_place": TransferCubeTask,
        "insertion": InsertionTask,
        "tea_bag": TransferTeaBagTask,
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


class BimanualViperXTask(base.Task):
    def __init__(
        self,
        random=None,
        render_images: bool = True,
        object_pose=None,
        randomize_object_pose: bool = False,
    ):
        super().__init__(random=random)
        self.render_images = render_images
        self.object_pose = None if object_pose is None else np.asarray(object_pose).copy()
        self.randomize_object_pose = bool(randomize_object_pose)

    def before_step(self, action, physics):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (14,) or not np.all(np.isfinite(action)):
            raise ValueError("Joint-control actions must be a finite array with shape (14,)")
        left_gripper = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action[6])
        right_gripper = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action[13])
        env_action = np.concatenate(
            [
                action[:6],
                [left_gripper, -left_gripper],
                action[7:13],
                [right_gripper, -right_gripper],
            ]
        )
        super().before_step(env_action, physics)

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
        if self.render_images:
            obs["images"] = {
                "top": physics.render(height=480, width=640, camera_id="top"),
                "angle": physics.render(height=480, width=640, camera_id="angle"),
                "vis": physics.render(height=480, width=640, camera_id="front_close"),
            }
        return obs

    def _requested_pose(self):
        if self.object_pose is not None:
            return self.object_pose
        if BOX_POSE[0] is not None:
            return np.asarray(BOX_POSE[0])
        return None

    def _initialize_robot(self, physics):
        physics.named.data.qpos[:16] = START_ARM_POSE
        np.copyto(physics.data.ctrl, START_ARM_POSE)


class TransferCubeTask(BimanualViperXTask):
    max_reward = 4

    def initialize_episode(self, physics):
        with physics.reset_context():
            self._initialize_robot(physics)
            pose = self._requested_pose()
            physics.named.data.qpos["red_box_joint"] = (
                sample_box_pose(self.random) if pose is None else pose
            )
        super().initialize_episode(physics)

    def get_reward(self, physics):
        return transfer_cube_reward(physics)


class InsertionTask(BimanualViperXTask):
    max_reward = 4

    def initialize_episode(self, physics):
        with physics.reset_context():
            self._initialize_robot(physics)
            pose = self._requested_pose()
            if pose is None:
                peg_pose, socket_pose = sample_insertion_pose(self.random)
                pose = np.concatenate([peg_pose, socket_pose])
            if np.asarray(pose).shape != (14,):
                raise ValueError("Insertion object_pose must have shape (14,)")
            physics.named.data.qpos["red_peg_joint"] = pose[:7]
            physics.named.data.qpos["blue_socket_joint"] = pose[7:]
        super().initialize_episode(physics)

    def get_reward(self, physics):
        return insertion_reward(physics)


class TransferTeaBagTask(BimanualViperXTask):
    max_reward = 3

    def initialize_episode(self, physics):
        with physics.reset_context():
            self._initialize_robot(physics)
            pose = self._requested_pose()
            if pose is None and self.randomize_object_pose:
                physics.named.data.qpos["red_box_joint"] = sample_box_pose(self.random)
            elif pose is None:
                physics.named.data.qpos["red_box_joint"] = [
                    0.15,
                    0.5,
                    0.05,
                    1,
                    0,
                    0,
                    0,
                ]
            else:
                pose = np.asarray(pose)
                if pose.shape != physics.data.qpos[16:].shape:
                    raise ValueError(
                        f"Tea-bag object_pose must have shape {physics.data.qpos[16:].shape}"
                    )
                physics.data.qpos[16:] = pose
        super().initialize_episode(physics)

    def get_reward(self, physics):
        return tea_bag_reward(physics)
