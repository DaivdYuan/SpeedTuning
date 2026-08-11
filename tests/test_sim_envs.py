import numpy as np
import pytest

from ee_sim_env import make_ee_sim_env
from policy_speed_env import create_speed_env
from scripts.run_sim import run_task
from scripted_policy import make_scripted_policy
from sim_env import BOX_POSE, make_sim_env
from sim_tasks import TASK_SPECS, normalize_task_name


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("pick-and-place", "pick_and_place"),
        ("sim_transfer_cube_scripted", "pick_and_place"),
        ("sim_insertion", "insertion"),
        ("teabag", "tea_bag"),
        ("sim_transfer_tea_bag_scripted", "tea_bag"),
    ],
)
def test_task_aliases(alias, expected):
    assert normalize_task_name(alias) == expected


@pytest.mark.parametrize("task_name", TASK_SPECS)
def test_end_effector_env_resets_and_steps(task_name):
    env = make_ee_sim_env(task_name, render_images=False, seed=7)
    timestep = env.reset()
    assert timestep.observation["qpos"].shape == (14,)
    assert "images" not in timestep.observation
    action = make_scripted_policy(task_name)(timestep)
    timestep = env.step(action)
    assert np.isfinite(timestep.observation["qpos"]).all()


@pytest.mark.parametrize("task_name", TASK_SPECS)
def test_joint_env_resets_without_external_object_pose(task_name):
    BOX_POSE[0] = None
    env = make_sim_env(task_name, render_images=False, seed=7)
    timestep = env.reset()
    timestep = env.step(timestep.observation["qpos"])
    assert timestep.observation["qpos"].shape == (14,)
    assert np.isfinite(timestep.observation["env_state"]).all()


def test_tea_bag_pose_randomization_is_opt_in_and_seeded():
    fixed = make_ee_sim_env("tea_bag", render_images=False, seed=7)
    fixed_pose = fixed.reset().observation["env_state"][:7].copy()
    np.testing.assert_allclose(
        fixed.reset().observation["env_state"][:7], fixed_pose
    )

    randomized = make_ee_sim_env(
        "tea_bag",
        render_images=False,
        seed=7,
        randomize_object_pose=True,
    )
    first_pose = randomized.reset().observation["env_state"][:7].copy()
    second_pose = randomized.reset().observation["env_state"][:7].copy()
    assert not np.allclose(first_pose[:2], second_pose[:2])
    assert np.all(first_pose[:2] >= [0.0, 0.4])
    assert np.all(first_pose[:2] <= [0.2, 0.6])


@pytest.mark.parametrize("task_name", TASK_SPECS)
def test_original_scripted_policy_completes_task(task_name):
    spec = TASK_SPECS[task_name]
    env = make_ee_sim_env(task_name, render_images=False, seed=0)
    timestep = env.reset()
    policy = make_scripted_policy(task_name)
    rewards = []
    for _ in range(spec.episode_len):
        timestep = env.step(policy(timestep))
        rewards.append(timestep.reward)
    assert max(rewards) == env.task.max_reward


@pytest.mark.parametrize("task_name", TASK_SPECS)
def test_speed_wrapper_completes_task_at_original_speed(task_name):
    env = create_speed_env(task_name=task_name, seed=0)
    observation = env.reset()
    assert observation.shape == (env.obs_space,)
    done = False
    info = {}
    while not done:
        observation, _, done, info = env.step(1.0, quantized=False)
    assert info["success"]


@pytest.mark.parametrize("task_name", TASK_SPECS)
def test_scripted_policy_completes_task_at_1_5x(task_name):
    result = run_task(task_name, speed=1.5, seed=0)
    assert result["success"]
    assert result["steps"] < TASK_SPECS[task_name].episode_len


@pytest.mark.parametrize("task_name", TASK_SPECS)
def test_quantized_speed_wrapper_completes_task_at_1_5x(task_name):
    env = create_speed_env(task_name=task_name, seed=0)
    env.reset()
    done = False
    info = {}
    while not done:
        _, _, done, info = env.step(1, quantized=True)
    assert info["success"]
