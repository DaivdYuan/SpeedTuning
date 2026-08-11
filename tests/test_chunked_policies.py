import numpy as np
import pytest

from chunked_policy import (
    ChunkPredictorAdapter,
    ChunkedPolicyRunner,
    TorchChunkPredictor,
    as_action_chunk,
    interpolate_action_chunk,
    replay_recorded_chunks,
)
from sim_env import make_sim_env
from sim_tasks import TASK_SPECS


def test_action_chunk_interpolation_changes_execution_length():
    actions = np.arange(10 * 14, dtype=float).reshape(10, 14)
    accelerated = interpolate_action_chunk(actions, speed=2.0)
    slowed = interpolate_action_chunk(actions, speed=0.5)
    assert accelerated.shape == (5, 14)
    assert slowed.shape == (20, 14)
    np.testing.assert_array_equal(accelerated, actions[::2])


def test_upstream_chunk_adapter_accepts_common_batch_and_dict_output():
    actions = np.arange(4 * 14, dtype=float).reshape(1, 4, 14)
    adapter = ChunkPredictorAdapter(lambda observation: {"actions": actions})
    np.testing.assert_array_equal(adapter({}), actions[0])
    np.testing.assert_array_equal(as_action_chunk(actions[0, 0]), actions[0, :1])


def test_chunk_runner_accepts_online_speed_changes():
    actions = np.repeat(np.arange(10, dtype=float)[:, None], 14, axis=1)
    runner = ChunkedPolicyRunner(lambda observation: actions)
    samples = [
        runner.action({}, speed=speed)[0]
        for speed in (1.0, 2.0, 0.5, 1.5)
    ]
    np.testing.assert_allclose(samples, [0.0, 1.0, 3.0, 3.5])


@pytest.mark.parametrize("task_name", TASK_SPECS)
def test_recorded_action_chunks_complete_joint_task(task_name):
    result = replay_recorded_chunks(task_name, chunk_size=25, seed=0)
    assert result["success"]


@pytest.mark.learned
def test_actual_act_model_accepts_every_simulator_observation():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from policy import ACTPolicy

    config = {
        "lr": 1e-4,
        "num_queries": 4,
        "kl_weight": 1,
        "hidden_dim": 32,
        "dim_feedforward": 64,
        "lr_backbone": 0.0,
        "backbone": "resnet18",
        "enc_layers": 1,
        "dec_layers": 1,
        "nheads": 4,
        "camera_names": ["top"],
        "pretrained_backbone": False,
        "device": "cpu",
    }
    policy = ACTPolicy(config).eval()
    for task_name in TASK_SPECS:
        env = make_sim_env(task_name, render_images=True, seed=0)
        timestep = env.reset()
        initial_qpos = timestep.observation["qpos"].copy()
        predictor = TorchChunkPredictor(
            policy,
            ["top"],
            qpos_mean=np.zeros(14),
            qpos_std=np.ones(14),
            action_mean=initial_qpos,
            action_std=np.full(14, 1e-3),
            device="cpu",
        )
        chunk = predictor(timestep.observation)
        assert chunk.shape == (4, 14)
        assert np.isfinite(chunk).all()
        timestep = env.step(chunk[0])
        assert torch.isfinite(torch.as_tensor(timestep.observation["qpos"])).all()


@pytest.mark.rl
def test_rainbow_dqn_optimization_poc():
    pytest.importorskip("torch")
    from scripts.rainbow_poc import run_poc

    result = run_poc(seed=0, min_transitions=64, updates=8)
    assert result["passed"]
    assert result["successful_episodes"] >= 1
    assert result["actions_seen"] == [0, 1, 2, 3, 4]
