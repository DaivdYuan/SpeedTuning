import json
from pathlib import Path

import numpy as np
import pytest

from chunked_policy import ChunkedPolicyRunner
from experiment_config import load_experiment_config
from policy_speed_env import create_speed_env, make_speed_reward
from speed_evaluation import speed_grid
from speed_observation import StateObservationEncoder, VisualObservationEncoder
from speed_policy import FixedSpeedPolicy, rollout_speed_policy, summarize_rollouts


class CountingChunkPredictor:
    def __init__(self):
        self.calls = 0

    def __call__(self, observation):
        self.calls += 1
        return np.repeat(np.asarray(observation["qpos"])[None], 100, axis=0)


def test_chunk_runner_replans_at_each_decision_boundary():
    predictor = CountingChunkPredictor()
    runner = ChunkedPolicyRunner(predictor)
    observation = {"qpos": np.zeros(14)}
    runner.begin_decision(observation, speed=2.0)
    runner.action(observation, speed=2.0)
    runner.action(observation, speed=2.0)
    assert predictor.calls == 1
    runner.begin_decision(observation, speed=1.5)
    assert predictor.calls == 2


def test_decision_step_uses_one_fresh_chunk_and_holds_speed():
    predictor = CountingChunkPredictor()
    env = create_speed_env(
        "tea_bag",
        chunk_predictor=predictor,
        render_images=False,
        seed=0,
    )
    env.reset()
    _, _, done, info = env.step_decision(1.5, frame_skip=4, quantized=False)
    assert not done
    assert predictor.calls == 1
    assert info["decision_physics_steps"] == 4
    assert env.speed_list == [1.5] * 4
    env.step_decision(1.0, frame_skip=2, quantized=False)
    assert predictor.calls == 2


def test_stacked_unprivileged_proprioceptive_observations():
    env = create_speed_env(
        "pick_and_place",
        observation_encoder=StateObservationEncoder(include_env_state=False),
        frame_stack=5,
        seed=0,
    )
    observation = env.reset()
    assert observation.shape == (5 * 28,)
    frames = observation.reshape(5, 28)
    np.testing.assert_allclose(frames, np.repeat(frames[:1], 5, axis=0))
    next_observation, _, _, _ = env.step_decision(
        1.0, frame_skip=1, quantized=False
    )
    assert next_observation.shape == observation.shape
    np.testing.assert_allclose(next_observation[: 4 * 28], observation[28:])


class DummyImageEncoder:
    feature_dim = 2

    def __call__(self, images):
        means = images.mean(axis=(1, 2, 3))
        return np.stack([means, means + 1.0], axis=1)

    def spec(self):
        return {"type": "dummy", "feature_dim": self.feature_dim}


def test_visual_encoder_fuses_camera_features_without_env_state():
    encoder = VisualObservationEncoder(
        camera_names=("top", "angle"),
        image_encoder=DummyImageEncoder(),
        include_env_state=False,
    )
    observation = {
        "qpos": np.zeros(14),
        "qvel": np.ones(14),
        "env_state": np.full(7, 9.0),
        "images": {
            "top": np.zeros((4, 5, 3), dtype=np.uint8),
            "angle": np.full((4, 5, 3), 10, dtype=np.uint8),
        },
    }
    features = encoder(observation)
    assert features.shape == (32,)
    assert not np.any(features == 9.0)
    assert encoder.spec()["image_encoder"]["type"] == "dummy"


def test_paper_manifest_and_ablation_inheritance():
    paper, _ = load_experiment_config("paper-sim")
    assert paper["speed_values"] == [1.5, 2.0, 3.0, 4.5]
    assert paper["frame_stack"] == 5
    assert paper["include_env_state"] is False
    assert paper["speed_weight"] == 0.01
    assert paper["update_schedule"] == "episode"
    no_image_path = Path(__file__).parents[1] / "configs/ablations/no_image.json"
    no_image, _ = load_experiment_config(no_image_path)
    assert no_image["speed_observation"] == "state"
    assert no_image["hidden_dim"] == 1024

    scripted, _ = load_experiment_config("scripted-tea-bag")
    assert scripted["base_policy"] == "scripted"
    assert scripted["speed_values"] == [1.0, 1.5, 2.0, 2.5, 3.0]
    assert scripted["decisions"] == 100_000
    assert scripted["learning_starts"] == 128
    assert scripted["beta_schedule"] == "legacy"
    randomized, _ = load_experiment_config("scripted-tea-bag-randomized")
    assert randomized["randomize_object_pose"] is True
    assert randomized["update_schedule"] == "episode"

    pick, _ = load_experiment_config("scripted-pick-and-place")
    insertion, _ = load_experiment_config("scripted-insertion")
    assert pick["speed_values"][-1] == 4.5
    assert insertion["speed_values"] == [1.0, 1.5, 2.0, 2.5, 3.0]
    assert pick["decisions"] == insertion["decisions"] == 100_000


def test_paper_metrics_use_physical_length_not_commanded_speed():
    env = create_speed_env("tea_bag", seed=0, decision_frame_skip=10)
    rollout = rollout_speed_policy(env, FixedSpeedPolicy(1.5))
    assert rollout["acceleration"] == pytest.approx(
        env.episode_len / rollout["physics_steps"]
    )
    summary = summarize_rollouts([rollout])
    assert summary["mean_acceleration"] == rollout["acceleration"]
    assert "mean_commanded_speed" in summary


def test_speed_grid_is_stable_at_decimal_intervals():
    assert speed_grid(1.0, 1.3, 0.1) == (1.0, 1.1, 1.2, 1.3)


def test_zero_degree_speed_reward_ablation_is_supported():
    reward = make_speed_reward(speed_weight=0.0, speed_power=0.0)
    assert reward(4.5, done=False, success=False) == 0.0


def test_reference_results_cover_all_tasks_without_checkpoint_paths():
    path = Path(__file__).parents[1] / "benchmarks/scripted_results.json"
    benchmark = json.loads(path.read_text())
    assert set(benchmark["results"]) == {"pick_and_place", "insertion", "tea_bag"}
    serialized = json.dumps(benchmark)
    assert not any(suffix in serialized for suffix in (".pt\"", ".pth\"", ".ckpt\""))
    for result in benchmark["results"].values():
        assert 0.0 <= result["learned_speed"]["success_rate"] <= 1.0
        assert result["learned_speed"]["mean_physical_acceleration"] > 1.0


@pytest.mark.learned
def test_resnet_visual_speed_observation_and_state_round_trip():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    first = VisualObservationEncoder(
        camera_names=("top",),
        pretrained=False,
        image_size=64,
        device="cpu",
        include_env_state=False,
    )
    env = create_speed_env(
        "tea_bag",
        observation_encoder=first,
        frame_stack=2,
        render_images=True,
        seed=0,
    )
    observation = env.reset()
    assert observation.shape == (2 * (28 + 512),)
    saved_state = first.state_dict()
    second = VisualObservationEncoder(
        camera_names=("top",),
        pretrained=False,
        image_size=64,
        device="cpu",
        include_env_state=False,
    )
    second.load_state_dict(saved_state)
    assert first.spec() == second.spec()
    env.close()


@pytest.mark.learned
def test_retained_act_checkpoint_loader_runs_on_simulator(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from act_integration import build_act_chunk_predictor
    from policy import ACTPolicy
    from sim_env import make_sim_env

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
    checkpoint = tmp_path / "act.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "policy_config": config,
            "stats": {
                "qpos_mean": np.zeros(14),
                "qpos_std": np.ones(14),
                "action_mean": np.zeros(14),
                "action_std": np.ones(14),
            },
        },
        checkpoint,
    )
    predictor = build_act_chunk_predictor(
        "tea_bag", checkpoint=checkpoint, device="cpu"
    )
    env = make_sim_env("tea_bag", render_images=True, seed=0)
    chunk = predictor(env.reset().observation)
    assert chunk.shape == (4, 14)
    assert np.isfinite(chunk).all()
