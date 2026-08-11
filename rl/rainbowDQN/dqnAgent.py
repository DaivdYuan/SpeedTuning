"""Rainbow DQN optimization core used by speed-policy training."""

from __future__ import annotations

import random
from pathlib import Path
from time import perf_counter
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from .network import Network
from .replayBuffer import PrioritizedReplayBuffer, ReplayBuffer


class DQNAgent:
    """Categorical Double DQN with dueling NoisyNet, PER, and n-step replay.

    The high-level, simulator-specific loop lives in ``speed_training.py``.
    This class owns action selection, transition collection, and optimizer
    updates so it can also be reused by short integration checks.
    """

    def __init__(
        self,
        env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int,
        lr: float = 1e-4,
        gamma: float = 0.97,
        tau: float = 0.5,
        frame_skip: int = 10,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.1,
        hard_exploration_steps: int = 0,
        exploration_steps: int = 0,
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        v_min: float = 0.0,
        v_max: float = 120.0,
        atom_size: int = 121,
        n_step: int = 3,
        n_step_alpha: float = 1.0,
        hidden_dim: int = 256,
        device=None,
        **legacy_options,
    ):
        # Logging/checkpoint options belonged to the unrecovered monolithic
        # trainer. Accept them for source compatibility; public loops own I/O.
        ignored = {"log_dir", "file_path", "name", "ckpt_save_freq"}
        unknown = set(legacy_options).difference(ignored)
        if unknown:
            raise TypeError(f"Unexpected DQN options: {', '.join(sorted(unknown))}")
        if batch_size <= 0 or memory_size < batch_size:
            raise ValueError("memory_size must be at least a positive batch_size")
        if frame_skip <= 0 or atom_size < 2 or v_max <= v_min:
            raise ValueError("Invalid frame skip or categorical support")

        random.seed(seed)
        np.random.seed(seed)
        self.env = env
        self.frame_skip = int(frame_skip)
        self.batch_size = int(batch_size)
        self.target_update = int(target_update)
        self.seed = int(seed)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.exploration_steps = int(exploration_steps)
        self.hard_exploration_steps = int(hard_exploration_steps)
        self.min_epsilon = float(min_epsilon)
        self.beta = float(beta)
        self.prior_eps = float(prior_eps)

        obs_dim = int(env.obs_space)
        action_dim = int(env.action_space)
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
        )
        self.use_n_step = n_step > 1
        self.n_step = int(n_step)
        self.n_step_alpha = float(n_step_alpha)
        if self.use_n_step:
            self.memory_n = ReplayBuffer(
                obs_dim,
                memory_size,
                batch_size,
                n_step=n_step,
                gamma=gamma,
            )

        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.atom_size = int(atom_size)
        self.support = torch.linspace(v_min, v_max, atom_size).to(self.device)
        self.dqn = Network(
            obs_dim, action_dim, atom_size, self.support, hidden_dim=hidden_dim
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, atom_size, self.support, hidden_dim=hidden_dim
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.transition = []
        self.is_test = False

    def select_action(self, state: np.ndarray) -> int:
        """Select a discrete speed action and start a replay transition."""

        explore = not self.is_test and np.random.uniform() < self.epsilon
        if explore:
            action = int(np.random.randint(self.env.action_space))
        else:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action = int(self.dqn(state_tensor).argmax(dim=1).item())
        if not self.is_test:
            self.transition = [np.asarray(state, dtype=np.float32).copy(), action]
        return action

    def step(self, action: int, frame_skip: int | None = None):
        """Execute one decision-level speed action and store one transition."""

        repeats = self.frame_skip if frame_skip is None else int(frame_skip)
        if repeats <= 0:
            raise ValueError("frame_skip must be positive")
        started = perf_counter()
        step_decision = getattr(self.env, "step_decision", None)
        if step_decision is not None:
            next_state, total_reward, done, info = step_decision(
                action, frame_skip=repeats
            )
        else:
            total_reward = 0.0
            info = {"success": False}
            for _ in range(repeats):
                next_state, reward, done, info = self.env.step(action)
                total_reward += float(reward)
                if done:
                    break
        env_step_time = perf_counter() - started

        buffer_started = perf_counter()
        if not self.is_test:
            if len(self.transition) != 2:
                raise RuntimeError("Call select_action() before step() while training")
            transition = self.transition + [total_reward, next_state, done]
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*transition)
            else:
                one_step_transition = transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
        buffer_time = perf_counter() - buffer_started
        result_info = dict(info)
        result_info.update(
            env_step_time=env_step_time,
            add_to_buffer_time=buffer_time,
        )
        return next_state, total_reward, done, result_info

    def update_model(self) -> float:
        """Perform one prioritized categorical DQN update."""

        samples = self.memory.sample_batch(self.beta)
        weights = torch.as_tensor(
            samples["weights"].reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        indices = samples["indices"]
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        if self.use_n_step:
            n_step_samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss = elementwise_loss + self.n_step_alpha * self._compute_dqn_loss(
                n_step_samples, self.gamma**self.n_step
            )
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        priorities = elementwise_loss.detach().cpu().numpy() + self.prior_eps
        self.memory.update_priorities(indices, priorities)
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()
        return float(loss.item())

    def decay_epsilon(self, step: int):
        if step <= self.hard_exploration_steps:
            self.epsilon = 1.0
        elif self.exploration_steps <= 0:
            self.epsilon = 0.0
        elif step >= self.exploration_steps:
            self.epsilon = self.min_epsilon
        else:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def set_eval(self, enabled=True):
        self.is_test = bool(enabled)
        self.dqn.eval() if enabled else self.dqn.train()

    def _compute_dqn_loss(
        self, samples: Dict[str, np.ndarray], gamma: float
    ) -> torch.Tensor:
        state = torch.as_tensor(samples["obs"], dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(
            samples["next_obs"], dtype=torch.float32, device=self.device
        )
        action = torch.as_tensor(samples["acts"], dtype=torch.long, device=self.device)
        reward = torch.as_tensor(
            samples["rews"].reshape(-1, 1), dtype=torch.float32, device=self.device
        )
        done = torch.as_tensor(
            samples["done"].reshape(-1, 1), dtype=torch.float32, device=self.device
        )
        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)[
                range(self.batch_size), next_action
            ]
            target_support = (
                reward + (1 - done) * gamma * self.support
            ).clamp(self.v_min, self.v_max)
            projection = (target_support - self.v_min) / delta_z
            lower = projection.floor().long()
            upper = projection.ceil().long()
            lower[(upper == lower) & (upper > 0)] -= 1
            upper[(upper == lower) & (lower < self.atom_size - 1)] += 1
            offset = (
                torch.arange(self.batch_size, device=self.device).unsqueeze(1)
                * self.atom_size
            )
            projected_dist = torch.zeros_like(next_dist)
            projected_dist.view(-1).index_add_(
                0,
                (lower + offset).view(-1),
                (next_dist * (upper.float() - projection)).view(-1),
            )
            projected_dist.view(-1).index_add_(
                0,
                (upper + offset).view(-1),
                (next_dist * (projection - lower.float())).view(-1),
            )

        distribution = self.dqn.dist(state)
        log_probability = torch.log(
            distribution[range(self.batch_size), action].clamp_min(1e-8)
        )
        return -(projected_dist * log_probability).sum(1)

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _target_soft_update(self):
        with torch.no_grad():
            for target, source in zip(
                self.dqn_target.parameters(), self.dqn.parameters()
            ):
                target.copy_(self.tau * source + (1.0 - self.tau) * target)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.dqn.state_dict(), path)

    def load(self, path):
        self.dqn.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self._target_hard_update()
