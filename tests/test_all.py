"""
Unit tests for environment, agent, and training components.
Run with: python -m pytest tests/ -v
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.grid_env import GridStrategyEnv, NUM_ACTIONS
from agents.actor_critic import ActorCritic
from agents.ppo_agent import PPOAgent, PPOConfig, RolloutBuffer
from utils.replay_buffer import PrioritizedReplayBuffer, SumTree


# ─── Environment Tests ────────────────────────────────────────────────────────

class TestGridEnv:

    def setup_method(self):
        self.env = GridStrategyEnv(grid_size=8, num_agents=2, max_steps=50, seed=0)
        self.env.reset(seed=0)

    def test_reset(self):
        assert len(self.env.agents) == 2
        assert self.env._step_count == 0

    def test_observation_shape(self):
        obs = self.env.observe(self.env.possible_agents[0])
        expected_dim = 8 * 8 * 5
        assert obs.shape == (expected_dim,), f"Expected {expected_dim}, got {obs.shape}"

    def test_step_basic(self):
        agent = self.env.agent_selection
        obs_before = self.env.observe(agent)
        self.env.step(0)  # MOVE_UP
        assert self.env._step_count == 1

    def test_all_actions_valid(self):
        for action in range(NUM_ACTIONS):
            env = GridStrategyEnv(grid_size=8, num_agents=2, max_steps=10, seed=42)
            env.reset(seed=42)
            for _ in range(5):
                if not env.agents:
                    break
                env.step(action % NUM_ACTIONS)

    def test_episode_terminates(self):
        env = GridStrategyEnv(grid_size=8, num_agents=2, max_steps=20, seed=1)
        env.reset(seed=1)
        steps = 0
        while env.agents and steps < 100:
            env.step(np.random.randint(NUM_ACTIONS))
            steps += 1
        # Episode should end (via truncation at max_steps)
        assert steps <= 21 * 2  # AEC: each agent acts each step

    def test_rewards_finite(self):
        self.env.reset(seed=5)
        for _ in range(20):
            if not self.env.agents:
                break
            self.env.step(np.random.randint(NUM_ACTIONS))
            for a in self.env.possible_agents:
                r = self.env.rewards.get(a, 0.0)
                assert np.isfinite(r), f"Non-finite reward: {r}"

    def test_4_agents(self):
        env = GridStrategyEnv(grid_size=12, num_agents=4, max_steps=50, seed=0)
        env.reset(seed=0)
        assert len(env.agents) == 4
        obs = env.observe(env.possible_agents[0])
        assert obs.shape == (12 * 12 * 5,)

    def test_obs_range(self):
        obs = self.env.observe(self.env.possible_agents[0])
        assert obs.min() >= 0.0 and obs.max() <= 1.0 + 1e-6


# ─── Actor-Critic Tests ───────────────────────────────────────────────────────

class TestActorCritic:

    def setup_method(self):
        self.obs_dim  = 8 * 8 * 5
        self.n_act    = NUM_ACTIONS
        self.n_agents = 2
        self.net = ActorCritic(
            obs_dim     = self.obs_dim,
            n_actions   = self.n_act,
            n_agents    = self.n_agents,
            hidden_dim  = 64,
            centralized = True,
        )

    def test_forward_shapes(self):
        bs   = 4
        obs  = torch.randn(bs, self.obs_dim)
        jt   = torch.randn(bs, self.obs_dim * self.n_agents)
        logits, value = self.net(obs)
        assert logits.shape == (bs, self.n_act)

    def test_act(self):
        obs = torch.randn(1, self.obs_dim)
        action, log_prob, entropy = self.net.act(obs)
        assert 0 <= action.item() < self.n_act
        assert torch.isfinite(log_prob)
        assert entropy.item() >= 0

    def test_evaluate_actions(self):
        bs      = 8
        obs     = torch.randn(bs, self.obs_dim)
        actions = torch.randint(0, self.n_act, (bs,))
        jo      = torch.randn(bs, self.obs_dim * self.n_agents)
        lp, ent, val = self.net.evaluate_actions(obs, actions, jo)
        assert lp.shape  == (bs,)
        assert ent.shape == (bs,)
        assert val.shape == (bs,)

    def test_deterministic_act(self):
        obs = torch.randn(1, self.obs_dim)
        a1, _, _ = self.net.act(obs, deterministic=True)
        a2, _, _ = self.net.act(obs, deterministic=True)
        assert a1.item() == a2.item()


# ─── PPO Agent Tests ──────────────────────────────────────────────────────────

class TestPPOAgent:

    def setup_method(self):
        cfg = PPOConfig(
            lr=1e-3, n_epochs=2, batch_size=16,
            hidden_dim=64, device="cpu"
        )
        self.obs_dim = 8 * 8 * 5
        self.agent = PPOAgent(
            agent_id  = "agent_0",
            obs_dim   = self.obs_dim,
            n_actions = NUM_ACTIONS,
            n_agents  = 2,
            cfg       = cfg,
        )

    def test_select_action(self):
        obs     = np.random.randn(self.obs_dim).astype(np.float32)
        jo      = np.random.randn(self.obs_dim * 2).astype(np.float32)
        action, lp, val = self.agent.select_action(obs, jo)
        assert 0 <= action < NUM_ACTIONS
        assert isinstance(lp, float)
        assert isinstance(val, float)

    def test_buffer_and_update(self):
        obs_dim = self.obs_dim
        for _ in range(32):
            obs  = np.random.randn(obs_dim).astype(np.float32)
            jo   = np.random.randn(obs_dim * 2).astype(np.float32)
            self.agent.buffer.add(obs, jo, 0, -1.0, 0.5, 0.3, False)

        stats = self.agent.update(last_value=0.0, last_done=True)
        assert "policy_loss" in stats
        assert "value_loss"  in stats
        assert "entropy"     in stats

    def test_save_load(self, tmp_path):
        path = str(tmp_path / "agent.pt")
        self.agent.save(path)
        self.agent.load(path)


# ─── Rollout Buffer Tests ─────────────────────────────────────────────────────

class TestRolloutBuffer:

    def test_add_and_clear(self):
        buf = RolloutBuffer()
        obs = np.zeros(10, dtype=np.float32)
        buf.add(obs, obs, 0, -1.0, 1.0, 0.5, False)
        assert len(buf) == 1
        buf.clear()
        assert len(buf) == 0


# ─── Prioritized Replay Buffer Tests ─────────────────────────────────────────

class TestPrioritizedReplayBuffer:

    def test_add_sample(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(50):
            buf.add({"obs": i}, priority=float(i + 1))
        assert len(buf) == 50

        batch, indices, weights = buf.sample(batch_size=10)
        assert len(batch) == 10
        assert len(indices) == 10
        assert weights.shape == (10,)

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(20):
            buf.add(i, priority=1.0)
        batch, indices, weights = buf.sample(8)
        new_priorities = np.abs(np.random.randn(8))
        buf.update_priorities(indices, new_priorities)   # should not raise

    def test_sum_tree_retrieval(self):
        tree = SumTree(capacity=8)
        for i in range(8):
            tree.add(float(i + 1), data=i)
        assert tree.total > 0
        idx, p, data = tree.get(tree.total * 0.5)
        assert p > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
