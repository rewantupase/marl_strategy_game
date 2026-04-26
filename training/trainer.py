"""
MARL Trainer — Centralized Training with Decentralized Execution (CTDE).

Responsibilities:
    1. Collect rollouts from the PettingZoo AEC environment.
    2. Assemble joint observations for the centralized critic.
    3. Trigger per-agent PPO updates.
    4. Log metrics and save checkpoints.
    5. Run self-play win-rate evaluations.
"""

import os
import time
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from envs.grid_env import GridStrategyEnv
from agents.ppo_agent import PPOAgent, PPOConfig
from utils.logger import MetricLogger

logger = logging.getLogger(__name__)


class MARLTrainer:
    """
    Orchestrates multi-agent training.

    CTDE Details:
        - Each agent has its own ActorCritic network.
        - During rollout collection, ALL agent observations are concatenated
          and fed to EACH agent's centralized critic.
        - During deployment / evaluation, only the actor (local obs) is used.
    """

    def __init__(
        self,
        cfg: PPOConfig,
        env_kwargs: Optional[Dict] = None,
        log_dir: str = "runs",
        checkpoint_dir: str = "checkpoints",
        eval_interval: int = 50,
        save_interval: int = 100,
        n_eval_episodes: int = 20,
        seed: int = 42,
    ):
        self.cfg             = cfg
        self.env_kwargs      = env_kwargs or {}
        self.log_dir         = log_dir
        self.checkpoint_dir  = checkpoint_dir
        self.eval_interval   = eval_interval
        self.save_interval   = save_interval
        self.n_eval_episodes = n_eval_episodes
        self.seed            = seed

        os.makedirs(log_dir,        exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Build training environment
        self.env = GridStrategyEnv(seed=seed, **self.env_kwargs)
        self.env.reset(seed=seed)

        self.possible_agents = self.env.possible_agents
        self.n_agents        = len(self.possible_agents)
        self.obs_dim         = self.env.observation_space(self.possible_agents[0]).shape[0]
        self.n_actions       = self.env.action_space(self.possible_agents[0]).n

        # Build one PPO agent per possible agent
        self.agents: Dict[str, PPOAgent] = {
            a: PPOAgent(
                agent_id  = a,
                obs_dim   = self.obs_dim,
                n_actions = self.n_actions,
                n_agents  = self.n_agents,
                cfg       = cfg,
            )
            for a in self.possible_agents
        }

        self.metric_logger = MetricLogger(log_dir)
        self._global_step  = 0

    # ─── Main Training Loop ──────────────────────────────────────────────────

    def train(self, total_timesteps: int) -> None:
        logger.info(f"Starting training | agents={self.n_agents} | obs_dim={self.obs_dim}")
        logger.info(f"CTDE={'enabled' if self.cfg.centralized else 'disabled'}")

        episode  = 0
        ep_start = time.time()

        while self._global_step < total_timesteps:
            ep_rewards, ep_len, winners = self._collect_rollout()

            # Trigger per-agent updates
            train_metrics = {}
            for agent_id, agent in self.agents.items():
                if len(agent.buffer) > 0:
                    metrics = agent.update(last_value=0.0, last_done=True)
                    train_metrics[agent_id] = metrics

            episode += 1
            self._log_episode(episode, ep_rewards, ep_len, winners, train_metrics)

            if episode % self.eval_interval == 0:
                win_rates = self.evaluate(self.n_eval_episodes)
                logger.info(f"[Eval ep={episode}] Win rates: {win_rates}")
                self.metric_logger.log_scalar("eval/win_rate_agent_0",
                                              win_rates.get("agent_0", 0.0), episode)

            if episode % self.save_interval == 0:
                self.save_checkpoint(episode)

        logger.info("Training complete.")
        self.save_checkpoint("final")

    # ─── Rollout Collection ──────────────────────────────────────────────────

    def _collect_rollout(self) -> Tuple[Dict, int, List[str]]:
        """
        Run one full episode, storing transitions in each agent's buffer.
        Returns per-agent cumulative rewards, episode length, and winner list.
        """
        self.env.reset(seed=self.seed + self._global_step)

        ep_rewards: Dict[str, float] = defaultdict(float)
        ep_len    = 0
        winners   = []

        # Cache latest obs for each agent
        last_obs: Dict[str, np.ndarray] = {}
        for a in self.env.possible_agents:
            last_obs[a] = self.env.observe(a)

        while self.env.agents:
            agent_id = self.env.agent_selection
            if agent_id not in self.env.agents:
                self.env.step(0)
                continue

            obs        = self.env.observe(agent_id)
            joint_obs  = self._get_joint_obs(last_obs, agent_id)

            agent     = self.agents[agent_id]
            action, log_prob, value = agent.select_action(obs, joint_obs)

            self.env.step(action)

            reward = self.env.rewards.get(agent_id, 0.0)
            done   = (
                self.env.terminations.get(agent_id, False)
                or self.env.truncations.get(agent_id, False)
            )

            agent.store_transition(obs, joint_obs, action, log_prob, reward, value, done)

            ep_rewards[agent_id] += reward
            ep_len               += 1
            self._global_step    += 1

            # Update cached obs
            for a in self.env.agents:
                last_obs[a] = self.env.observe(a)

            if self._global_step % 500 == 0:
                logger.debug(f"step={self._global_step}")

        # Determine winners (agents not terminated by death)
        for a in self.env.possible_agents:
            if not self.env.terminations.get(a, True):
                winners.append(a)

        return dict(ep_rewards), ep_len, winners

    def _get_joint_obs(
        self,
        last_obs: Dict[str, np.ndarray],
        requesting_agent: str,
    ) -> np.ndarray:
        """
        Concatenate observations of ALL agents in canonical order.
        Used only by centralized critic during training.
        """
        parts = []
        for a in self.possible_agents:
            if a in last_obs:
                parts.append(last_obs[a])
            else:
                parts.append(np.zeros(self.obs_dim, dtype=np.float32))
        return np.concatenate(parts, axis=0)

    # ─── Evaluation ──────────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        """
        Run n_episodes with deterministic policies.
        Returns win-rate per agent.
        """
        wins: Dict[str, int] = defaultdict(int)
        eval_env = GridStrategyEnv(seed=9999, **self.env_kwargs)

        for ep in range(n_episodes):
            eval_env.reset(seed=9999 + ep)
            last_obs = {a: eval_env.observe(a) for a in eval_env.possible_agents}

            while eval_env.agents:
                agent_id = eval_env.agent_selection
                if agent_id not in eval_env.agents:
                    eval_env.step(0)
                    continue

                obs = eval_env.observe(agent_id)
                agent = self.agents[agent_id]
                action, _, _ = agent.select_action(obs, deterministic=True)
                eval_env.step(action)

                for a in eval_env.agents:
                    last_obs[a] = eval_env.observe(a)

            # Winner = last surviving agent
            survivors = [
                a for a in eval_env.possible_agents
                if not eval_env.terminations.get(a, True)
                or eval_env.truncations.get(a, False)
            ]
            for a in survivors:
                wins[a] += 1

        return {a: wins[a] / n_episodes for a in self.possible_agents}

    # ─── Self-Play Evaluation ────────────────────────────────────────────────

    def self_play_win_rate(
        self,
        agent_id: str,
        opponent_id: str,
        n_episodes: int = 100,
    ) -> float:
        """
        Measure win rate of agent_id against opponent_id.
        """
        wins   = 0
        sp_env = GridStrategyEnv(num_agents=2, seed=1234, **{
            k: v for k, v in self.env_kwargs.items() if k != "num_agents"
        })

        for ep in range(n_episodes):
            sp_env.reset(seed=1234 + ep)

            agent_map = {
                sp_env.possible_agents[0]: self.agents[agent_id],
                sp_env.possible_agents[1]: self.agents[opponent_id],
            }

            while sp_env.agents:
                cur = sp_env.agent_selection
                if cur not in sp_env.agents:
                    sp_env.step(0)
                    continue
                obs = sp_env.observe(cur)
                acting_agent = agent_map.get(cur, self.agents[cur])
                action, _, _ = acting_agent.select_action(obs, deterministic=True)
                sp_env.step(action)

            if not sp_env.terminations.get(sp_env.possible_agents[0], True):
                wins += 1

        return wins / n_episodes

    # ─── Logging ─────────────────────────────────────────────────────────────

    def _log_episode(
        self,
        episode: int,
        ep_rewards: Dict,
        ep_len: int,
        winners: List[str],
        train_metrics: Dict,
    ) -> None:
        avg_reward = np.mean(list(ep_rewards.values())) if ep_rewards else 0.0
        logger.info(
            f"Episode {episode:5d} | steps={self._global_step:8d} | "
            f"len={ep_len:4d} | avg_reward={avg_reward:+.3f} | winners={winners}"
        )
        self.metric_logger.log_scalar("train/episode_length",  ep_len,    episode)
        self.metric_logger.log_scalar("train/avg_reward",      avg_reward, episode)

        for agent_id, m in train_metrics.items():
            for k, v in m.items():
                self.metric_logger.log_scalar(f"train/{agent_id}/{k}", v, episode)

    # ─── Checkpointing ───────────────────────────────────────────────────────

    def save_checkpoint(self, tag) -> None:
        for agent_id, agent in self.agents.items():
            path = os.path.join(self.checkpoint_dir, f"{agent_id}_ep{tag}.pt")
            agent.save(path)
        logger.info(f"Checkpoint saved: ep={tag}")

    def load_checkpoint(self, tag, agent_ids: Optional[List[str]] = None) -> None:
        ids = agent_ids or list(self.agents.keys())
        for agent_id in ids:
            path = os.path.join(self.checkpoint_dir, f"{agent_id}_ep{tag}.pt")
            self.agents[agent_id].load(path)
        logger.info(f"Checkpoint loaded: ep={tag}")
