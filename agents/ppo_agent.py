"""
Proximal Policy Optimization (PPO) with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Gradient norm clipping
    - Entropy bonus for exploration
    - Mini-batch updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from agents.actor_critic import ActorCritic


@dataclass
class PPOConfig:
    # Hyper-parameters
    lr:               float = 3e-4
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_eps:         float = 0.2
    value_clip_eps:   float = 0.2
    entropy_coef:     float = 0.01
    value_loss_coef:  float = 0.5
    max_grad_norm:    float = 0.5

    # Training loop
    n_epochs:         int   = 8
    batch_size:       int   = 256
    rollout_steps:    int   = 2048

    # Architecture
    hidden_dim:       int   = 256
    centralized:      bool  = True

    # Misc
    normalize_adv:    bool  = True
    device:           str   = "cpu"


@dataclass
class RolloutBuffer:
    """Stores transitions collected from one rollout for a single agent."""
    obs:         List[np.ndarray]   = field(default_factory=list)
    joint_obs:   List[np.ndarray]   = field(default_factory=list)
    actions:     List[int]          = field(default_factory=list)
    log_probs:   List[float]        = field(default_factory=list)
    rewards:     List[float]        = field(default_factory=list)
    values:      List[float]        = field(default_factory=list)
    dones:       List[bool]         = field(default_factory=list)

    def clear(self) -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, [])

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        obs: np.ndarray,
        joint_obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.joint_obs.append(joint_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)


class PPOAgent:
    """
    Single PPO agent wrapping an ActorCritic network.

    Each agent maintains its OWN network but the critic can receive
    joint observations from all agents (CTDE pattern).
    """

    def __init__(
        self,
        agent_id: str,
        obs_dim: int,
        n_actions: int,
        n_agents: int,
        cfg: PPOConfig,
    ):
        self.agent_id  = agent_id
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.n_agents  = n_agents
        self.cfg       = cfg
        self.device    = torch.device(cfg.device)

        self.network = ActorCritic(
            obs_dim     = obs_dim,
            n_actions   = n_actions,
            n_agents    = n_agents,
            hidden_dim  = cfg.hidden_dim,
            centralized = cfg.centralized,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=cfg.lr, eps=1e-5)
        self.buffer    = RolloutBuffer()

        # Logging
        self.train_stats: Dict[str, float] = {}

    # ─── Interaction ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        joint_obs: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Returns:
            action   (int)
            log_prob (float)
            value    (float)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if joint_obs is None:
            joint_obs = obs
        jo_t = torch.as_tensor(joint_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        action, log_prob, _ = self.network.act(obs_t, deterministic=deterministic)
        value               = self.network.get_value(jo_t)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    def store_transition(
        self,
        obs: np.ndarray,
        joint_obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.buffer.add(obs, joint_obs, action, log_prob, reward, value, done)

    # ─── Training ────────────────────────────────────────────────────────────

    def compute_gae(
        self,
        last_value: float,
        last_done: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        cfg      = self.cfg
        rewards  = self.buffer.rewards
        values   = self.buffer.values + [last_value]
        dones    = self.buffer.dones  + [last_done]
        T        = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + cfg.gamma * values[t + 1] * (1 - dones[t + 1]) - values[t]
            gae   = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[t + 1]) * gae
            advantages[t] = gae

        returns = advantages + np.array(self.buffer.values, dtype=np.float32)
        return (
            torch.tensor(advantages, dtype=torch.float32, device=self.device),
            torch.tensor(returns,    dtype=torch.float32, device=self.device),
        )

    def update(self, last_value: float = 0.0, last_done: bool = True) -> Dict[str, float]:
        """
        Run PPO update epochs on buffered rollout data.
        Returns dict of logged metrics.
        """
        if len(self.buffer) == 0:
            return {}

        cfg = self.cfg

        advantages, returns = self.compute_gae(last_value, last_done)

        # Convert buffer to tensors
        obs_arr      = np.array(self.buffer.obs,       dtype=np.float32)
        jo_arr       = np.array(self.buffer.joint_obs, dtype=np.float32)
        actions_arr  = np.array(self.buffer.actions,   dtype=np.int64)
        old_lp_arr   = np.array(self.buffer.log_probs, dtype=np.float32)
        old_val_arr  = np.array(self.buffer.values,    dtype=np.float32)

        obs_t      = torch.tensor(obs_arr,     device=self.device)
        jo_t       = torch.tensor(jo_arr,      device=self.device)
        actions_t  = torch.tensor(actions_arr, device=self.device)
        old_lp_t   = torch.tensor(old_lp_arr,  device=self.device)
        old_val_t  = torch.tensor(old_val_arr, device=self.device)

        if cfg.normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = len(obs_t)
        total_pg_loss = total_v_loss = total_ent = 0.0
        n_updates = 0

        for _ in range(cfg.n_epochs):
            idx = torch.randperm(N, device=self.device)

            for start in range(0, N, cfg.batch_size):
                b = idx[start: start + cfg.batch_size]

                log_prob, entropy, value = self.network.evaluate_actions(
                    obs_t[b], actions_t[b], jo_t[b]
                )

                # Policy loss (clipped surrogate)
                ratio    = torch.exp(log_prob - old_lp_t[b])
                adv_b    = advantages[b]
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_clipped   = old_val_t[b] + torch.clamp(
                    value - old_val_t[b], -cfg.value_clip_eps, cfg.value_clip_eps
                )
                v_loss_raw  = (value     - returns[b]).pow(2)
                v_loss_clip = (v_clipped - returns[b]).pow(2)
                v_loss      = 0.5 * torch.max(v_loss_raw, v_loss_clip).mean()

                entropy_loss = -entropy.mean()
                loss = pg_loss + cfg.value_loss_coef * v_loss + cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += float(pg_loss.item())
                total_v_loss  += float(v_loss.item())
                total_ent     += float(-entropy_loss.item())
                n_updates     += 1

        self.buffer.clear()

        stats = {
            "policy_loss": total_pg_loss / max(n_updates, 1),
            "value_loss":  total_v_loss  / max(n_updates, 1),
            "entropy":     total_ent     / max(n_updates, 1),
        }
        self.train_stats = stats
        return stats

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save(
            {
                "network":   self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
