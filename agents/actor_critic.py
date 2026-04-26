"""
Actor-Critic network used by both independent and centralized PPO agents.

Architecture:
    Shared encoder → [Actor head, Critic head]
    
    For CTDE (Centralized Training Decentralized Execution):
        - Actor  receives local observation only
        - Critic receives concatenated observations of ALL agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional


def _mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class SharedEncoder(nn.Module):
    """Common feature extractor shared by both actor and critic heads."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = _mlp([obs_dim, hidden_dim, hidden_dim])
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorHead(nn.Module):
    def __init__(self, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = _mlp([hidden_dim, 128, n_actions])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)  # logits


class CriticHead(nn.Module):
    """
    Centralized critic: accepts concatenated observations of all agents.
    During decentralized execution this is not used — only the actor runs.
    """

    def __init__(self, critic_input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = _mlp([critic_input_dim, hidden_dim, hidden_dim, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Full Actor-Critic module.

    Args:
        obs_dim:         Dimension of a single agent's observation.
        n_actions:       Number of discrete actions.
        n_agents:        Total number of agents (for centralized critic).
        hidden_dim:      Width of shared encoder layers.
        centralized:     If True, critic takes joint observations.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_agents: int = 1,
        hidden_dim: int = 256,
        centralized: bool = True,
    ):
        super().__init__()
        self.obs_dim     = obs_dim
        self.n_actions   = n_actions
        self.n_agents    = n_agents
        self.centralized = centralized

        self.encoder = SharedEncoder(obs_dim, hidden_dim)

        self.actor   = ActorHead(hidden_dim, n_actions)

        critic_in    = (obs_dim * n_agents) if centralized else obs_dim
        self.critic  = CriticHead(critic_in, hidden_dim)

    # ─── Forward Helpers ─────────────────────────────────────────────────────

    def get_action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (batch, obs_dim) → logits: (batch, n_actions)"""
        features = self.encoder(obs)
        return self.actor(features)

    def get_value(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """
        joint_obs: (batch, obs_dim * n_agents) for centralized critic,
                   or (batch, obs_dim) for decentralized.
        Returns:   (batch,)
        """
        return self.critic(joint_obs)

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Returns:
            action:    (batch,) int64
            log_prob:  (batch,) float32
            entropy:   (batch,) float32
        """
        logits = self.get_action_logits(obs)
        dist   = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        joint_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probs and values for a batch of (obs, action) pairs.

        Args:
            obs:       (batch, obs_dim)
            actions:   (batch,) int64
            joint_obs: (batch, obs_dim * n_agents) – required for centralized critic

        Returns:
            log_prob, entropy, value
        """
        logits   = self.get_action_logits(obs)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()

        if joint_obs is None:
            joint_obs = obs
        value = self.get_value(joint_obs)

        return log_prob, entropy, value

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: returns (logits, value) given a single obs."""
        features = self.encoder(obs)
        logits   = self.actor(features)
        value    = self.critic(obs)   # decentralized fallback
        return logits, value


# ─── Recurrent Actor-Critic (optional upgrade) ────────────────────────────────

class RecurrentActorCritic(nn.Module):
    """
    GRU-based Actor-Critic for handling partial observability.
    Drop-in replacement for ActorCritic with hidden state support.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_agents: int = 1,
        hidden_dim: int = 256,
        gru_layers: int = 1,
        centralized: bool = True,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.centralized = centralized
        self.n_agents    = n_agents

        self.input_fc = nn.Linear(obs_dim, hidden_dim)
        self.gru      = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_layers, batch_first=True)

        self.actor    = ActorHead(hidden_dim, n_actions)

        critic_in     = (obs_dim * n_agents) if centralized else hidden_dim
        self.critic   = CriticHead(critic_in, hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs:    (batch, seq_len, obs_dim) or (batch, obs_dim)
        hidden: (num_layers, batch, hidden_dim) or None

        Returns: logits, value, new_hidden
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)   # add seq_len=1

        x  = F.relu(self.input_fc(obs))
        if hidden is None:
            gru_out, new_hidden = self.gru(x)
        else:
            gru_out, new_hidden = self.gru(x, hidden)

        feat   = gru_out[:, -1]   # last step
        logits = self.actor(feat)
        value  = self.critic(feat)
        return logits, value, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)
