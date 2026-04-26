"""
Self-Play Manager — maintains a pool of agent snapshots and samples
opponents for curriculum training.

Strategies supported:
    - latest:       Always play against the most recent checkpoint.
    - random:       Sample a random past checkpoint.
    - prioritized:  Sample opponents that beat us more often (harder curricula).
"""

import os
import copy
import random
import logging
import numpy as np
from typing import Dict, List, Optional
from collections import deque

from agents.ppo_agent import PPOAgent, PPOConfig

logger = logging.getLogger(__name__)


class AgentSnapshot:
    """Lightweight container for a frozen copy of an agent's network weights."""

    def __init__(self, agent_id: str, step: int, state_dict: dict):
        self.agent_id   = agent_id
        self.step       = step
        self.state_dict = state_dict
        self.elo        = 1000.0   # Initial ELO rating

    def __repr__(self) -> str:
        return f"Snapshot(id={self.agent_id}, step={self.step}, elo={self.elo:.1f})"


class SelfPlayManager:
    """
    Manages an opponent pool and ELO ratings for self-play training.

    Usage:
        spm = SelfPlayManager(strategy="prioritized")
        spm.add_snapshot(agent, global_step)
        opponent = spm.sample_opponent(agent_id)
        # ... play episode ...
        spm.update_elo(winner_id, loser_id)
    """

    def __init__(
        self,
        strategy: str = "random",
        pool_size: int = 20,
        snapshot_interval: int = 50,
        elo_k: float = 32.0,
    ):
        assert strategy in ("latest", "random", "prioritized")
        self.strategy           = strategy
        self.pool_size          = pool_size
        self.snapshot_interval  = snapshot_interval
        self.elo_k              = elo_k

        self._pool: Dict[str, deque]  = {}       # agent_id → deque of snapshots
        self._step_counter: int       = 0

    # ─── Snapshot Management ─────────────────────────────────────────────────

    def add_snapshot(self, agent: PPOAgent, global_step: int) -> None:
        aid = agent.agent_id
        if aid not in self._pool:
            self._pool[aid] = deque(maxlen=self.pool_size)

        snap = AgentSnapshot(
            agent_id   = aid,
            step       = global_step,
            state_dict = copy.deepcopy(agent.network.state_dict()),
        )
        self._pool[aid].append(snap)
        logger.debug(f"Snapshot added: {snap}")

    def should_snapshot(self, episode: int) -> bool:
        return episode % self.snapshot_interval == 0

    # ─── Opponent Sampling ───────────────────────────────────────────────────

    def sample_opponent(
        self,
        agent_id: str,
        exclude_agent_id: Optional[str] = None,
    ) -> Optional[AgentSnapshot]:
        """
        Sample an opponent snapshot according to the chosen strategy.
        If the pool is empty, returns None (fall back to latest weights).
        """
        candidates = []
        for aid, pool in self._pool.items():
            if aid == exclude_agent_id:
                continue
            candidates.extend(list(pool))

        if not candidates:
            return None

        if self.strategy == "latest":
            return max(candidates, key=lambda s: s.step)

        if self.strategy == "random":
            return random.choice(candidates)

        if self.strategy == "prioritized":
            # Higher ELO opponents sampled with higher probability
            elos = np.array([s.elo for s in candidates], dtype=np.float64)
            elos = elos - elos.min() + 1.0   # shift to positive
            probs = elos / elos.sum()
            return np.random.choice(candidates, p=probs)

    def load_snapshot_into_agent(
        self,
        snapshot: AgentSnapshot,
        agent: PPOAgent,
    ) -> None:
        """Overwrite agent's network with snapshot weights (no grad)."""
        agent.network.load_state_dict(snapshot.state_dict)
        for p in agent.network.parameters():
            p.requires_grad_(False)

    # ─── ELO Updates ─────────────────────────────────────────────────────────

    def update_elo(
        self,
        winner_snapshot: AgentSnapshot,
        loser_snapshot: AgentSnapshot,
    ) -> None:
        ra, rb = winner_snapshot.elo, loser_snapshot.elo
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea
        winner_snapshot.elo += self.elo_k * (1 - ea)
        loser_snapshot.elo  += self.elo_k * (0 - eb)

    # ─── Stats ───────────────────────────────────────────────────────────────

    def pool_stats(self) -> Dict:
        stats = {}
        for aid, pool in self._pool.items():
            snaps = list(pool)
            stats[aid] = {
                "size":     len(snaps),
                "avg_elo":  np.mean([s.elo for s in snaps]) if snaps else 0.0,
                "max_elo":  max((s.elo for s in snaps), default=0.0),
                "latest_step": max((s.step for s in snaps), default=0),
            }
        return stats
