"""
Grid-based strategy environment built on PettingZoo's AEC (Agent Environment Cycle) API.
Supports 2–4 agents competing for territory control on an NxN grid.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers
from typing import Dict, List, Optional, Tuple
import functools

# ─── Tile Types ────────────────────────────────────────────────────────────────
EMPTY       = 0
WALL        = 1
RESOURCE    = 2

# ─── Actions ───────────────────────────────────────────────────────────────────
MOVE_UP     = 0
MOVE_DOWN   = 1
MOVE_LEFT   = 2
MOVE_RIGHT  = 3
ATTACK      = 4
CLAIM       = 5
FORTIFY     = 6
PASS        = 7

NUM_ACTIONS = 8

ACTION_NAMES = {
    MOVE_UP:    "MOVE_UP",
    MOVE_DOWN:  "MOVE_DOWN",
    MOVE_LEFT:  "MOVE_LEFT",
    MOVE_RIGHT: "MOVE_RIGHT",
    ATTACK:     "ATTACK",
    CLAIM:      "CLAIM",
    FORTIFY:    "FORTIFY",
    PASS:       "PASS",
}

MOVE_DELTAS = {
    MOVE_UP:    (-1,  0),
    MOVE_DOWN:  ( 1,  0),
    MOVE_LEFT:  ( 0, -1),
    MOVE_RIGHT: ( 0,  1),
}


def env(**kwargs):
    """Wraps raw env with PettingZoo utility wrappers."""
    raw = GridStrategyEnv(**kwargs)
    raw = wrappers.AssertOutOfBoundsWrapper(raw)
    raw = wrappers.OrderEnforcingWrapper(raw)
    return raw


class GridStrategyEnv(AECEnv):
    """
    Competitive grid-based strategy game for N agents.

    Observation (per agent):
        [grid_map, agent_positions, health_map, territory_map, resources_map]
        Flattened into a 1-D float32 vector of size obs_dim.

    Reward signals:
        - Territorial control  (+0.1 per owned tile per step)
        - Resource collection  (+1.0 per resource picked up)
        - Attacking enemies    (+0.5 per HP dealt)
        - Being attacked       (-0.5 per HP lost)
        - Winning the episode  (+10.0)
        - Losing the episode   (-10.0)
        - Cooperation bonus    (+0.2 when allied agents share adjacent tiles)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "grid_strategy_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        grid_size: int = 12,
        num_agents: int = 2,
        max_steps: int = 500,
        wall_density: float = 0.12,
        resource_density: float = 0.06,
        agent_hp: int = 100,
        attack_damage: int = 20,
        fortify_bonus: int = 10,
        view_radius: int = 4,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        assert 2 <= num_agents <= 4, "Supports 2–4 agents."
        assert grid_size >= 8,       "Grid must be at least 8×8."

        self.grid_size      = grid_size
        self.n_agents       = num_agents
        self.max_steps      = max_steps
        self.wall_density   = wall_density
        self.resource_density = resource_density
        self.max_hp         = agent_hp
        self.attack_damage  = attack_damage
        self.fortify_bonus  = fortify_bonus
        self.view_radius    = view_radius
        self.render_mode    = render_mode
        self._seed          = seed

        # PettingZoo required attributes
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents          = self.possible_agents[:]

        # Observation: flattened (grid_size x grid_size) channels × 5
        obs_channels = 5
        self._obs_size = grid_size * grid_size * obs_channels

        self.observation_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(self._obs_size,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Discrete(NUM_ACTIONS)
            for a in self.possible_agents
        }

        # Internal state (initialized in reset)
        self._grid: Optional[np.ndarray]       = None   # (H, W) tile type
        self._territory: Optional[np.ndarray]  = None   # (H, W) owner index (-1 = unclaimed)
        self._resources: Optional[np.ndarray]  = None   # (H, W) bool resource present
        self._positions: Optional[Dict]        = None   # agent_id → (row, col)
        self._health: Optional[Dict]           = None   # agent_id → int
        self._fortified: Optional[Dict]        = None   # agent_id → bool
        self._step_count: int                  = 0
        self._cumulative_rewards: Dict         = {}
        self._agent_selector: Optional[agent_selector] = None

    # ─── PettingZoo API ──────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observe(self, agent: str) -> np.ndarray:
        return self._build_observation(agent)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> None:
        if seed is not None:
            self._seed = seed
        rng = np.random.default_rng(self._seed)

        self.agents = self.possible_agents[:]
        self._step_count = 0

        # Build grid
        self._grid      = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self._territory = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)
        self._resources = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        self._place_walls(rng)
        self._place_resources(rng)
        self._place_agents(rng)

        self._health    = {a: self.max_hp for a in self.agents}
        self._fortified = {a: False       for a in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards         = {a: 0.0  for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations    = {a: False for a in self.agents}
        self.truncations     = {a: False for a in self.agents}
        self.infos           = {a: {}    for a in self.agents}

    def step(self, action: int) -> None:
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(None)
            return

        self.rewards = {a: 0.0 for a in self.agents}
        self._fortified[agent] = False

        self._execute_action(agent, action)
        self._apply_territory_rewards()
        self._check_termination()

        self._cumulative_rewards[agent] += self.rewards[agent]
        self._step_count += 1

        if self._step_count >= self.max_steps:
            for a in self.agents:
                self.truncations[a] = True

        self.agent_selection = self._agent_selector.next()
        self._deads_step_first()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        elif self.render_mode == "human":
            self._render_human()
        return None

    def close(self) -> None:
        pass

    # ─── Action Execution ────────────────────────────────────────────────────

    def _execute_action(self, agent: str, action: int) -> None:
        agent_idx = int(agent.split("_")[1])
        r, c      = self._positions[agent]

        if action in MOVE_DELTAS:
            dr, dc = MOVE_DELTAS[action]
            nr, nc = r + dr, c + dc
            if self._is_valid_move(nr, nc):
                self._positions[agent] = (nr, nc)
                # Collect resource
                if self._resources[nr, nc]:
                    self._resources[nr, nc] = False
                    self.rewards[agent] += 1.0

        elif action == ATTACK:
            self._do_attack(agent, agent_idx, r, c)

        elif action == CLAIM:
            if self._territory[r, c] != agent_idx:
                self._territory[r, c] = agent_idx
                self.rewards[agent] += 0.3

        elif action == FORTIFY:
            self._fortified[agent] = True  # Halves incoming damage this step

        # PASS → no-op

    def _do_attack(self, attacker: str, attacker_idx: int, r: int, c: int) -> None:
        for target in self.agents:
            if target == attacker:
                continue
            tr, tc = self._positions[target]
            if abs(tr - r) <= 1 and abs(tc - c) <= 1:
                damage = self.attack_damage
                if self._fortified[target]:
                    damage //= 2
                self._health[target] -= damage
                self.rewards[attacker] += 0.5
                self.rewards[target]   -= 0.5
                if self._health[target] <= 0:
                    self._eliminate_agent(target, winner=attacker)

    def _eliminate_agent(self, loser: str, winner: str) -> None:
        self.terminations[loser]   = True
        self.rewards[winner]      += 10.0
        self.rewards[loser]       -= 10.0
        self.agents.remove(loser)

    # ─── Rewards ─────────────────────────────────────────────────────────────

    def _apply_territory_rewards(self) -> None:
        for i, agent in enumerate(self.possible_agents):
            if agent not in self.agents:
                continue
            owned = np.sum(self._territory == i)
            self.rewards[agent] += 0.01 * owned

        # Cooperation bonus: allied agents adjacent to each other
        self._apply_cooperation_bonus()

    def _apply_cooperation_bonus(self) -> None:
        """Positive reward when agents of the same 'team' stand adjacent."""
        # In a competitive 2-agent game there's no team; skip in that case.
        if self.n_agents <= 2:
            return
        # For 4-agent games treat 0+1 vs 2+3 as teams
        teams = [
            [a for a in self.agents if int(a.split("_")[1]) < self.n_agents // 2],
            [a for a in self.agents if int(a.split("_")[1]) >= self.n_agents // 2],
        ]
        for team in teams:
            for i, a in enumerate(team):
                for b in team[i + 1:]:
                    ra, ca = self._positions[a]
                    rb, cb = self._positions[b]
                    if abs(ra - rb) <= 1 and abs(ca - cb) <= 1:
                        self.rewards[a] += 0.2
                        self.rewards[b] += 0.2

    # ─── Termination ─────────────────────────────────────────────────────────

    def _check_termination(self) -> None:
        alive = [a for a in self.agents if not self.terminations[a]]
        if len(alive) == 1:
            self.terminations[alive[0]] = True  # Last agent wins

    # ─── Observation Builder ─────────────────────────────────────────────────

    def _build_observation(self, agent: str) -> np.ndarray:
        g = self.grid_size
        channels = np.zeros((5, g, g), dtype=np.float32)

        # Ch 0: walls
        channels[0] = (self._grid == WALL).astype(np.float32)

        # Ch 1: resources
        channels[1] = self._resources.astype(np.float32)

        # Ch 2: territory (normalized owner index)
        channels[2] = np.where(
            self._territory >= 0,
            self._territory.astype(np.float32) / max(self.n_agents - 1, 1),
            0.0,
        )

        # Ch 3: all agent positions (with health intensity)
        for i, a in enumerate(self.possible_agents):
            if a in self.agents and not self.terminations.get(a, False):
                r, c = self._positions[a]
                channels[3, r, c] = self._health[a] / self.max_hp

        # Ch 4: self position
        if agent in self.agents:
            r, c = self._positions[agent]
            channels[4, r, c] = 1.0

        return channels.flatten()

    # ─── Map Generation ──────────────────────────────────────────────────────

    def _place_walls(self, rng: np.random.Generator) -> None:
        n_walls = int(self.grid_size ** 2 * self.wall_density)
        flat    = rng.choice(self.grid_size ** 2, size=n_walls, replace=False)
        for idx in flat:
            r, c = divmod(int(idx), self.grid_size)
            self._grid[r, c] = WALL

    def _place_resources(self, rng: np.random.Generator) -> None:
        n_res = int(self.grid_size ** 2 * self.resource_density)
        placed = 0
        attempts = 0
        while placed < n_res and attempts < 10_000:
            r = rng.integers(0, self.grid_size)
            c = rng.integers(0, self.grid_size)
            if self._grid[r, c] == EMPTY:
                self._resources[r, c] = True
                placed += 1
            attempts += 1

    def _place_agents(self, rng: np.random.Generator) -> None:
        # Deterministic spawn corners then mid-edges
        spawn_candidates = [
            (1, 1),
            (self.grid_size - 2, self.grid_size - 2),
            (1, self.grid_size - 2),
            (self.grid_size - 2, 1),
        ]
        self._positions = {}
        for i, agent in enumerate(self.possible_agents):
            r, c = spawn_candidates[i % len(spawn_candidates)]
            self._grid[r, c] = EMPTY  # Clear any wall at spawn
            self._positions[agent] = (r, c)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _is_valid_move(self, r: int, c: int) -> bool:
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return False
        if self._grid[r, c] == WALL:
            return False
        # Cannot move onto another agent
        for pos in self._positions.values():
            if pos == (r, c):
                return False
        return True

    def _render_rgb(self) -> np.ndarray:
        cell   = 32
        img    = np.ones((self.grid_size * cell, self.grid_size * cell, 3), dtype=np.uint8) * 230
        colors = [
            (52,  152, 219),   # agent 0 – blue
            (231,  76,  60),   # agent 1 – red
            ( 46, 204, 113),   # agent 2 – green
            (241, 196,  15),   # agent 3 – yellow
        ]
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                y1, y2 = r * cell, (r + 1) * cell
                x1, x2 = c * cell, (c + 1) * cell
                if self._grid[r, c] == WALL:
                    img[y1:y2, x1:x2] = (60, 60, 60)
                elif self._resources[r, c]:
                    img[y1:y2, x1:x2] = (255, 215, 0)
                elif self._territory[r, c] >= 0:
                    base = np.array(colors[self._territory[r, c]])
                    img[y1:y2, x1:x2] = (base * 0.5 + 115).astype(np.uint8)

        for i, agent in enumerate(self.possible_agents):
            if agent in self.agents:
                r, c = self._positions[agent]
                y1, y2 = r * cell + 4, (r + 1) * cell - 4
                x1, x2 = c * cell + 4, (c + 1) * cell - 4
                img[y1:y2, x1:x2] = colors[i]
        return img

    def _render_human(self) -> None:
        symbols = {EMPTY: ".", WALL: "#"}
        agent_chars = {a: str(i) for i, a in enumerate(self.possible_agents)}
        for r in range(self.grid_size):
            row = ""
            for c in range(self.grid_size):
                char = symbols.get(self._grid[r, c], ".")
                for a, pos in self._positions.items():
                    if pos == (r, c):
                        char = agent_chars[a]
                if self._resources[r, c] and char == ".":
                    char = "$"
                row += char + " "
            print(row)
        print()
