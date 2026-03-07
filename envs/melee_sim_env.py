"""Gymnasium environment wrapping the Melee-like physics simulator.

Observation and action spaces are designed to match what a future
Dolphin/libmelee-backed env would expose, so policies transfer directly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from physics.constants import Action, CHAR, MAX_FRAMES, STAGE
from physics.simulator import Simulator
from physics.state import CharacterState, GameState


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

_OBS_PER_PLAYER = 13  # see _player_obs()
OBS_DIM = _OBS_PER_PLAYER * 2  # two players


def _player_obs(p: CharacterState) -> np.ndarray:
    """Flatten one player's state into a fixed-size float vector."""
    return np.array(
        [
            p.x / 100.0,
            p.y / 100.0,
            p.speed_x_self / 5.0,
            p.speed_y_self / 5.0,
            p.speed_x_attack / 5.0,
            p.speed_y_attack / 5.0,
            p.percent / 200.0,
            float(p.stock) / 4.0,
            float(p.on_ground),
            float(p.facing_right),
            float(p.action) / float(Action.NUM_ACTIONS),
            float(p.action_frame) / 60.0,
            float(p.hitstun_frames_left) / 60.0,
        ],
        dtype=np.float32,
    )


def _build_obs(gs: GameState, player_idx: int) -> np.ndarray:
    """Build observation from the perspective of ``player_idx``."""
    me = gs.players[player_idx]
    opp = gs.players[1 - player_idx]
    return np.concatenate([_player_obs(me), _player_obs(opp)])


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

STICK_X_BINS = [-1.0, -0.6, 0.0, 0.6, 1.0]
STICK_Y_BINS = [-1.0, 0.0, 0.5, 1.0]


def _decode_action(action: np.ndarray) -> Dict:
    """Convert MultiDiscrete action array to simulator dict."""
    sx_idx, sy_idx, jump, attack, grab, special = (
        int(action[0]), int(action[1]), int(action[2]),
        int(action[3]), int(action[4]), int(action[5]),
    )
    return {
        "stick_x": STICK_X_BINS[sx_idx],
        "stick_y": STICK_Y_BINS[sy_idx],
        "jump": bool(jump),
        "attack": bool(attack),
        "grab": bool(grab),
        "special": bool(special),
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MeleeSimEnv(gym.Env):
    """Single-agent Melee sim env.  Player 0 is the agent; player 1 is
    controlled by ``opponent_fn`` (defaults to a standing dummy).

    Observation: float32 vector of length ``OBS_DIM``.
    Action: MultiDiscrete([5, 4, 2, 2, 2, 2]) —
        stick_x (5 bins), stick_y (4 bins), jump, attack, grab, special.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_fn=None,
        max_frames: int = MAX_FRAMES,
        render_mode=None,
    ):
        super().__init__()
        self.sim = Simulator()
        self.opponent_fn = opponent_fn or self._dummy_opponent
        self.max_frames = max_frames

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([
            len(STICK_X_BINS),  # stick_x
            len(STICK_Y_BINS),  # stick_y
            2,                  # jump
            2,                  # attack
            2,                  # grab
            2,                  # special (down+special = Rest)
        ])

        self._state: Optional[GameState] = None
        self._prev_percent_self: float = 0.0
        self._prev_percent_opp: float = 0.0
        self._prev_opp_stocks: int = 4

    # ---- Gym API --------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._state = self.sim.reset(seed=seed)
        self._prev_percent_self = 0.0
        self._prev_percent_opp = 0.0
        self._prev_opp_stocks = 4
        obs = _build_obs(self._state, 0)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        agent_action = _decode_action(action)
        opp_action = self.opponent_fn(self._state, 1)

        self._state = self.sim.step(self._state, [agent_action, opp_action])

        obs = _build_obs(self._state, 0)
        reward = self._compute_reward()
        terminated = self._state.done
        truncated = self._state.frame >= self.max_frames and not terminated

        info: Dict[str, Any] = {
            "frame": self._state.frame,
            "p0_percent": self._state.players[0].percent,
            "p1_percent": self._state.players[1].percent,
            "p0_stocks": self._state.players[0].stock,
            "p1_stocks": self._state.players[1].stock,
        }
        if terminated:
            info["winner"] = self._state.winner

        return obs, reward, terminated, truncated, info

    # ---- Reward ----------------------------------------------------------

    def _compute_reward(self) -> float:
        me = self._state.players[0]
        opp = self._state.players[1]

        damage_dealt = opp.percent - self._prev_percent_opp
        damage_taken = me.percent - self._prev_percent_self
        stocks_taken = self._prev_opp_stocks - opp.stock

        self._prev_percent_opp = opp.percent
        self._prev_percent_self = me.percent
        self._prev_opp_stocks = opp.stock

        reward = 0.0
        reward += damage_dealt * 0.01
        reward -= damage_taken * 0.01

        if stocks_taken > 0:
            reward += 0.5 * stocks_taken
            is_rest = getattr(me, "_current_move_name", "") == "rest"
            if is_rest:
                reward += 0.5  # extra bonus for a Rest kill

        # Per-frame penalty while stuck in Rest sleep after a miss
        if (me.action == Action.REST_SLEEP
                and not me.attack_connected):
            reward -= 0.002

        if self._state.done:
            if self._state.winner == 0:
                reward += 1.0
            elif self._state.winner == 1:
                reward -= 1.0

        return reward

    # ---- Default opponent ------------------------------------------------

    @staticmethod
    def _dummy_opponent(state: GameState, idx: int) -> Dict:
        """Standing dummy that does nothing."""
        return {"stick_x": 0.0, "stick_y": 0.0, "jump": False, "attack": False, "grab": False, "special": False}
