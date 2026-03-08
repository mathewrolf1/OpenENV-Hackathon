"""Gymnasium environment wrapping the Melee-like physics simulator.

Observation and action spaces are designed to match what a future
Dolphin/libmelee-backed env would expose, so policies transfer directly.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Tuple

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

# Action factorisation: [stick_x, stick_y, jump, attack, grab, special]
ACTION_NVEC = [5, 4, 2, 2, 2, 2]
ACTION_FLAT = 5 * 4 * 2 * 2 * 2 * 2  # 320 — used for Discrete action space


def _decode_action(action: np.ndarray) -> Dict:
    """Convert 6-element action index array to simulator dict (used by opponents)."""
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


def _decode_flat_action(idx: int) -> Dict:
    """Decode a flat Discrete(320) integer to a simulator action dict."""
    indices = []
    for n in reversed(ACTION_NVEC):
        indices.append(idx % n)
        idx //= n
    sx_idx, sy_idx, jump, attack, grab, special = reversed(indices)
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
    Action: Discrete(320) — flat encoding of [stick_x(5), stick_y(4), jump, attack, grab, special].
        Use _decode_flat_action() to inspect; _decode_action() still works for 6-element arrays
        (used by opponent loaders).
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
        # Flat Discrete(320) so TorchRL ParallelEnv shared-memory works correctly.
        # The 320 actions encode [stick_x(5) × stick_y(4) × jump(2) × attack(2) × grab(2) × special(2)].
        self.action_space = spaces.Discrete(ACTION_FLAT)

        self._state: Optional[GameState] = None
        self._prev_percent_self: float = 0.0
        self._prev_percent_opp: float = 0.0
        self._prev_opp_stocks: int = 4

        self._ep_rest_attempts: int = 0
        self._ep_rest_hits: int = 0
        self._ep_rest_kills: int = 0
        self._ep_damage_dealt: float = 0.0
        self._ep_damage_taken: float = 0.0
        self._ep_stocks_taken: int = 0
        self._ep_stocks_lost: int = 0
        self._prev_self_stocks: int = 4

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
        self._prev_self_stocks = 4

        self._ep_rest_attempts = 0
        self._ep_rest_hits = 0
        self._ep_rest_kills = 0
        self._ep_damage_dealt = 0.0
        self._ep_damage_taken = 0.0
        self._ep_stocks_taken = 0
        self._ep_stocks_lost = 0

        obs = _build_obs(self._state, 0)
        return obs, {}

    def step(
        self, action
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Accept flat int (Discrete(320)) or numpy/tensor scalar
        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        flat_idx = int(np.asarray(action).flat[0])
        agent_action = _decode_flat_action(flat_idx)
        opp_action = self.opponent_fn(self._state, 1)

        prev_opp_stocks = self._state.players[1].stock
        prev_self_stocks = self._state.players[0].stock

        self._state = self.sim.step(self._state, [agent_action, opp_action])

        me = self._state.players[0]
        opp = self._state.players[1]

        if agent_action.get("special") and agent_action.get("stick_y", 0.0) < -0.3:
            self._ep_rest_attempts += 1
        if (me.action == Action.REST_SLEEP and me.attack_connected
                and getattr(me, "_current_move_name", "") == "rest"
                and me.action_frame == 0):
            self._ep_rest_hits += 1

        stocks_taken = prev_opp_stocks - opp.stock
        stocks_lost = prev_self_stocks - me.stock
        if stocks_taken > 0:
            self._ep_stocks_taken += stocks_taken
            if getattr(me, "_current_move_name", "") == "rest":
                self._ep_rest_kills += stocks_taken
        if stocks_lost > 0:
            self._ep_stocks_lost += stocks_lost

        dmg_dealt = opp.percent - self._prev_percent_opp
        dmg_taken = me.percent - self._prev_percent_self
        if dmg_dealt > 0:
            self._ep_damage_dealt += dmg_dealt
        if dmg_taken > 0:
            self._ep_damage_taken += dmg_taken

        obs = _build_obs(self._state, 0)
        reward = self._compute_reward()
        terminated = self._state.done
        truncated = self._state.frame >= self.max_frames and not terminated

        info: Dict[str, Any] = {
            "frame": self._state.frame,
            "p0_percent": me.percent,
            "p1_percent": opp.percent,
            "p0_stocks": me.stock,
            "p1_stocks": opp.stock,
        }
        if terminated:
            info["winner"] = self._state.winner

        if terminated or truncated:
            info["episode_metrics"] = {
                "rest_attempts": self._ep_rest_attempts,
                "rest_hits": self._ep_rest_hits,
                "rest_kills": self._ep_rest_kills,
                "damage_dealt": self._ep_damage_dealt,
                "damage_taken": self._ep_damage_taken,
                "stocks_taken": self._ep_stocks_taken,
                "stocks_lost": self._ep_stocks_lost,
            }

        return obs, reward, terminated, truncated, info

    # ---- Reward (base signals only — wrappers add style-specific shaping) --

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


# ---------------------------------------------------------------------------
# Self-play opponent
# ---------------------------------------------------------------------------


class SelfPlayOpponent:
    """Wraps a frozen PPO policy as an ``opponent_fn`` for MeleeSimEnv.

    Call ``update_from_model(model)`` to copy the current training policy's
    weights into the opponent.  The opponent then runs inference each frame
    to choose its actions — same observation/action format as the agent.
    """

    def __init__(self):
        self._model = None

    def update_from_model(self, model) -> None:
        """Snapshot the policy by saving/loading through a temp file."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            tmppath = f.name
        try:
            model.save(tmppath)
            from stable_baselines3 import PPO
            self._model = PPO.load(tmppath)
            self._model.policy.set_training_mode(False)
        finally:
            import os
            os.unlink(tmppath)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def __call__(self, state: GameState, idx: int) -> Dict:
        if self._model is None:
            return MeleeSimEnv._dummy_opponent(state, idx)

        obs = _build_obs(state, idx)
        action, _ = self._model.predict(obs, deterministic=False)
        return _decode_action(action)
