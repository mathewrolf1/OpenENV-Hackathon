"""Puff-style reward calculator: defensive spacing, Rest kills, weave-in/out.

Same interface as CompetitiveMeleeReward — works with both libmelee PlayerState
(Dolphin) and physics.state.CharacterState (sim).  Call reset() at episode
start, step(p1, p2, done, winner) each frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _g(obj: Any, name: str, default: Any = 0) -> Any:
    return getattr(obj, name, default)


def _pos_x(p: Any) -> float:
    if hasattr(p, "position") and hasattr(p.position, "x"):
        return float(p.position.x)
    return float(_g(p, "x", 0.0))


def _pos_y(p: Any) -> float:
    if hasattr(p, "position") and hasattr(p.position, "y"):
        return float(p.position.y)
    return float(_g(p, "y", 0.0))


def _action_name(p: Any) -> str:
    act = _g(p, "action", None)
    if act is None:
        return ""
    if hasattr(act, "name"):
        return act.name.upper()
    return str(act).upper()


def _is_rest_sleep(p: Any) -> bool:
    name = _action_name(p)
    return name in ("REST_SLEEP", "25", "FURA_SLEEP", "REST_WAIT")


def _attack_connected(p: Any) -> bool:
    return bool(_g(p, "attack_connected", False))


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

@dataclass
class PuffWeights:
    # Base
    damage_dealt: float = 0.01
    damage_taken: float = 0.01
    stock_taken: float = 0.5
    win_bonus: float = 1.0
    loss_penalty: float = 1.0

    # Spacing (Gaussian sweet-spot)
    spacing_ideal: float = 10.0
    spacing_inner: float = 5.0
    spacing_outer: float = 20.0
    spacing_bonus: float = 0.003
    spacing_sigma: float = 5.0
    spacing_approach_coef: float = 0.005

    # Damage amplification (on top of base)
    damage_extra: float = 0.04

    # Rest
    rest_kill_bonus: float = 0.5
    rest_miss_penalty: float = 0.002


# ---------------------------------------------------------------------------
# Per-frame state
# ---------------------------------------------------------------------------

@dataclass
class _PuffFrameState:
    p1_percent: float = 0.0
    p2_percent: float = 0.0
    p1_stocks: int = 4
    p2_stocks: int = 4
    prev_distance: float = 20.0


# ---------------------------------------------------------------------------
# PuffReward
# ---------------------------------------------------------------------------


class PuffReward:
    """Stateful per-episode Puff reward calculator.

    Mirrors PuffRewardWrapper logic but decoupled from Gymnasium.
    Works with any player object that has percent/stock/x/y/action attrs.
    """

    __slots__ = ("w", "_prev", "_ep_reward", "_ep_damage_dealt", "_ep_stocks_won")

    def __init__(self, weights: Optional[PuffWeights] = None):
        self.w = weights or PuffWeights()
        self._prev = _PuffFrameState()
        self._ep_reward: float = 0.0
        self._ep_damage_dealt: float = 0.0
        self._ep_stocks_won: int = 0

    def reset(self) -> None:
        self._prev = _PuffFrameState()
        self._ep_reward = 0.0
        self._ep_damage_dealt = 0.0
        self._ep_stocks_won = 0

    def step(
        self,
        p1: Any,
        p2: Any,
        done: bool = False,
        winner: Optional[int] = None,
    ) -> tuple[float, Dict[str, float]]:
        w = self.w
        prev = self._prev

        p1_pct = float(p1.percent) if p1 else 0.0
        p2_pct = float(p2.percent) if p2 else 0.0
        p1_stk = int(p1.stock) if p1 else 4
        p2_stk = int(p2.stock) if p2 else 4

        # ---- 1. Base deltas ----
        damage_dealt = max(0.0, p2_pct - prev.p2_percent)
        damage_taken = max(0.0, p1_pct - prev.p1_percent)
        stocks_taken = max(0, prev.p2_stocks - p2_stk)

        r_damage = damage_dealt * w.damage_dealt - damage_taken * w.damage_taken
        r_stocks = stocks_taken * w.stock_taken

        r_terminal = 0.0
        if done:
            if winner == 0:
                r_terminal = w.win_bonus
            elif winner == 1:
                r_terminal = -w.loss_penalty

        # ---- 2. Spacing (Gaussian sweet-spot) ----
        r_spacing = 0.0
        if p1 is not None and p2 is not None:
            distance_now = abs(_pos_x(p1) - _pos_x(p2)) + abs(_pos_y(p1) - _pos_y(p2))

            if w.spacing_inner <= distance_now <= w.spacing_outer:
                deviation = distance_now - w.spacing_ideal
                r_spacing = w.spacing_bonus * math.exp(
                    -0.5 * (deviation / w.spacing_sigma) ** 2
                )
            elif distance_now > w.spacing_outer:
                distance_change = prev.prev_distance - distance_now
                r_spacing = distance_change * w.spacing_approach_coef

            prev.prev_distance = distance_now

        # ---- 3. Extra damage amplification ----
        r_damage_extra = 0.0
        if damage_dealt > 0:
            r_damage_extra = damage_dealt * w.damage_extra

        # ---- 4. Rest kill bonus ----
        r_rest_kill = 0.0
        if p1 is not None and _is_rest_sleep(p1) and _attack_connected(p1) and stocks_taken > 0:
            r_rest_kill = w.rest_kill_bonus

        # ---- 5. Missed Rest penalty ----
        r_rest_miss = 0.0
        if p1 is not None and _is_rest_sleep(p1) and not _attack_connected(p1):
            r_rest_miss = -w.rest_miss_penalty

        # ---- Aggregate ----
        reward = (
            r_damage
            + r_stocks
            + r_terminal
            + r_spacing
            + r_damage_extra
            + r_rest_kill
            + r_rest_miss
        )

        # ---- Update prev state ----
        prev.p1_percent = p1_pct
        prev.p2_percent = p2_pct
        prev.p1_stocks = p1_stk
        prev.p2_stocks = p2_stk

        # ---- Episode accumulators ----
        self._ep_reward += reward
        self._ep_damage_dealt += damage_dealt
        self._ep_stocks_won += stocks_taken

        info: Dict[str, float] = {
            "reward_damage": r_damage,
            "reward_stocks": r_stocks,
            "reward_terminal": r_terminal,
            "reward_spacing": r_spacing,
            "reward_damage_extra": r_damage_extra,
            "reward_rest_kill": r_rest_kill,
            "reward_rest_miss": r_rest_miss,
            "episode_reward": self._ep_reward,
            "total_damage_dealt": self._ep_damage_dealt,
            "stocks_won": float(self._ep_stocks_won),
        }

        return reward, info
