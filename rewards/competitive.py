"""Unified competitive reward calculator using deep libmelee variables.

Replaces the separate Mango/Puff reward shaping with a single class that
leverages ECB, shield_strength, hitstun, off_stage, and the 5-speed system
for fine-grained reward signals.

Designed for H100: all per-frame math is scalar (no allocations), and the
decomposed reward dict enables per-component W&B logging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Lightweight protocol so the class works with both libmelee PlayerState
# (Dolphin) and physics.state.CharacterState (sim) without importing either.
# ---------------------------------------------------------------------------

@runtime_checkable
class _PlayerLike(Protocol):
    percent: float
    stock: int
    on_ground: bool


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


def _ecb_bottom_y(p: Any) -> float:
    ecb_bottom = _g(p, "ecb_bottom", None)
    if ecb_bottom is None:
        return _pos_y(p)
    if hasattr(ecb_bottom, "y"):
        return float(ecb_bottom.y)
    if isinstance(ecb_bottom, (tuple, list)) and len(ecb_bottom) >= 2:
        return float(ecb_bottom[1])
    return _pos_y(p)


def _shield(p: Any) -> float:
    return float(_g(p, "shield_strength", 60.0))


def _hitstun(p: Any) -> int:
    return int(_g(p, "hitstun_frames_left", 0))


def _hitlag(p: Any) -> int:
    return int(_g(p, "hitlag_left", 0))


def _off_stage(p: Any) -> bool:
    return bool(_g(p, "off_stage", False))


def _jumps_left(p: Any) -> int:
    return int(_g(p, "jumps_left", 2))


def _speed_ground_x(p: Any) -> float:
    v = _g(p, "speed_ground_x_self", None)
    if v is not None:
        return float(v)
    return float(_g(p, "speed_x_self", 0.0))


def _speed_air_x(p: Any) -> float:
    v = _g(p, "speed_air_x_self", None)
    if v is not None:
        return float(v)
    return float(_g(p, "speed_x_self", 0.0))


# ---------------------------------------------------------------------------
# Stage geometry constants (Final Destination)
# ---------------------------------------------------------------------------

STAGE_LEFT_EDGE = -68.4
STAGE_RIGHT_EDGE = 68.4
STAGE_FLOOR_Y = 0.0


def _nearest_edge_x(px: float) -> float:
    if px <= 0:
        return STAGE_LEFT_EDGE
    return STAGE_RIGHT_EDGE


def _edge_distance(p: Any) -> float:
    """Horizontal distance from ECB bottom to nearest ledge."""
    px = _pos_x(p)
    edge_x = _nearest_edge_x(px)
    return abs(px - edge_x)


# ---------------------------------------------------------------------------
# Reward weights (all tuneable)
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    # Base
    damage_dealt: float = 0.05
    damage_taken: float = 0.000
    stock_taken: float = 1.0
    stock_lost: float = 5.0
    win_bonus: float = 2.0
    loss_penalty: float = 1.0

    # Kinetic recovery (off-stage)
    recovery_scale: float = 0.03

    # Shield pressure 2.0
    shield_pressure: float = 0.05

    # Hitstun follow-up (combo bonus)
    combo_bonus: float = 0.06

    # Edge-guarding
    edgeguard_bonus: float = 0.08

    # Active spacing (velocity while close)
    velocity_bonus_scale: float = 0.02
    velocity_proximity: float = 50.0

    # Approach: reward for closing distance to opponent
    approach_scale: float = 0.008

    # Proximity: per-frame reward for being near opponent
    proximity_bonus: float = 0.003
    proximity_threshold: float = 25.0

    # Anti-SD: survival incentives
    existence_reward: float = 0.002
    off_stage_penalty: float = -0.015
    blastzone_threshold: float = 120.0
    blastzone_penalty_scale: float = -0.0005


# ---------------------------------------------------------------------------
# Per-frame state snapshot for delta tracking
# ---------------------------------------------------------------------------

@dataclass
class _FrameState:
    p1_percent: float = 0.0
    p2_percent: float = 0.0
    p1_stocks: int = 4
    p2_stocks: int = 4
    p2_shield: float = 60.0
    p1_edge_dist: float = 0.0
    p1_off_stage: bool = False
    distance: float = 50.0


# ---------------------------------------------------------------------------
# CompetitiveMeleeReward
# ---------------------------------------------------------------------------


class CompetitiveMeleeReward:
    """Stateful per-episode reward calculator.

    Call ``reset()`` at episode start, ``step(p1, p2, done, winner)`` each
    frame.  Returns ``(total_reward, info_dict)`` where ``info_dict``
    decomposes every reward component for W&B logging.

    Works with both libmelee ``PlayerState`` objects (Dolphin) and
    ``physics.state.CharacterState`` objects (sim).
    """

    __slots__ = ("w", "_prev", "_ep_reward", "_ep_damage_dealt", "_ep_stocks_won")

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.w = weights or RewardWeights()
        self._prev = _FrameState()
        self._ep_reward: float = 0.0
        self._ep_damage_dealt: float = 0.0
        self._ep_stocks_won: int = 0

    def reset(self) -> None:
        self._prev = _FrameState()
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
        """Compute reward for one frame.

        Args:
            p1: Agent PlayerState (port 1 / player index 0).
            p2: Opponent PlayerState (port 2 / player index 1).
            done: Whether the episode ended this frame.
            winner: 0 or 1 (player index) if known, else None.
                    For libmelee use 0 for port 1, 1 for port 2.

        Returns:
            (reward, info) where info has per-component breakdowns.
        """
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
        stocks_lost = max(0, prev.p1_stocks - p1_stk)

        r_damage = damage_dealt * w.damage_dealt - damage_taken * w.damage_taken
        r_stocks = stocks_taken * w.stock_taken - stocks_lost * w.stock_lost

        r_terminal = 0.0
        if done:
            if winner == 0:
                r_terminal = w.win_bonus
            elif winner == 1:
                r_terminal = -w.loss_penalty

        # ---- 2. Kinetic recovery ----
        r_recovery = 0.0
        if p1 is not None:
            p1_off = _off_stage(p1)
            p1_edge = _edge_distance(p1)
            if p1_off and prev.p1_off_stage:
                dist_delta = prev.p1_edge_dist - p1_edge
                if dist_delta > 0:
                    jumps = max(1, _jumps_left(p1))
                    r_recovery = dist_delta * w.recovery_scale * (jumps / 5.0)
        else:
            p1_off = False
            p1_edge = 0.0

        # ---- 3. Shield pressure 2.0 ----
        r_pressure = 0.0
        if p2 is not None:
            p2_sh = _shield(p2)
            shield_delta = prev.p2_shield - p2_sh
            if shield_delta > 0 and _hitlag(p1) > 0 if p1 else False:
                r_pressure = shield_delta * w.shield_pressure
        else:
            p2_sh = 60.0

        # ---- 4. Hitstun follow-up (combo) ----
        r_combo = 0.0
        if p2 is not None and damage_dealt > 0 and _hitstun(p2) > 0:
            r_combo = w.combo_bonus

        # ---- 5. Edge-guarding ----
        r_edgeguard = 0.0
        if p2 is not None and damage_dealt > 0 and _off_stage(p2):
            r_edgeguard = w.edgeguard_bonus

        # ---- 6. Active spacing (velocity while close) ----
        r_velocity = 0.0
        r_approach = 0.0
        r_proximity = 0.0
        if p1 is not None and p2 is not None:
            dx = _pos_x(p1) - _pos_x(p2)
            dy = _pos_y(p1) - _pos_y(p2)
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= w.velocity_proximity:
                speed = abs(_speed_ground_x(p1)) + abs(_speed_air_x(p1))
                r_velocity = min(speed, 3.0) * w.velocity_bonus_scale

            # ---- 7. Approach: reward for closing distance ----
            prev_dist = getattr(prev, "distance", dist)
            dist_delta = prev_dist - dist
            if dist_delta > 0:
                r_approach = dist_delta * w.approach_scale
            prev.distance = dist

            # ---- 8. Proximity: bonus for staying close ----
            if dist <= w.proximity_threshold:
                r_proximity = w.proximity_bonus

        # ---- 9. Anti-SD: existence, off-stage, blastzone ----
        r_existence = 0.0
        r_off_stage = 0.0
        r_blastzone = 0.0
        if p1 is not None:
            px = _pos_x(p1)
            if not p1_off:
                r_existence = w.existence_reward
            else:
                r_off_stage = w.off_stage_penalty
            if abs(px) > w.blastzone_threshold:
                excess = abs(px) - w.blastzone_threshold
                r_blastzone = w.blastzone_penalty_scale * (excess * excess)

        # ---- Aggregate ----
        reward = (
            r_damage
            + r_stocks
            + r_terminal
            + r_recovery
            + r_pressure
            + r_combo
            + r_edgeguard
            + r_velocity
            + r_approach
            + r_proximity
            + r_existence
            + r_off_stage
            + r_blastzone
        )

        # ---- Update prev state (no ghosting) ----
        prev.p1_percent = p1_pct
        prev.p2_percent = p2_pct
        prev.p1_stocks = p1_stk
        prev.p2_stocks = p2_stk
        prev.p2_shield = p2_sh if p2 is not None else 60.0
        prev.p1_edge_dist = p1_edge
        prev.p1_off_stage = p1_off

        # ---- Episode accumulators ----
        self._ep_reward += reward
        self._ep_damage_dealt += damage_dealt
        self._ep_stocks_won += stocks_taken

        info: Dict[str, float] = {
            "reward_damage": r_damage,
            "reward_stocks": r_stocks,
            "reward_terminal": r_terminal,
            "reward_recovery": r_recovery,
            "reward_pressure": r_pressure,
            "reward_combo": r_combo,
            "reward_edgeguard": r_edgeguard,
            "reward_velocity": r_velocity,
            "reward_approach": r_approach,
            "reward_proximity": r_proximity,
            "reward_existence": r_existence,
            "reward_off_stage": r_off_stage,
            "reward_blastzone": r_blastzone,
            "episode_reward": self._ep_reward,
            "total_damage_dealt": self._ep_damage_dealt,
            "stocks_won": float(self._ep_stocks_won),
        }

        return reward, info
