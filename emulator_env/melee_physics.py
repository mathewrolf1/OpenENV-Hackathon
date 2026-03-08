"""Melee physics helpers for emulator state.

Computes total velocity from the 5-speed system. Values are populated from
libmelee; this module only provides the aggregation logic for state-tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import PlayerState


def total_velocity_x(p: "PlayerState") -> float:
    """Total horizontal velocity = ground/air self + attack.
    On ground: speed_ground_x_self + speed_x_attack
    In air: speed_air_x_self + speed_x_attack
    """
    if p.on_ground:
        return p.speed_ground_x_self + p.speed_x_attack
    return p.speed_air_x_self + p.speed_x_attack


def total_velocity_y(p: "PlayerState") -> float:
    """Total vertical velocity = self (gravity/jump) + attack (knockback)."""
    return p.speed_y_self + p.speed_y_attack


def total_velocity(p: "PlayerState") -> tuple[float, float]:
    """Return (total_vx, total_vy) for movement display / reward shaping."""
    return total_velocity_x(p), total_velocity_y(p)
