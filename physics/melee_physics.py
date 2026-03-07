"""Core Melee physics formulas: knockback, hitstun, gravity, traction.

References:
  - https://www.ssbwiki.com/Knockback
  - https://wiki.supercombo.gg/w/SSBM/Technical_Data
  - https://www.ssbwiki.com/Hitstun_modifier
"""

from __future__ import annotations

import math
from physics.constants import CHAR


# ---------------------------------------------------------------------------
# Knockback
# ---------------------------------------------------------------------------

def compute_knockback(
    target_percent: float,
    damage: float,
    weight: float,
    base_knockback: float,
    knockback_growth: float,
) -> float:
    """Return total knockback magnitude (Melee-style formula).

    Formula (simplified):
        KB = ((((p + d) / 10 + (p + d) * d / 20) * (200 / (w + 100)) * 1.4 + 18)
              * (KBG / 100)) + BKB

    where p = target percent *before* hit, d = move damage, w = weight.
    """
    p = target_percent
    d = damage
    w = weight

    damage_term = (p + d) / 10.0 + ((p + d) * d) / 20.0
    weight_term = 200.0 / (w + 100.0)
    raw = damage_term * weight_term * 1.4 + 18.0
    kb = raw * (knockback_growth / 100.0) + base_knockback
    return max(kb, 0.0)


def knockback_to_velocity(knockback: float, angle_deg: float, facing_sign: float):
    """Convert knockback magnitude + angle to (vx, vy) launch velocity.

    ``angle_deg`` is relative to horizontal-forward (0 = pure horizontal,
    90 = pure vertical).  ``facing_sign`` flips the horizontal component so
    the victim is launched *away* from the attacker.
    """
    angle_rad = math.radians(angle_deg)
    vx = knockback * math.cos(angle_rad) * 0.05 * facing_sign
    vy = knockback * math.sin(angle_rad) * 0.05
    return vx, vy


# ---------------------------------------------------------------------------
# Hitstun
# ---------------------------------------------------------------------------

def compute_hitstun(knockback: float) -> int:
    """Melee hitstun = floor(0.4 * knockback)."""
    return max(int(knockback * 0.4), 0)


# ---------------------------------------------------------------------------
# Gravity & movement helpers
# ---------------------------------------------------------------------------

def apply_gravity(speed_y: float, gravity: float = CHAR["gravity"],
                  terminal: float = CHAR["terminal_velocity"]) -> float:
    """Apply one frame of gravity, clamped to terminal velocity."""
    speed_y -= gravity
    if speed_y < -terminal:
        speed_y = -terminal
    return speed_y


def apply_traction(speed_x: float, traction: float = CHAR["traction"]) -> float:
    """Decelerate ground speed toward zero by traction amount per frame."""
    if abs(speed_x) <= traction:
        return 0.0
    return speed_x - math.copysign(traction, speed_x)


def apply_air_friction(speed_x: float,
                       friction: float = CHAR["air_friction"]) -> float:
    """Decelerate aerial self-speed toward zero."""
    if abs(speed_x) <= friction:
        return 0.0
    return speed_x - math.copysign(friction, speed_x)


def decay_attack_velocity(vx: float, vy: float, decay: float = 0.051):
    """Melee decays attack-induced velocity each frame (≈0.051 per frame)."""
    if abs(vx) <= decay:
        vx = 0.0
    else:
        vx -= math.copysign(decay, vx)

    if abs(vy) <= decay:
        vy = 0.0
    else:
        vy -= math.copysign(decay, vy)

    return vx, vy
