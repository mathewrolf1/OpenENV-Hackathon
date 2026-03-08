"""Libmelee-parity state structures for the Dolphin emulator.

These dataclasses mirror libmelee.PlayerState and libmelee.GameState memory
structures. Values are populated from raw bytes via libmelee — no computation
is performed here. Structure is ready to receive emulator data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Action enum (libmelee parity — JUMPSQUAT, HITLAG states)
# ---------------------------------------------------------------------------

class Action(IntEnum):
    """Action/state enum aligned with libmelee.Animation."""

    IDLE = 0
    WALK = 1
    RUN = 2
    JUMPSQUAT = 3   # Pre-jump crouch (KNEE_BEND in libmelee)
    AIRBORNE = 4
    LANDING = 5

    ATTACK_STARTUP = 10
    ATTACK_ACTIVE = 11
    ATTACK_ENDLAG = 12

    GRAB_STARTUP = 13
    GRAB_ACTIVE = 14
    GRAB_ENDLAG = 15
    GRABBED = 16
    THROW = 17

    HITSTUN = 20
    TUMBLE = 21
    HITLAG = 22     # Frozen in place upon hit connecting (both attacker & victim)

    REST_SLEEP = 25
    DEAD = 30
    RESPAWN_INVULN = 31

    NUM_ACTIONS = 32


# ---------------------------------------------------------------------------
# ECB (Environment Collision Box) — 4-point diamond from Dolphin
# ---------------------------------------------------------------------------

@dataclass
class ECBPoint:
    """Single point of the ECB diamond. (x, y) offset from character root."""

    x: float = 0.0
    y: float = 0.0


@dataclass
class ECB:
    """Environment Collision Box: four-point diamond used by Dolphin.
    Each point is (x, y) relative to the character's root position.
    """

    top: ECBPoint = field(default_factory=lambda: ECBPoint(0.0, 0.0))
    bottom: ECBPoint = field(default_factory=lambda: ECBPoint(0.0, 0.0))
    left: ECBPoint = field(default_factory=lambda: ECBPoint(0.0, 0.0))
    right: ECBPoint = field(default_factory=lambda: ECBPoint(0.0, 0.0))


# ---------------------------------------------------------------------------
# PlayerState — libmelee.PlayerState parity (5-speed system)
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    """Per-player state snapshot. Mirrors libmelee.PlayerState.
    Velocity uses the 5-speed system; total movement = sum of components.
    """

    # Position
    x: float = 0.0
    y: float = 0.0

    # --- 5-Speed System (Melee tracks these separately) ---
    speed_air_x_self: float = 0.0   # Horizontal momentum in air (player-controlled)
    speed_ground_x_self: float = 0.0  # Horizontal momentum on ground (player-controlled)
    speed_y_self: float = 0.0       # Vertical momentum (gravity, jumping)
    speed_x_attack: float = 0.0     # Horizontal momentum from being hit (knockback)
    speed_y_attack: float = 0.0     # Vertical momentum from being hit (knockback)

    # Damage / stocks
    percent: float = 0.0
    stock: int = 4

    # Ground / facing
    on_ground: bool = True
    facing_right: bool = True

    # Action
    action: Action = Action.IDLE
    action_frame: int = 0

    # --- State timers (frame counts) ---
    hitlag_left: int = 0            # Frames frozen when hit connects
    jumpsquat_frames_left: int = 0  # Frames in pre-jump crouch
    invulnerability_left: int = 0   # Frames of star/ghost invincibility
    hitstun_frames_left: int = 0

    # Shield (Melee: starts at 60.0, breaks at 0.0)
    shield_strength: float = 60.0

    # ECB (Environment Collision Box) — 4-point diamond
    ecb: ECB = field(default_factory=ECB)

    # Grab / hit tracking
    jumps_left: int = 2


# ---------------------------------------------------------------------------
# Projectile — libmelee.Projectile parity
# ---------------------------------------------------------------------------

@dataclass
class Projectile:
    """Projectile state. Mirrors libmelee.Projectile."""

    x: float = 0.0
    y: float = 0.0
    speed_x: float = 0.0
    speed_y: float = 0.0
    owner_id: int = -1   # Controller port; -1 = no owner
    frame: int = 0       # How long the item has been out
    type_id: Optional[int] = None
    subtype: Optional[int] = None


# ---------------------------------------------------------------------------
# GameState — libmelee.GameState parity
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Full game snapshot. Mirrors libmelee.GameState."""

    frame: int = 0
    players: List[PlayerState] = field(default_factory=list)
    projectiles: List[Projectile] = field(default_factory=list)
    menu_state: str = "unknown"
    stage_id: Optional[int] = None
