"""Dataclasses representing the Melee-like game state.

Field names are chosen to align with libmelee's PlayerState so the same
observation builder can be reused when switching to a Dolphin-backed env.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional

from physics.constants import Action, CHAR, STAGE as STAGE_CONST


@dataclass
class MoveData:
    """Static data describing a single attack / grab."""

    damage: float
    base_knockback: float  # BKB
    knockback_growth: float  # KBG (as used in formula: KBG/100)
    angle: float  # launch angle in degrees (0 = forward, 90 = up)
    hitbox_x_range: float  # how far in front the hitbox extends
    hitbox_y_range: float  # vertical extent (half-height)
    startup_frames: int  # frames before hitbox becomes active
    active_frames: int  # how many frames hitbox is out
    endlag_frames: int  # recovery frames after active window
    is_grab: bool = False

    @property
    def total_frames(self) -> int:
        return self.startup_frames + self.active_frames + self.endlag_frames


@dataclass
class CharacterState:
    """Per-player state snapshot, aligned with libmelee PlayerState fields."""

    x: float = 0.0
    y: float = 0.0

    speed_x_self: float = 0.0
    speed_y_self: float = 0.0
    speed_x_attack: float = 0.0
    speed_y_attack: float = 0.0

    percent: float = 0.0
    stock: int = 4
    on_ground: bool = True
    facing_right: bool = True

    action: Action = Action.IDLE
    action_frame: int = 0

    hitstun_frames_left: int = 0
    invulnerable_frames_left: int = 0
    jumps_left: int = CHAR["max_jumps"]

    grabbed_by: Optional[int] = None  # port index of grabber, or None
    grab_timer: int = 0
    attack_connected: bool = False  # prevents multi-hit on same attack

    @property
    def facing_sign(self) -> float:
        return 1.0 if self.facing_right else -1.0

    def is_actionable(self) -> bool:
        return (
            self.hitstun_frames_left <= 0
            and self.action
            not in (
                Action.ATTACK_STARTUP,
                Action.ATTACK_ACTIVE,
                Action.ATTACK_ENDLAG,
                Action.GRAB_STARTUP,
                Action.GRAB_ACTIVE,
                Action.GRAB_ENDLAG,
                Action.GRABBED,
                Action.THROW,
                Action.REST_SLEEP,
                Action.DEAD,
                Action.RESPAWN_INVULN,
            )
        )


@dataclass
class Stage:
    """Axis-aligned stage geometry (Final-Destination-like)."""

    floor_y: float = STAGE_CONST["floor_y"]
    left_edge: float = STAGE_CONST["left_edge"]
    right_edge: float = STAGE_CONST["right_edge"]
    left_blastzone: float = STAGE_CONST["left_blastzone"]
    right_blastzone: float = STAGE_CONST["right_blastzone"]
    top_blastzone: float = STAGE_CONST["top_blastzone"]
    bottom_blastzone: float = STAGE_CONST["bottom_blastzone"]


@dataclass
class GameState:
    """Full snapshot of the game at one frame."""

    frame: int = 0
    players: List[CharacterState] = field(default_factory=lambda: [
        CharacterState(x=-30.0, facing_right=True),
        CharacterState(x=30.0, facing_right=False),
    ])
    stage: Stage = field(default_factory=Stage)
    done: bool = False
    winner: Optional[int] = None  # player index or None

    def copy(self) -> GameState:
        return copy.deepcopy(self)
