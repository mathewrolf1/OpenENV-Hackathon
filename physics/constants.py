"""Numeric constants for the Melee-like physics simulator.

Values are based on Jigglypuff in SSBM.  Where exact Melee data is
available the source is noted; otherwise values are tuned to *feel*
Melee-like while keeping the sim simple.
"""

from enum import IntEnum


# ---------------------------------------------------------------------------
# Action / state enum
# ---------------------------------------------------------------------------

class Action(IntEnum):
    IDLE = 0
    WALK = 1
    RUN = 2
    JUMPSQUAT = 3
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

    REST_SLEEP = 25  # Puff-specific: asleep after Rest (hit or miss)

    DEAD = 30
    RESPAWN_INVULN = 31

    NUM_ACTIONS = 32  # sentinel for observation encoding


# ---------------------------------------------------------------------------
# Character constants  (approx Jigglypuff values)
# ---------------------------------------------------------------------------

CHAR = {
    "weight": 60,              # Puff is the lightest in Melee
    "gravity": 0.064,          # lowest gravity in the game — very floaty
    "terminal_velocity": 1.3,  # very slow fall speed
    "walk_speed": 0.7,
    "run_speed": 1.1,          # slow on the ground
    "air_speed": 1.35,         # best air speed in Melee
    "air_accel": 0.05,
    "traction": 0.06,
    "air_friction": 0.01,
    "jump_velocity": 2.4,
    "short_hop_velocity": 1.3,
    "double_jump_velocity": 2.0,
    "max_jumps": 6,            # 1 ground + 5 midair jumps
    "jumpsquat_frames": 6,     # Puff = 6
    "landing_frames": 4,
    "respawn_invuln_frames": 120,
    "respawn_x": 0.0,
    "respawn_y": 40.0,
}


# ---------------------------------------------------------------------------
# Stage constants  (roughly Final Destination)
# ---------------------------------------------------------------------------

STAGE = {
    "floor_y": 0.0,
    "left_edge": -68.4,
    "right_edge": 68.4,
    "left_blastzone": -224.0,
    "right_blastzone": 224.0,
    "top_blastzone": 180.0,
    "bottom_blastzone": -109.0,
}


# ---------------------------------------------------------------------------
# Move data – dict of {name: params}
# Each move has: damage, BKB, KBG, angle (degrees), hitbox ranges,
# startup / active / endlag frame counts, is_grab flag.
# ---------------------------------------------------------------------------

MOVES = {
    "jab": {
        "damage": 4.0,
        "base_knockback": 0.0,
        "knockback_growth": 100.0,
        "angle": 80.0,
        "hitbox_x_range": 8.0,
        "hitbox_y_range": 6.0,
        "startup_frames": 2,
        "active_frames": 2,
        "endlag_frames": 8,
        "is_grab": False,
    },
    "grab": {
        "damage": 0.0,
        "base_knockback": 0.0,
        "knockback_growth": 0.0,
        "angle": 45.0,
        "hitbox_x_range": 10.0,
        "hitbox_y_range": 8.0,
        "startup_frames": 6,
        "active_frames": 2,
        "endlag_frames": 20,
        "is_grab": True,
    },
    "smash": {
        "damage": 16.0,
        "base_knockback": 30.0,
        "knockback_growth": 103.0,
        "angle": 80.0,
        "hitbox_x_range": 12.0,
        "hitbox_y_range": 14.0,
        "startup_frames": 7,
        "active_frames": 3,
        "endlag_frames": 25,
        "is_grab": False,
    },
    "rest": {
        "damage": 20.0,
        "base_knockback": 40.0,
        "knockback_growth": 180.0,   # extreme growth — kills early
        "angle": 88.0,               # nearly straight up
        "hitbox_x_range": 4.0,       # tiny hitbox: must overlap the opponent
        "hitbox_y_range": 4.0,
        "startup_frames": 1,         # frame-1 active — instant
        "active_frames": 1,          # only 1 frame to land it
        "endlag_frames": 240,        # 4 seconds asleep if you miss (the risk)
        "is_grab": False,
    },
}


# ---------------------------------------------------------------------------
# Grab / throw constants
# ---------------------------------------------------------------------------

GRAB_HOLD_FRAMES = 30       # base frames held in grab
GRAB_PUMMEL_DAMAGE = 3.0
THROW_DAMAGE = 7.0
THROW_BKB = 60.0
THROW_KBG = 70.0
THROW_ANGLE = 45.0          # degrees

# ---------------------------------------------------------------------------
# Time limit (in frames, 60 fps)
# ---------------------------------------------------------------------------

MAX_FRAMES = 8 * 60 * 60    # 8 minutes
