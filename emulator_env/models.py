# models.py
"""Pydantic models that define the OpenEnv contract for the Smash emulator.
These models are used by both the server (Environment) and any client that talks to it.

Observation fields mirror libmelee.PlayerState and libmelee.GameState for
100% data parity with the Dolphin emulator.
"""

from typing import Annotated, Any, Dict, List

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


# ---------------------------------------------------------------------------
# Nested structures for libmelee parity (ECB, Projectile)
# ---------------------------------------------------------------------------

def _ecb_default() -> Dict[str, Dict[str, float]]:
    """Default ECB: four points (top, bottom, left, right) with (x, y) offsets."""
    zero = {"x": 0.0, "y": 0.0}
    return {"top": zero.copy(), "bottom": zero.copy(), "left": zero.copy(), "right": zero.copy()}


class SmashAction(Action):
    """Controller inputs the RL agent can send each frame.
    All values are clamped/validated by Pydantic.

    Maps directly to a GameCube controller:
    - Main stick (analog): movement, tilts, smashes
    - C-stick (analog): smash attacks, aerials
    - A: normals / smash attacks
    - B: special moves
    - X / Y: jump
    - Z: grab (grounded) / z-air (aerial)
    - L / R: shield, tech, L-cancel (digital press)
    """

    # --- Main analog stick ---
    stick_x: Annotated[float, Field(ge=-1.0, le=1.0)] = Field(
        0.0, description="Main stick horizontal (-1 left, 1 right)"
    )
    stick_y: Annotated[float, Field(ge=-1.0, le=1.0)] = Field(
        0.0, description="Main stick vertical (-1 down, 1 up)"
    )

    # --- C-stick (right analog stick) ---
    c_stick_x: Annotated[float, Field(ge=-1.0, le=1.0)] = Field(
        0.0, description="C-stick horizontal (-1 left, 1 right)"
    )
    c_stick_y: Annotated[float, Field(ge=-1.0, le=1.0)] = Field(
        0.0, description="C-stick vertical (-1 down, 1 up)"
    )

    # --- Face buttons ---
    button_a: bool = Field(False, description="A button (normals / smash attacks)")
    button_b: bool = Field(False, description="B button (special moves)")
    button_x: bool = Field(False, description="X button (jump)")
    button_y: bool = Field(False, description="Y button (jump)")
    button_z: bool = Field(False, description="Z button (grab / z-air)")

    # --- Shoulder buttons (digital) ---
    button_l: bool = Field(False, description="L button (shield / tech / L-cancel)")
    button_r: bool = Field(False, description="R button (shield / tech / L-cancel)")


class SmashObservation(Observation):
    """State information the agent receives after each step.
    Values are taken from libmelee's GameState for player 1 and player 2.
    """

    # --- Player 1 (agent) ---
    player_x: Annotated[float, Field(ge=-2000, le=2000)] = Field(
        0.0, description="Player X coordinate (world units)"
    )
    player_y: Annotated[float, Field(ge=-2000, le=2000)] = Field(
        0.0, description="Player Y coordinate (world units)"
    )
    player_damage: Annotated[int, Field(ge=0, le=999)] = Field(
        0, description="Damage percent of the player"
    )
    player_action_state: str = Field(
        "unknown",
        description="Current action/state string (e.g., STANDING, JUMPING, DEAD_DOWN)",
    )
    player_stocks: int = Field(0, description="Remaining stock count for the player")

    # --- Player 2 (opponent / CPU) ---
    opponent_x: Annotated[float, Field(ge=-2000, le=2000)] = Field(
        0.0, description="Opponent X coordinate (world units)"
    )
    opponent_y: Annotated[float, Field(ge=-2000, le=2000)] = Field(
        0.0, description="Opponent Y coordinate (world units)"
    )
    opponent_damage: Annotated[int, Field(ge=0, le=999)] = Field(
        0, description="Damage percent of the opponent"
    )
    opponent_action_state: str = Field(
        "unknown",
        description="Current action/state string for the opponent",
    )
    opponent_stocks: int = Field(
        0, description="Remaining stock count for the opponent"
    )

    # --- Physics / state (5-speed system, libmelee parity) ---
    # Player 1 — 5-speed breakdown
    player_speed_air_x_self: float = Field(0.0, description="Player horizontal air speed (self)")
    player_speed_ground_x_self: float = Field(0.0, description="Player horizontal ground speed (self)")
    player_speed_y_self: float = Field(0.0, description="Player vertical speed (gravity/jump)")
    player_speed_x_attack: float = Field(0.0, description="Player horizontal speed from knockback")
    player_speed_y_attack: float = Field(0.0, description="Player vertical speed from knockback")
    # Player 1 — computed totals (for reward shaping)
    player_speed_x: float = Field(0.0, description="Player total X velocity")
    player_speed_y: float = Field(0.0, description="Player total Y velocity")
    player_on_ground: bool = Field(True, description="Whether the player is grounded")
    player_facing_right: bool = Field(True, description="Whether the player faces right")
    player_hitstun_left: int = Field(0, description="Player hitstun frames remaining")
    player_hitlag_left: int = Field(0, description="Player hitlag frames (frozen on hit)")
    player_jumpsquat_frames_left: int = Field(0, description="Player jumpsquat frames remaining")
    player_invulnerability_left: int = Field(0, description="Player invincibility frames remaining")
    player_shield_strength: float = Field(60.0, description="Player shield (60 max, 0 = broken)")
    player_ecb: Dict[str, Dict[str, float]] = Field(
        default_factory=_ecb_default,
        description="Player ECB: top/bottom/left/right with {x,y} offsets from root",
    )

    # Player 2 — 5-speed breakdown
    opponent_speed_air_x_self: float = Field(0.0, description="Opponent horizontal air speed (self)")
    opponent_speed_ground_x_self: float = Field(0.0, description="Opponent horizontal ground speed (self)")
    opponent_speed_y_self: float = Field(0.0, description="Opponent vertical speed (gravity/jump)")
    opponent_speed_x_attack: float = Field(0.0, description="Opponent horizontal speed from knockback")
    opponent_speed_y_attack: float = Field(0.0, description="Opponent vertical speed from knockback")
    opponent_speed_x: float = Field(0.0, description="Opponent total X velocity")
    opponent_speed_y: float = Field(0.0, description="Opponent total Y velocity")
    opponent_on_ground: bool = Field(True, description="Whether the opponent is grounded")
    opponent_facing_right: bool = Field(True, description="Whether the opponent faces right")
    opponent_hitstun_left: int = Field(0, description="Opponent hitstun frames remaining")
    opponent_hitlag_left: int = Field(0, description="Opponent hitlag frames (frozen on hit)")
    opponent_jumpsquat_frames_left: int = Field(0, description="Opponent jumpsquat frames remaining")
    opponent_invulnerability_left: int = Field(0, description="Opponent invincibility frames remaining")
    opponent_shield_strength: float = Field(60.0, description="Opponent shield (60 max, 0 = broken)")
    opponent_ecb: Dict[str, Dict[str, float]] = Field(
        default_factory=_ecb_default,
        description="Opponent ECB: top/bottom/left/right with {x,y} offsets from root",
    )

    # Projectiles (libmelee.GameState.projectiles)
    projectiles: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of projectiles: {x, y, speed_x, speed_y, owner_id}",
    )

    frame: int = Field(0, description="Current game frame number")

    # --- General ---
    menu_state: str = Field(
        "unknown",
        description="Current menu/game state (e.g., IN_GAME, CHARACTER_SELECT)",
    )
