# models.py
"""Pydantic models that define the OpenEnv contract for the Smash emulator.
These models are used by both the server (Environment) and any client that talks to it.
"""

from typing import Annotated

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


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

    # --- General ---
    menu_state: str = Field(
        "unknown",
        description="Current menu/game state (e.g., IN_GAME, CHARACTER_SELECT)",
    )
