"""Custom menu navigation to eliminate Ghost Players and Character Drift.

Bypasses buggy menu_helper_simple with:
- Sequential port initialization (P2 first, verify, then P1)
- Hard-coded Fox coordinates and tight tolerance
- Debounced Start logic
- State-sync guard (neutral when STAGE_SELECT but cursor in character area)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

log = logging.getLogger(__name__)

# Fox character select coordinates (libmelee CSS grid)
FOX_TARGET_X = -22.0
FOX_TARGET_Y = 11.5
CURSOR_TOLERANCE = 0.5  # Tightened from 1.5 — no A press unless within range

# P2 HMN/CPU toggle (top of P2 slot) — click to switch Human -> CPU
P2_CPU_TOGGLE_X = -18.0
P2_CPU_TOGGLE_Y = 18.0
P2_VERIFY_TIMEOUT_FRAMES = 60

# Cursor position "above character level" for Start to work (libmelee requirement)
READY_CURSOR_X = 0.0
READY_CURSOR_Y = 20.0  # Above the character grid

# Debounced Start
START_COOLDOWN_FRAMES = 10   # Brief delay before first Start press
START_RETRY_FRAMES = 60      # Retry Start if menu hasn't changed
START_HOLD_FRAMES = 5        # Hold Start for multiple frames (Melee can miss 1-frame presses)

# Character area threshold: cursor.y above this = still in character grid
# Stage select cursor is typically lower; character grid is upper portion
CHARACTER_AREA_Y_THRESHOLD = 0.0  # y > this = character area


def _get_cursor(gamestate: Any, port: int) -> Tuple[float, float]:
    """Get cursor (x, y) for port. Returns (0, 0) if unavailable."""
    if gamestate is None:
        return (0.0, 0.0)
    p = gamestate.players.get(port)
    if p is None:
        return (0.0, 0.0)
    cursor = getattr(p, "cursor", None)
    if cursor is not None:
        if hasattr(cursor, "x") and hasattr(cursor, "y"):
            return (float(cursor.x), float(cursor.y))
        if isinstance(cursor, (tuple, list)) and len(cursor) >= 2:
            return (float(cursor[0]), float(cursor[1]))
    cx = float(getattr(p, "cursor_x", 0.0))
    cy = float(getattr(p, "cursor_y", 0.0))
    return (cx, cy)


def _cursor_in_character_area(cursor_y: float) -> bool:
    """True if cursor is still in the character grid (upper portion of CSS)."""
    return cursor_y > CHARACTER_AREA_Y_THRESHOLD


def move_cursor_to(
    controller: Any,
    gamestate: Any,
    port: int,
    target_x: float,
    target_y: float,
) -> bool:
    """Move cursor toward target. Returns True if within tolerance (no A press)."""
    if gamestate is None or controller is None:
        if controller is not None:
            try:
                controller.release_all()
            except Exception:
                pass
        return False

    cx, cy = _get_cursor(gamestate, port)
    dx = target_x - cx
    dy = target_y - cy
    dist = (dx * dx + dy * dy) ** 0.5

    controller.release_all()
    if dist <= CURSOR_TOLERANCE:
        return True
    if abs(dx) > 0.1 or abs(dy) > 0.1:
        scale = min(1.0, 2.0 / max(0.01, dist))
        stick_x = max(-1.0, min(1.0, dx * scale))
        stick_y = max(-1.0, min(1.0, dy * scale))
        try:
            from melee import Button
            controller.tilt_analog_unit(Button.BUTTON_MAIN, stick_x, stick_y)
        except ImportError:
            pass
    return False


def move_and_click(
    controller: Any,
    gamestate: Any,
    port: int,
    target_x: float,
    target_y: float,
    tolerance: float = CURSOR_TOLERANCE,
    button_a: Any = None,
) -> bool:
    """Move cursor toward target and press A only when within tolerance.

    Bypasses menu_helper_simple's buggy grid calculation.
    Returns True if A was pressed (cursor was in range).
    """
    if gamestate is None or controller is None:
        if controller is not None:
            try:
                controller.release_all()
            except Exception:
                pass
        return False

    cx, cy = _get_cursor(gamestate, port)
    dx = target_x - cx
    dy = target_y - cy
    dist = (dx * dx + dy * dy) ** 0.5

    # Release all first, then apply movement
    controller.release_all()

    if dist <= tolerance:
        # Within range — press A
        btn = button_a
        if btn is None:
            try:
                from melee import Button
                btn = Button.BUTTON_A
            except ImportError:
                return False
        controller.press_button(btn)
        return True

    # Move toward target (normalize to unit-ish stick)
    if abs(dx) > 0.1 or abs(dy) > 0.1:
        scale = min(1.0, 2.0 / max(0.01, dist))
        stick_x = max(-1.0, min(1.0, dx * scale))
        stick_y = max(-1.0, min(1.0, dy * scale))
        try:
            from melee import Button
            controller.tilt_analog_unit(Button.BUTTON_MAIN, stick_x, stick_y)
        except ImportError:
            pass
    return False


def ensure_p2_cpu(
    gamestate: Any,
    cpu_controller: Any,
    menu_helper: Any,
    cpu_level: int,
    cpu_character: Any,
    button_a: Any = None,
) -> bool:
    """Force P2 to CPU if it's Human. Uses change_controller_status and/or move to toggle.

    Returns True when P2.controller_status == CONTROLLER_CPU and P2.cpu_level == cpu_level.
    """
    if gamestate is None or cpu_controller is None:
        return False
    try:
        from melee import Menu
        if getattr(gamestate, "menu_state", None) != Menu.CHARACTER_SELECT:
            return False
    except ImportError:
        return False

    p2 = gamestate.players.get(2)
    if p2 is None:
        return False
    status = getattr(p2, "controller_status", None)
    level = getattr(p2, "cpu_level", 0)
    try:
        from melee.enums import ControllerStatus
        if status == ControllerStatus.CONTROLLER_CPU and level == cpu_level:
            return True
    except ImportError:
        return False

    # P2 is Human — fix it
    try:
        menu_helper.change_controller_status(
            gamestate, 2, ControllerStatus.CONTROLLER_CPU, character=cpu_character
        )
    except Exception:
        pass

    # If P2 cursor is on screen as Human, move to HMN/CPU toggle and click
    move_and_click(
        cpu_controller,
        gamestate,
        port=2,
        target_x=P2_CPU_TOGGLE_X,
        target_y=P2_CPU_TOGGLE_Y,
        tolerance=CURSOR_TOLERANCE,
        button_a=button_a,
    )
    return False


def should_hold_neutral_stage_sync(gamestate: Any, port: int = 1) -> bool:
    """State-sync guard: STAGE_SELECT but cursor still in character area -> hold neutral."""
    if gamestate is None:
        return False
    try:
        from melee import Menu
        if getattr(gamestate, "menu_state", None) != Menu.STAGE_SELECT:
            return False
    except ImportError:
        return False
    _, cy = _get_cursor(gamestate, port)
    return _cursor_in_character_area(cy)
