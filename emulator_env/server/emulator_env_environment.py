# server/emulator_env_environment.py
"""OpenEnv server implementation that wraps libmelee (Slippi Dolphin).
It inherits from `openenv.core.env_server.interfaces.Environment` and defines:
* `reset()` – navigates menus, starts a match, returns initial observation.
* `step(action)` – translates a `SmashAction` into libmelee controller commands,
  advances one frame, and returns a `SmashObservation`.
* `state` property – returns the current session state.
* `close()` – shuts down the Dolphin process.

Port 1: RL agent, controlled by client (dolphin_train.py).
Port 2: Opponent — CPU AI or frozen policy on the server.

Environment variables:
    DOLPHIN_PATH         – (required) directory containing the Slippi Dolphin executable.
    ISO_PATH             – (optional) path to the Melee ISO file.
    DOLPHIN_HOME         – (optional) path to the Dolphin user/home directory.
    P1_CHARACTER         – (optional) character for port 1 / agent. Default: JIGGLYPUFF.
    P2_CHARACTER         – (optional) character for port 2 / opponent. Default: FOX.
    CPU_LEVEL            – 0 = P2 driven by model checkpoint; >0 = Dolphin CPU AI at that level.
    P2_CHECKPOINT_PATH   – (optional) path to opponent model for P2. Default: checkpoints/mango_final.pt
    TRAINING_MODE        – "NORMAL" (default) or "RECOVERY". If RECOVERY, each reset runs a short
                           scripted phase to spawn agent off-stage and opponent center for recovery training.
"""

import math
import os
import logging
import configparser
from typing import Any, Optional
from uuid import uuid4

# ---------------------------------------------------------------------------
# Python 3.14 compatibility: configparser now raises DuplicateOptionError by
# default when an INI file contains duplicate keys.  Slippi's Dolphin.ini
# ships with duplicate "slippireplaymonthfolders" entries, which crashes
# libmelee on startup.  We monkey-patch configparser so that new
# ConfigParser instances default to strict=False (the pre-3.14 behaviour).
# ---------------------------------------------------------------------------
_OrigConfigParser = configparser.ConfigParser


class _LenientConfigParser(_OrigConfigParser):
    """ConfigParser that tolerates duplicate keys (strict=False)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("strict", False)
        super().__init__(*args, **kwargs)


configparser.ConfigParser = _LenientConfigParser  # type: ignore[misc]

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..melee_constants import (
    CENTER_STAGE_REWARD,
    CENTER_STAGE_X_THRESHOLD,
    DISTANCE_REWARD_SCALE,
    FD_HALF_HEIGHT_Y,
    FD_HALF_WIDTH_X,
    FD_MID_X,
    FD_MID_Y,
    FOX_MAX_HORIZONTAL_SPEED,
    FOX_MAX_JUMP_VELOCITY,
    FOX_TERMINAL_VELOCITY,
    NOISE_PENALTY_HITLAG_JUMPSQUAT,
    PUFF_MAX_HORIZONTAL_SPEED,
    PUFF_MAX_JUMP_VELOCITY,
    PUFF_TERMINAL_VELOCITY,
    STAGE_LEFT_EDGE,
    STAGE_RIGHT_EDGE,
)
from ..models import SmashAction, SmashObservation
from ..policy_runner import (
    action_to_smash,
    get_action,
    load_model,
    obs_to_vector,
)

import melee
from melee import Console, Controller, Button

log = logging.getLogger(__name__)

# Maximum number of frames to spend in menu navigation before giving up.
_MAX_MENU_FRAMES = 3600  # ~60 seconds at 60 fps

# RECOVERY mode: frames to run scripted "agent off-stage, opponent center" before returning control.
_RECOVERY_SETUP_FRAMES = 120  # ~2 seconds


class EmulatorEnvServer(Environment[SmashAction, SmashObservation, State]):
    """Environment that runs a Slippi Dolphin instance via libmelee.
    The server is launched by OpenEnv (FastAPI + WebSocket) and will be
    reachable at http://localhost:8000 (or whatever port you choose).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._connected = False

        # Delta-tracking for reward computation (mirrors MeleeSimEnv)
        self._prev_player_damage: float = 0.0
        self._prev_opponent_damage: float = 0.0
        self._prev_opponent_stocks: int = 4

        # P2 opponent policy
        self._p2_model = None
        self._last_gamestate = None
        self._training_mode = (os.environ.get("TRAINING_MODE", "NORMAL") or "NORMAL").strip().upper()
        self._first_match_started = False  # True once the first match begins

        try:
            self._init_emulator()
        except Exception:
            log.exception("EmulatorEnvServer.__init__ FAILED — see traceback above")
            raise

    def _init_emulator(self) -> None:
        """Perform the actual emulator setup (separated for clean error logging)."""

        # ---- configuration from environment variables ----
        dolphin_path = os.environ.get("DOLPHIN_PATH")
        if not dolphin_path:
            raise RuntimeError(
                "DOLPHIN_PATH environment variable is not set. "
                "Set it to the directory containing your Slippi Dolphin executable."
            )

        self._iso_path = os.environ.get("ISO_PATH")
        dolphin_home = os.environ.get("DOLPHIN_HOME")

        log.info("DOLPHIN_PATH = %s", dolphin_path)
        log.info("ISO_PATH     = %s", self._iso_path)
        log.info("DOLPHIN_HOME = %s", dolphin_home)
        log.info("DISPLAY      = %s", os.environ.get("DISPLAY"))

        # Quick sanity check: does the dolphin directory / exe exist?
        if not os.path.isdir(dolphin_path):
            raise RuntimeError(f"DOLPHIN_PATH directory does not exist: {dolphin_path}")

        # On Linux libmelee looks for Slippi_Online-x86_64.AppImage in the dir
        import platform

        if platform.system() == "Linux":
            expected_exe = os.path.join(dolphin_path, "Slippi_Online-x86_64.AppImage")
            if not os.path.exists(expected_exe):
                contents = os.listdir(dolphin_path)
                raise RuntimeError(
                    f"Expected '{expected_exe}' but it does not exist. "
                    f"Directory contents: {contents}"
                )
            log.info("Found Dolphin exe/symlink: %s", expected_exe)

        # Build Console with optional home directory.
        console_kwargs: dict[str, Any] = dict(
            path=dolphin_path,
            fullscreen=False,
        )
        if dolphin_home:
            console_kwargs["dolphin_home_path"] = dolphin_home
            # When pointing to an existing home dir, don't use a temp copy.
            console_kwargs["tmp_home_directory"] = False

        log.info("Creating melee.Console with: %s", console_kwargs)
        self.console = Console(**console_kwargs)

        # Player 1 – the RL agent.
        self.controller = Controller(self.console, port=1)
        # Player 2 – always created (needed to select character + CPU level
        # in the CSS). During gameplay we release_all() so CPU AI takes over.
        self.cpu_controller = Controller(self.console, port=2)

        # Menu navigation helpers.
        self.menu_helper = melee.MenuHelper()
        self.cpu_menu_helper = melee.MenuHelper()

        # Characters are configurable via P1_CHARACTER / P2_CHARACTER env vars.
        # Defaults: P1=JIGGLYPUFF (agent), P2=FOX (opponent).
        p1_char_name = os.environ.get("P1_CHARACTER", "JIGGLYPUFF").strip().upper()
        p2_char_name = os.environ.get("P2_CHARACTER", "FOX").strip().upper()
        self._character = getattr(melee.Character, p1_char_name, melee.Character.JIGGLYPUFF)
        self._cpu_character = getattr(melee.Character, p2_char_name, melee.Character.FOX)
        self._stage = melee.Stage.FINAL_DESTINATION
        log.info("P1 (agent): %s | P2 (opponent): %s", p1_char_name, p2_char_name)

        # CPU_LEVEL: if > 0, use Dolphin's built-in CPU AI at that level (e.g. 9).
        # If 0 (default), P2 is driven by a model checkpoint.
        cpu_level_env = os.environ.get("CPU_LEVEL", "0").strip()
        try:
            self._cpu_level = int(cpu_level_env)
        except ValueError:
            self._cpu_level = 0

        if self._cpu_level > 0:
            log.info("P2 (%s) CPU level %d", p2_char_name, self._cpu_level)
        else:
            p2_ckpt_path = os.environ.get(
                "P2_CHECKPOINT_PATH",
                os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "mango_final.pt"),
            )
            if os.path.isfile(p2_ckpt_path):
                self._p2_model = load_model(p2_ckpt_path, device="cpu")
                log.info("P2 (%s) controlled by model from %s", p2_char_name, p2_ckpt_path)
            else:
                log.warning(
                    "P2 model not found at %s – P2 will hold neutral. "
                    "Set P2_CHECKPOINT_PATH or place the checkpoint in checkpoints/.",
                    p2_ckpt_path,
                )

        # Launch the emulator process.
        log.info("Launching Dolphin from %s ...", dolphin_path)
        self.console.run(iso_path=self._iso_path)
        log.info("console.run() returned — Dolphin process started.")

        # Wait for the Slippi connection to be established.
        log.info("Waiting for Dolphin to connect ...")
        if not self.console.connect():
            raise RuntimeError(
                "Failed to connect to Dolphin. Make sure Dolphin started "
                "correctly and Slippi is enabled."
            )
        log.info("Dolphin connected.")

        log.info("Connecting controllers ...")
        self.controller.connect()
        self.cpu_controller.connect()
        log.info("Controllers connected (port 1 + port 2).")

        self._connected = True

    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SmashObservation:
        """Reset the environment: navigate menus until a match begins.

        This loops through the Melee menu system (character select, stage
        select) using `melee.MenuHelper`, then returns the first in-game
        observation.  If we're already in a match (e.g. previous episode
        ended), the postgame screen is skipped first.

        kwargs:
            training_mode: Override for this reset only ("NORMAL" | "RECOVERY").
                           Use when implementing curriculum (e.g. switch to NORMAL when recovery success rate > 80%).
        """
        training_override = (kwargs.get("training_mode") or "").strip().upper()
        use_recovery = (training_override == "RECOVERY") or (
            training_override != "NORMAL" and self._training_mode == "RECOVERY"
        )
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Reset delta-tracking for the new episode.
        self._prev_player_damage = 0.0
        self._prev_opponent_damage = 0.0
        self._prev_opponent_stocks = 4

        log.info("reset(): navigating menus to start a new match ...")

        # Release all controller inputs immediately so stale presses from
        # the previous episode don't interfere with menu navigation.
        self.controller.release_all()
        self.cpu_controller.release_all()

        # Fresh helpers so stage_selected / name_tag_index don't carry over.
        self.menu_helper = melee.MenuHelper()
        self.cpu_menu_helper = melee.MenuHelper()

        # ---- Phase 1: Dismiss postgame screen if present ----
        # POSTGAME_SCORES requires explicit A/START presses to advance.
        # Some Melee builds need several frames of button presses to advance.
        _POSTGAME_DISMISS_FRAMES = 180  # 3 seconds max to clear postgame
        last_menu_state = None

        for dismiss_idx in range(_POSTGAME_DISMISS_FRAMES):
            gamestate = self.console.step()
            if gamestate is None:
                continue

            menu_state = gamestate.menu_state

            # Log transitions for debugging
            if menu_state != last_menu_state:
                log.info("reset() phase 1: menu_state=%s (frame %d)",
                         menu_state, dismiss_idx)
                last_menu_state = menu_state

            if menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                # Already in a match (shouldn't happen, but handle it)
                log.info("reset(): already in-game at dismiss frame %d.", dismiss_idx)
                self._first_match_started = True
                self._last_gamestate = gamestate
                if use_recovery:
                    gamestate = self._run_recovery_setup(gamestate, dismiss_idx)
                    self._last_gamestate = gamestate
                return self._make_observation(gamestate, reward=0.0, done=False)

            if menu_state == melee.Menu.POSTGAME_SCORES:
                # Spam A and START on both controllers to dismiss the screen.
                # Alternate between A and START every few frames for robustness.
                if dismiss_idx % 4 < 2:
                    self.controller.press_button(Button.BUTTON_A)
                    self.cpu_controller.press_button(Button.BUTTON_A)
                else:
                    self.controller.press_button(Button.BUTTON_START)
                    self.cpu_controller.press_button(Button.BUTTON_START)
                continue

            # Reached a non-postgame menu state — postgame is dismissed.
            # Release all buttons and break to phase 2 (menu navigation).
            self.controller.release_all()
            self.cpu_controller.release_all()
            break
        else:
            # If we exhausted the loop without breaking, release and continue.
            self.controller.release_all()
            self.cpu_controller.release_all()
            log.warning("reset(): postgame dismiss phase timed out after %d frames.",
                        _POSTGAME_DISMISS_FRAMES)

        # Small settle period: advance a few frames with no inputs to let
        # Melee transition cleanly into the CSS (prevents ghost inputs).
        for _ in range(10):
            self.controller.release_all()
            self.cpu_controller.release_all()
            gamestate = self.console.step()

        # ---- Phase 2: Menu navigation to next match ----
        last_menu_state = None

        for frame_idx in range(_MAX_MENU_FRAMES):
            gamestate = self.console.step()
            if gamestate is None:
                continue

            menu_state = gamestate.menu_state

            # Log transitions
            if menu_state != last_menu_state:
                log.info("reset() phase 2: menu_state=%s (frame %d)",
                         menu_state, frame_idx)
                last_menu_state = menu_state

            if menu_state in (
                melee.Menu.IN_GAME,
                melee.Menu.SUDDEN_DEATH,
            ):
                log.info("reset(): match started after %d menu frames.", frame_idx)
                self._first_match_started = True
                self._last_gamestate = gamestate
                if use_recovery:
                    gamestate = self._run_recovery_setup(gamestate, frame_idx)
                    self._last_gamestate = gamestate
                return self._make_observation(gamestate, reward=0.0, done=False)

            # If we somehow land back on POSTGAME_SCORES, dismiss again.
            if menu_state == melee.Menu.POSTGAME_SCORES:
                self.controller.press_button(Button.BUTTON_A)
                self.cpu_controller.press_button(Button.BUTTON_A)
                continue

            # ---- P2 ----
            if not self._first_match_started:
                self.cpu_menu_helper.menu_helper_simple(
                    gamestate=gamestate,
                    controller=self.cpu_controller,
                    character_selected=self._cpu_character,
                    stage_selected=self._stage,
                    connect_code="",
                    cpu_level=self._cpu_level,
                    costume=0,
                    autostart=False,
                    swag=False,
                    frozen_stadium=False,
                )
            else:
                # Subsequent matches: character persists, hands off.
                self.cpu_controller.release_all()

            # ---- P1 ----
            # On the first match's CSS, hold autostart until P2 finishes
            # selecting. Everywhere else, autostart=True so libmelee
            # instantly selects character → FD → go.
            p1_autostart = True
            if (
                not self._first_match_started
                and gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT
            ):
                p2 = gamestate.players.get(2)
                if p2 is None or not getattr(p2, "coin_down", False):
                    p1_autostart = False
                elif self._cpu_level > 0 and (
                    p2.controller_status != melee.enums.ControllerStatus.CONTROLLER_CPU
                    or p2.cpu_level != self._cpu_level
                ):
                    p1_autostart = False

            self.menu_helper.menu_helper_simple(
                gamestate=gamestate,
                controller=self.controller,
                character_selected=self._character,
                stage_selected=self._stage,
                connect_code="",
                cpu_level=0,
                costume=0,
                autostart=p1_autostart,
                swag=False,
                frozen_stadium=False,
            )

        # If we never reached IN_GAME, return a placeholder.
        log.warning("reset(): timed out waiting for match to start!")
        gamestate = self.console.step()
        self._last_gamestate = gamestate
        return self._make_observation(gamestate, reward=0.0, done=False)

    def _run_recovery_setup(self, gamestate, menu_frame_start: int):
        """Run scripted phase: P1 moves off-stage, P2 holds center. Returns gamestate after setup."""
        log.info("RECOVERY mode: running %d setup frames (P1 off-stage, P2 center).", _RECOVERY_SETUP_FRAMES)
        for i in range(_RECOVERY_SETUP_FRAMES):
            p1 = gamestate.players.get(1) if gamestate else None
            p2 = gamestate.players.get(2) if gamestate else None
            # P1: walk toward nearest ledge and jump off
            if p1 and hasattr(p1.position, "x"):
                px = float(p1.position.x)
                # Move toward left edge if we're right of center, else toward right edge
                if px > 10:
                    self.controller.tilt_analog_unit(Button.BUTTON_MAIN, 1.0, 0.0)  # right toward edge
                elif px < -10:
                    self.controller.tilt_analog_unit(Button.BUTTON_MAIN, -1.0, 0.0)  # left toward edge
                else:
                    self.controller.tilt_analog_unit(Button.BUTTON_MAIN, -1.0, 0.0)  # default left
                # Jump after a short run so we go off-stage (e.g. frame 20–40 and 60–80)
                if 25 <= i <= 28 or 55 <= i <= 58:
                    self.controller.press_button(Button.BUTTON_X)
                else:
                    self.controller.release_button(Button.BUTTON_X)
                self.controller.tilt_analog_unit(Button.BUTTON_C, 0.0, 0.0)
                for btn in (Button.BUTTON_A, Button.BUTTON_B, Button.BUTTON_Y, Button.BUTTON_Z, Button.BUTTON_L, Button.BUTTON_R):
                    self.controller.release_button(btn)
            # P2: release all during recovery setup
            self.cpu_controller.release_all()
            gamestate = self.console.step()
            self._last_gamestate = gamestate
            if gamestate is not None and gamestate.menu_state == melee.Menu.POSTGAME_SCORES:
                break
        log.info("RECOVERY setup complete.")
        return gamestate

    # ------------------------------------------------------------------
    def step(
        self,
        action: SmashAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SmashObservation:
        """Apply a controller action and advance one frame.
        The reward is a simple heuristic: +1 per frame survived minus a
        penalty proportional to damage.
        """
        self._state.step_count += 1

        # Only apply model actions while a match is live.  If we're already
        # on the postgame/menu screen, release everything and let libmelee's
        # reset() handle navigation.
        currently_in_game = (
            self._last_gamestate is not None
            and self._last_gamestate.menu_state in (
                melee.Menu.IN_GAME,
                melee.Menu.SUDDEN_DEATH,
            )
        )

        if currently_in_game:
            # Translate the Pydantic action into libmelee controller calls.
            self.controller.tilt_analog_unit(
                Button.BUTTON_MAIN, action.stick_x, action.stick_y
            )
            self.controller.tilt_analog_unit(
                Button.BUTTON_C, action.c_stick_x, action.c_stick_y
            )
            for pressed, button in (
                (action.button_a, Button.BUTTON_A),
                (action.button_b, Button.BUTTON_B),
                (action.button_x, Button.BUTTON_X),
                (action.button_y, Button.BUTTON_Y),
                (action.button_z, Button.BUTTON_Z),
                (action.button_l, Button.BUTTON_L),
                (action.button_r, Button.BUTTON_R),
            ):
                if pressed:
                    self.controller.press_button(button)
                else:
                    self.controller.release_button(button)
        else:
            self.controller.release_all()

        # P2: if CPU mode, release all inputs so Dolphin's internal CPU AI drives.
        # If model mode, only act while in-game — release otherwise.
        if self._cpu_level > 0 or not currently_in_game:
            self.cpu_controller.release_all()
        elif currently_in_game:
            if self._p2_model is not None:
                obs_p2 = self._make_observation(self._last_gamestate, done=False)
                obs_vec = obs_to_vector(obs_p2, player_idx=1)  # P2 perspective
                action_indices = get_action(self._p2_model, obs_vec, deterministic=True)
                smash_p2 = action_to_smash(action_indices)
            else:
                smash_p2 = SmashAction()  # neutral when model not loaded
            self.cpu_controller.tilt_analog_unit(Button.BUTTON_MAIN, smash_p2.stick_x, smash_p2.stick_y)
            self.cpu_controller.tilt_analog_unit(Button.BUTTON_C, smash_p2.c_stick_x, smash_p2.c_stick_y)
            for pressed, button in (
                (smash_p2.button_a, Button.BUTTON_A),
                (smash_p2.button_b, Button.BUTTON_B),
                (smash_p2.button_x, Button.BUTTON_X),
                (smash_p2.button_y, Button.BUTTON_Y),
                (smash_p2.button_z, Button.BUTTON_Z),
                (smash_p2.button_l, Button.BUTTON_L),
                (smash_p2.button_r, Button.BUTTON_R),
            ):
                if pressed:
                    self.cpu_controller.press_button(button)
                else:
                    self.cpu_controller.release_button(button)
        # If self._cpu_level > 0, Dolphin's built-in CPU AI controls P2 automatically.

        # Advance the emulator one frame and fetch the new gamestate.
        prev_gamestate = self._last_gamestate  # for action-clipping reward (pre-step state)
        gamestate = self.console.step()
        self._last_gamestate = gamestate

        # Determine whether the match ended.
        done = (
            gamestate is not None and gamestate.menu_state == melee.Menu.POSTGAME_SCORES
        )

        # Release all controller inputs the moment the match ends so no model
        # button presses bleed into menus — libmelee takes over from here.
        if done:
            self.controller.release_all()
            self.cpu_controller.release_all()

        obs = self._make_observation(gamestate, done=done)

        # Delta-based reward (ported from MeleeSimEnv._compute_reward)
        damage_dealt = max(0.0, obs.opponent_damage - self._prev_opponent_damage)
        damage_taken = max(0.0, obs.player_damage - self._prev_player_damage)
        stocks_taken = max(0, self._prev_opponent_stocks - obs.opponent_stocks)

        reward = 0.0
        reward += damage_dealt * 0.01  # +0.01 per % dealt
        reward -= damage_taken * 0.01  # -0.01 per % taken
        reward += stocks_taken * 0.5  # +0.5 per stock taken

        # Progressive incentives (Pro-Play reward shaping) — use raw positions from gamestate
        p1_raw = gamestate.players.get(1) if gamestate else None
        p2_raw = gamestate.players.get(2) if gamestate else None
        if p1_raw and p2_raw and hasattr(p1_raw.position, "x") and hasattr(p2_raw.position, "x"):
            px = float(p1_raw.position.x)
            py = float(p1_raw.position.y)
            ox = float(p2_raw.position.x)
            oy = float(p2_raw.position.y)
            dx = px - ox
            dy = py - oy
            distance = math.sqrt(dx * dx + dy * dy)
            reward += DISTANCE_REWARD_SCALE * distance  # encourage facing/moving toward opponent
            off_stage = px < STAGE_LEFT_EDGE or px > STAGE_RIGHT_EDGE
            if not off_stage and abs(px) < CENTER_STAGE_X_THRESHOLD:
                reward += CENTER_STAGE_REWARD  # center stage bias (stage control)

        # Action clipping: penalize noise during hitlag or jumpsquat (use pre-step state)
        if prev_gamestate is not None:
            p1_prev = prev_gamestate.players.get(1)
            if p1_prev:
                hitlag = int(getattr(p1_prev, "hitlag_left", 0) or 0)
                jumpsquat = int(getattr(p1_prev, "jumpsquat_frames_left", 0) or 0)
                if (hitlag > 0 or jumpsquat > 0):
                    noise = (
                        abs(action.stick_x) > 0.1 or abs(action.stick_y) > 0.1
                        or abs(action.c_stick_x) > 0.1 or abs(action.c_stick_y) > 0.1
                        or action.button_a or action.button_b or action.button_z
                        or action.button_l or action.button_r
                    )
                    if noise:
                        reward += NOISE_PENALTY_HITLAG_JUMPSQUAT

        if done:
            if obs.opponent_stocks <= 0:
                reward += 1.0  # win bonus
            elif obs.player_stocks <= 0:
                reward -= 1.0  # loss penalty

        # Update trackers for next frame
        self._prev_player_damage = obs.player_damage
        self._prev_opponent_damage = obs.opponent_damage
        self._prev_opponent_stocks = obs.opponent_stocks

        obs.reward = reward
        return obs

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Terminate the Dolphin process cleanly."""
        if self._connected:
            self._connected = False
            try:
                self.console.stop()
            except Exception:
                log.warning("Error stopping Dolphin", exc_info=True)

    # ------------------------------------------------------------------
    def _normalize_position(self, x: float, y: float) -> tuple[float, float]:
        """Scale position relative to FD blastzones to ~[-1, 1]."""
        norm_x = (x - FD_MID_X) / FD_HALF_WIDTH_X
        norm_y = (y - FD_MID_Y) / FD_HALF_HEIGHT_Y
        return (
            max(-1.0, min(1.0, norm_x)),
            max(-1.0, min(1.0, norm_y)),
        )

    def _normalize_speed_y(self, vy: float, terminal: float, max_jump: float) -> float:
        """Normalize vertical self speed: negative by terminal_velocity, positive by max_jump_velocity."""
        if vy <= 0:
            return vy / terminal if terminal > 0 else 0.0
        return vy / max_jump if max_jump > 0 else 0.0

    def _extract_player_physics(
        self, p, prefix: str, *, terminal_velocity: float, max_jump_velocity: float, max_horizontal: float
    ) -> dict:
        """Extract libmelee PlayerState into observation dict; speeds normalized by character constants."""
        if p is None:
            return {}

        def g(name, default=0):
            return getattr(p, name, default)

        # 5-speed system (libmelee parity)
        air_x = float(g("speed_air_x_self", 0))
        ground_x = float(g("speed_ground_x_self", 0))
        y_self = float(g("speed_y_self", 0))
        x_attack = float(g("speed_x_attack", 0))
        y_attack = float(g("speed_y_attack", 0))
        on_ground = bool(g("on_ground", True))

        # Normalize 5 speeds by character limits
        air_x_n = max(-1.0, min(1.0, air_x / max_horizontal)) if max_horizontal > 0 else 0.0
        ground_x_n = max(-1.0, min(1.0, ground_x / max_horizontal)) if max_horizontal > 0 else 0.0
        y_self_n = self._normalize_speed_y(y_self, terminal_velocity, max_jump_velocity)
        y_self_n = max(-1.0, min(1.0, y_self_n))
        # Attack knockback can exceed character limits; use same horizontal scale, cap
        x_attack_n = max(-1.0, min(1.0, x_attack / max(1.0, max_horizontal * 1.5)))
        y_attack_n = max(-1.0, min(1.0, y_attack / max(terminal_velocity, max_jump_velocity)))

        # Total velocity (normalized) = sum of normalized components
        total_x = (ground_x_n if on_ground else air_x_n) + x_attack_n
        total_y = y_self_n + y_attack_n
        total_x = max(-1.0, min(1.0, total_x))
        total_y = max(-1.0, min(1.0, total_y))

        # ECB: libmelee has ecb_top, ecb_bottom, ecb_left, ecb_right as (x,y) tuples
        def ecb_point(name: str) -> dict:
            pt = g(name, None)
            if pt is None:
                return {"x": 0.0, "y": 0.0}
            if hasattr(pt, "x") and hasattr(pt, "y"):
                x_val = pt.x if pt.x is not None else 0.0
                y_val = pt.y if pt.y is not None else 0.0
                return {"x": float(x_val), "y": float(y_val)}
            if isinstance(pt, (tuple, list)) and len(pt) >= 2:
                x_val = pt[0] if pt[0] is not None else 0.0
                y_val = pt[1] if pt[1] is not None else 0.0
                return {"x": float(x_val), "y": float(y_val)}
            return {"x": 0.0, "y": 0.0}

        ecb = {
            "top": ecb_point("ecb_top"),
            "bottom": ecb_point("ecb_bottom"),
            "left": ecb_point("ecb_left"),
            "right": ecb_point("ecb_right"),
        }

        return {
            f"{prefix}speed_air_x_self": air_x_n,
            f"{prefix}speed_ground_x_self": ground_x_n,
            f"{prefix}speed_y_self": y_self_n,
            f"{prefix}speed_x_attack": x_attack_n,
            f"{prefix}speed_y_attack": y_attack_n,
            f"{prefix}speed_x": total_x,
            f"{prefix}speed_y": total_y,
            f"{prefix}on_ground": on_ground,
            f"{prefix}facing_right": bool(g("facing", True)),
            f"{prefix}hitstun_left": int(g("hitstun_frames_left", 0)),
            f"{prefix}hitlag_left": int(g("hitlag_left", 0)),
            f"{prefix}jumpsquat_frames_left": int(g("jumpsquat_frames_left", 0)),
            f"{prefix}invulnerability_left": int(g("invulnerability_left", 0)),
            f"{prefix}shield_strength": float(g("shield_strength", 60.0)),
            f"{prefix}ecb": ecb,
        }

    def _extract_projectiles(self, gamestate) -> list:
        """Extract projectiles from libmelee GameState."""
        projs = getattr(gamestate, "projectiles", None) or []
        out = []
        for pr in projs:
            pos = getattr(pr, "position", None) or (0.0, 0.0)
            spd = getattr(pr, "speed", None) or (0.0, 0.0)
            owner = getattr(pr, "owner", -1)
            if hasattr(pos, "x"):
                x, y = float(pos.x), float(pos.y)
            else:
                x = float(pos[0]) if len(pos) > 0 else 0.0
                y = float(pos[1]) if len(pos) > 1 else 0.0
            if hasattr(spd, "x"):
                sx, sy = float(spd.x), float(spd.y)
            else:
                sx = float(spd[0]) if len(spd) > 0 else 0.0
                sy = float(spd[1]) if len(spd) > 1 else 0.0
            out.append(
                {"x": x, "y": y, "speed_x": sx, "speed_y": sy, "owner_id": int(owner)}
            )
        return out

    def _make_observation(
        self,
        gamestate,
        reward: Optional[float] = None,
        done: bool = False,
    ) -> SmashObservation:
        """Extract the fields we care about from libmelee's GameState.
        If `gamestate` is `None` (which can happen on the very first call),
        we return a placeholder observation with zeros.
        """
        menu_state_str = "unknown"
        if gamestate is not None and gamestate.menu_state is not None:
            menu_state_str = gamestate.menu_state.name

        if gamestate is None:
            return SmashObservation(
                player_x=0.0,
                player_y=0.0,
                player_damage=0,
                player_action_state="unknown",
                player_stocks=0,
                opponent_x=0.0,
                opponent_y=0.0,
                opponent_damage=0,
                opponent_action_state="unknown",
                opponent_stocks=0,
                menu_state=menu_state_str,
                reward=reward,
                done=done,
            )

        p1 = gamestate.players.get(1)
        p2 = gamestate.players.get(2)

        # Positions scaled relative to blastzones (~[-1, 1])
        if p1 and hasattr(p1.position, "x"):
            p1_x, p1_y = self._normalize_position(float(p1.position.x), float(p1.position.y))
        else:
            p1_x, p1_y = 0.0, 0.0
        if p2 and hasattr(p2.position, "x"):
            p2_x, p2_y = self._normalize_position(float(p2.position.x), float(p2.position.y))
        else:
            p2_x, p2_y = 0.0, 0.0

        phys1 = self._extract_player_physics(
            p1, "player_",
            terminal_velocity=FOX_TERMINAL_VELOCITY,
            max_jump_velocity=FOX_MAX_JUMP_VELOCITY,
            max_horizontal=FOX_MAX_HORIZONTAL_SPEED,
        )
        phys2 = self._extract_player_physics(
            p2, "opponent_",
            terminal_velocity=PUFF_TERMINAL_VELOCITY,
            max_jump_velocity=PUFF_MAX_JUMP_VELOCITY,
            max_horizontal=PUFF_MAX_HORIZONTAL_SPEED,
        )
        projectiles = self._extract_projectiles(gamestate)

        return SmashObservation(
            # Player 1 (positions normalized by blastzones)
            player_x=p1_x,
            player_y=p1_y,
            player_damage=int(p1.percent) if p1 else 0,
            player_action_state=getattr(p1.action, "name", str(p1.action)) if p1 else "unknown",
            player_stocks=int(p1.stock) if p1 else 0,
            **phys1,
            # Player 2 (positions normalized by blastzones)
            opponent_x=p2_x,
            opponent_y=p2_y,
            opponent_damage=int(p2.percent) if p2 else 0,
            opponent_action_state=getattr(p2.action, "name", str(p2.action)) if p2 else "unknown",
            opponent_stocks=int(p2.stock) if p2 else 0,
            **phys2,
            # General
            projectiles=projectiles,
            frame=int(gamestate.frame) if hasattr(gamestate, "frame") else 0,
            menu_state=menu_state_str,
            reward=reward,
            done=done,
        )
