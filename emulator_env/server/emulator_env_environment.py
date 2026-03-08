# server/emulator_env_environment.py
"""OpenEnv server implementation that wraps libmelee (Slippi Dolphin).
It inherits from `openenv.core.env_server.interfaces.Environment` and defines:
* `reset()` – navigates menus, starts a match, returns initial observation.
* `step(action)` – translates a `SmashAction` into libmelee controller commands,
  advances one frame, and returns a `SmashObservation`.
* `state` property – returns the current session state.
* `close()` – shuts down the Dolphin process.

Environment variables:
    DOLPHIN_PATH  – (required) directory containing the Slippi Dolphin executable.
    ISO_PATH      – (optional) path to the Melee ISO file.
    DOLPHIN_HOME  – (optional) path to the Dolphin user/home directory.
                    If not set, libmelee creates a temporary directory.
"""

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

from ..models import SmashAction, SmashObservation

import melee
from melee import Console, Controller, Button

log = logging.getLogger(__name__)

# Maximum number of frames to spend in menu navigation before giving up.
_MAX_MENU_FRAMES = 3600  # ~60 seconds at 60 fps


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
        # Player 2 – CPU opponent.
        self.cpu_controller = Controller(self.console, port=2)

        # Menu navigation helper – must be instantiated once and reused.
        self.menu_helper = melee.MenuHelper()
        self.cpu_menu_helper = melee.MenuHelper()

        # Configurable match parameters (can be exposed later).
        self._character = melee.Character.FOX
        self._cpu_character = melee.Character.FALCO
        self._stage = melee.Stage.FINAL_DESTINATION
        self._cpu_level = 3  # CPU difficulty 1-9

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

        # Plug in both controllers.
        log.info("Connecting controllers ...")
        self.controller.connect()
        self.cpu_controller.connect()
        log.info("Controllers connected (port 1 + port 2 CPU).")

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
        """
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Reset delta-tracking for the new episode.
        self._prev_player_damage = 0.0
        self._prev_opponent_damage = 0.0
        self._prev_opponent_stocks = 4

        # Re-create menu helpers so internal state is clean for a new episode.
        self.menu_helper = melee.MenuHelper()
        self.cpu_menu_helper = melee.MenuHelper()

        log.info("reset(): navigating menus to start a new match ...")

        for frame_idx in range(_MAX_MENU_FRAMES):
            gamestate = self.console.step()
            if gamestate is None:
                continue

            if gamestate.menu_state in (
                melee.Menu.IN_GAME,
                melee.Menu.SUDDEN_DEATH,
            ):
                log.info("reset(): match started after %d menu frames.", frame_idx)
                return self._make_observation(gamestate, reward=0.0, done=False)

            # Let MenuHelper drive inputs for both controllers.
            self.menu_helper.menu_helper_simple(
                gamestate=gamestate,
                controller=self.controller,
                character_selected=self._character,
                stage_selected=self._stage,
                connect_code="",
                cpu_level=0,  # 0 = bot/human controlled (our RL agent)
                costume=0,
                autostart=True,
                swag=False,
            )
            self.cpu_menu_helper.menu_helper_simple(
                gamestate=gamestate,
                controller=self.cpu_controller,
                character_selected=self._cpu_character,
                stage_selected=self._stage,
                connect_code="",
                cpu_level=self._cpu_level,
                costume=0,
                autostart=False,  # only one controller should autostart
                swag=False,
            )

        # If we never reached IN_GAME, return a placeholder.
        log.warning("reset(): timed out waiting for match to start!")
        gamestate = self.console.step()
        return self._make_observation(gamestate, reward=0.0, done=False)

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

        # Translate the Pydantic action into libmelee controller calls.

        # Analog sticks
        self.controller.tilt_analog_unit(
            Button.BUTTON_MAIN, action.stick_x, action.stick_y
        )
        self.controller.tilt_analog_unit(
            Button.BUTTON_C, action.c_stick_x, action.c_stick_y
        )

        # Digital buttons – press or release each one.
        _digital_buttons = (
            (action.button_a, Button.BUTTON_A),
            (action.button_b, Button.BUTTON_B),
            (action.button_x, Button.BUTTON_X),
            (action.button_y, Button.BUTTON_Y),
            (action.button_z, Button.BUTTON_Z),
            (action.button_l, Button.BUTTON_L),
            (action.button_r, Button.BUTTON_R),
        )
        for pressed, button in _digital_buttons:
            if pressed:
                self.controller.press_button(button)
            else:
                self.controller.release_button(button)

        # Advance the emulator one frame and fetch the new gamestate.
        gamestate = self.console.step()

        # Determine whether the match ended.
        done = (
            gamestate is not None and gamestate.menu_state == melee.Menu.POSTGAME_SCORES
        )

        obs = self._make_observation(gamestate, done=done)

        # Delta-based reward (ported from MeleeSimEnv._compute_reward)
        damage_dealt = max(0.0, obs.opponent_damage - self._prev_opponent_damage)
        damage_taken = max(0.0, obs.player_damage - self._prev_player_damage)
        stocks_taken = max(0, self._prev_opponent_stocks - obs.opponent_stocks)

        reward = 0.0
        reward += damage_dealt * 0.01   # +0.01 per % dealt
        reward -= damage_taken * 0.01   # -0.01 per % taken
        reward += stocks_taken * 0.5    # +0.5 per stock taken

        if done:
            if obs.opponent_stocks <= 0:
                reward += 1.0           # win bonus
            elif obs.player_stocks <= 0:
                reward -= 1.0           # loss penalty

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

        return SmashObservation(
            # Player 1
            player_x=float(p1.position.x) if p1 else 0.0,
            player_y=float(p1.position.y) if p1 else 0.0,
            player_damage=int(p1.percent) if p1 else 0,
            player_action_state=p1.action.name if p1 else "unknown",
            player_stocks=int(p1.stock) if p1 else 0,
            # Player 1 physics
            player_speed_x=float(getattr(p1, 'speed_x_attack', 0) + getattr(p1, 'speed_ground_x_self', 0)) if p1 else 0.0,
            player_speed_y=float(getattr(p1, 'speed_y_attack', 0) + getattr(p1, 'speed_y_self', 0)) if p1 else 0.0,
            player_on_ground=bool(p1.on_ground) if p1 else True,
            player_facing_right=bool(p1.facing) if p1 else True,
            player_hitstun_left=int(p1.hitstun_frames_left) if p1 else 0,
            # Player 2
            opponent_x=float(p2.position.x) if p2 else 0.0,
            opponent_y=float(p2.position.y) if p2 else 0.0,
            opponent_damage=int(p2.percent) if p2 else 0,
            opponent_action_state=p2.action.name if p2 else "unknown",
            opponent_stocks=int(p2.stock) if p2 else 0,
            # Player 2 physics
            opponent_speed_x=float(getattr(p2, 'speed_x_attack', 0) + getattr(p2, 'speed_ground_x_self', 0)) if p2 else 0.0,
            opponent_speed_y=float(getattr(p2, 'speed_y_attack', 0) + getattr(p2, 'speed_y_self', 0)) if p2 else 0.0,
            opponent_on_ground=bool(p2.on_ground) if p2 else True,
            opponent_facing_right=bool(p2.facing) if p2 else True,
            opponent_hitstun_left=int(p2.hitstun_frames_left) if p2 else 0,
            # General
            frame=int(gamestate.frame) if hasattr(gamestate, 'frame') else 0,
            menu_state=menu_state_str,
            reward=reward,
            done=done,
        )
