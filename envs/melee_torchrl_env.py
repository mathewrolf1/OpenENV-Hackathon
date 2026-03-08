"""Production-ready MeleeTorchEnv bridging libmelee to TorchRL.

Inherits from torchrl.envs.EnvBase. Uses libmelee for Dolphin emulation with
2-frame input latency buffer, Mango reward shaping, and full observation parity.
"""

from __future__ import annotations

import configparser
import logging
import os
from collections import deque
from typing import Any, Optional

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, MultiCategorical, Unbounded
from torchrl.envs import EnvBase

from rewards.competitive import CompetitiveMeleeReward

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Python 3.14 compatibility: configparser + duplicate keys in Slippi Dolphin.ini
# ---------------------------------------------------------------------------
_OrigConfigParser = configparser.ConfigParser


class _LenientConfigParser(_OrigConfigParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("strict", False)
        super().__init__(*args, **kwargs)


configparser.ConfigParser = _LenientConfigParser  # type: ignore[misc]

# Lazy import to avoid requiring libmelee at import time
melee = None
Console = None
Controller = None
Button = None
MenuHelper = None


def _ensure_melee():
    global melee, Console, Controller, Button, MenuHelper
    if melee is None:
        import melee as _melee
        from melee import Console as _Console, Controller as _Controller, Button as _Button
        melee = _melee
        Console = _Console
        Controller = _Controller
        Button = _Button
        MenuHelper = _melee.MenuHelper


# ---------------------------------------------------------------------------
# Normalization constants (map raw values to [-1, 1])
# ---------------------------------------------------------------------------
POS_SCALE = 250.0   # x, y roughly in [-250, 250]
PERCENT_SCALE = 200.0  # percent 0..999 -> [-1, 1] via (p/200 - 2.5)
VEL_SCALE = 5.0
SHIELD_SCALE = 60.0
TIMER_SCALE = 60.0
ECB_SCALE = 50.0

STAGE_RIGHT_EDGE = 68.4


def _norm_pos(v: float) -> float:
    return max(-1.0, min(1.0, v / POS_SCALE))


def _norm_percent(p: float) -> float:
    return max(-1.0, min(1.0, (p / PERCENT_SCALE) - 2.5))


def _norm_vel(v: float) -> float:
    return max(-1.0, min(1.0, v / VEL_SCALE))


def _norm_shield(s: float) -> float:
    return max(-1.0, min(1.0, (s / SHIELD_SCALE) * 2.0 - 1.0))


def _norm_timer(t: int) -> float:
    return max(-1.0, min(1.0, float(t) / TIMER_SCALE))


def _norm_ecb(v: float) -> float:
    return max(-1.0, min(1.0, v / ECB_SCALE))


# ---------------------------------------------------------------------------
# MeleeTorchEnv
# ---------------------------------------------------------------------------


class MeleeTorchEnv(EnvBase):
    """TorchRL environment wrapping libmelee (Dolphin). Agent controls P1 (Fox)."""

    def __init__(
        self,
        dolphin_path: Optional[str] = None,
        iso_path: Optional[str] = None,
        dolphin_home: Optional[str] = None,
        device: Optional[torch.device] = None,
        batch_size: Optional[tuple] = None,
        skip_emulator_init: bool = False,
        **kwargs,
    ):
        super().__init__(device=device, batch_size=batch_size or (), **kwargs)

        self._dolphin_path = dolphin_path or os.environ.get("DOLPHIN_PATH")
        self._iso_path = iso_path or os.environ.get("ISO_PATH")
        self._dolphin_home = dolphin_home or os.environ.get("DOLPHIN_HOME")
        self._skip_emulator_init = skip_emulator_init

        if not skip_emulator_init and (not self._dolphin_path or not os.path.isdir(self._dolphin_path)):
            raise RuntimeError(
                "DOLPHIN_PATH must be set to the Slippi Dolphin directory. "
                "Example: export DOLPHIN_PATH=/path/to/Slippi-Dolphin"
            )

        self._make_specs()
        self._connected = False
        self._console = None
        self._controller = None
        self._cpu_controller = None
        self._menu_helper = None
        self._cpu_menu_helper = None
        self._character = None  # set in _init_emulator: melee.Character.FOX
        self._cpu_character = None
        self._stage = None  # melee.Stage.BATTLEFIELD
        self._cpu_level = 3

        # 2-frame input latency buffer: agent action queued, action from 2 frames ago sent
        self._action_buffer: deque = deque(maxlen=2)

        # Unified reward calculator
        self._reward_calc = CompetitiveMeleeReward()

        if not skip_emulator_init:
            _ensure_melee()
            self._init_emulator()

    def _make_specs(self) -> None:
        """Build observation_spec, action_spec, reward_spec, done_spec."""
        device = self.device
        shape = self.batch_size

        def obs_bounded(low: float = -1.0, high: float = 1.0, s: tuple = ()):
            return Bounded(low=low, high=high, shape=(*shape, *s) if s else shape, dtype=torch.float32, device=device)

        # Per-player observation (5 velocities, ECB, shield, timers, position, etc.)
        player_obs = Composite(
            # 5-speed system
            speed_air_x_self=obs_bounded(),
            speed_ground_x_self=obs_bounded(),
            speed_y_self=obs_bounded(),
            speed_x_attack=obs_bounded(),
            speed_y_attack=obs_bounded(),
            # ECB (4 points x 2 coords)
            ecb_top_x=obs_bounded(),
            ecb_top_y=obs_bounded(),
            ecb_bottom_x=obs_bounded(),
            ecb_bottom_y=obs_bounded(),
            ecb_left_x=obs_bounded(),
            ecb_left_y=obs_bounded(),
            ecb_right_x=obs_bounded(),
            ecb_right_y=obs_bounded(),
            # Shield & timers
            shield_strength=obs_bounded(),
            hitlag_left=obs_bounded(),
            jumpsquat_frames_left=obs_bounded(),
            invulnerability_left=obs_bounded(),
            # Position & state
            x=obs_bounded(),
            y=obs_bounded(),
            percent=obs_bounded(),
            stock=obs_bounded(),
            on_ground=obs_bounded(),
            facing_right=obs_bounded(),
            shape=shape,
            device=device,
        )

        self.observation_spec = Composite(
            p1=player_obs,
            p2=player_obs,
            shape=shape,
            device=device,
        )

        # Action: main_stick (2), c_stick (2), buttons (6: A, B, X, Y, Z, Start)
        self.action_spec = Composite(
            main_stick=Composite(
                x=Bounded(low=-1.0, high=1.0, shape=(*shape, 1), dtype=torch.float32, device=device),
                y=Bounded(low=-1.0, high=1.0, shape=(*shape, 1), dtype=torch.float32, device=device),
                shape=shape,
                device=device,
            ),
            c_stick=Composite(
                x=Bounded(low=-1.0, high=1.0, shape=(*shape, 1), dtype=torch.float32, device=device),
                y=Bounded(low=-1.0, high=1.0, shape=(*shape, 1), dtype=torch.float32, device=device),
                shape=shape,
                device=device,
            ),
            buttons=MultiCategorical(
                nvec=[2, 2, 2, 2, 2, 2],  # A, B, X, Y, Z, Start
                shape=(*shape, 6),
                device=device,
            ),
            shape=shape,
            device=device,
        )

        self.reward_spec = Unbounded(shape=(*shape, 1), device=device)
        self.done_spec = Composite(
            done=Bounded(low=0, high=1, shape=(*shape, 1), dtype=torch.bool, device=device),
            terminated=Bounded(low=0, high=1, shape=(*shape, 1), dtype=torch.bool, device=device),
            shape=shape,
            device=device,
        )

    def _init_emulator(self) -> None:
        self._character = melee.Character.FOX
        self._cpu_character = melee.Character.FALCO
        self._stage = melee.Stage.BATTLEFIELD

        console_kwargs = dict(path=self._dolphin_path, fullscreen=False)
        if self._dolphin_home:
            console_kwargs["dolphin_home_path"] = self._dolphin_home
            console_kwargs["tmp_home_directory"] = False

        self._console = Console(**console_kwargs)
        self._controller = Controller(self._console, port=1)
        self._cpu_controller = Controller(self._console, port=2)
        self._menu_helper = MenuHelper()
        self._cpu_menu_helper = MenuHelper()

        log.info("Launching Dolphin from %s ...", self._dolphin_path)
        self._console.run(iso_path=self._iso_path)

        if not self._console.connect():
            raise RuntimeError("Failed to connect to Dolphin. Ensure Slippi is enabled.")

        self._controller.connect()
        self._cpu_controller.connect()
        self._connected = True
        log.info("MeleeTorchEnv: Dolphin connected.")

    def _extract_player_obs(self, p, prefix: str) -> dict:
        """Extract libmelee PlayerState into normalized observation dict."""
        if p is None:
            return {f"{prefix}{k}": torch.tensor(0.0, device=self.device) for k in [
                "speed_air_x_self", "speed_ground_x_self", "speed_y_self",
                "speed_x_attack", "speed_y_attack",
                "ecb_top_x", "ecb_top_y", "ecb_bottom_x", "ecb_bottom_y",
                "ecb_left_x", "ecb_left_y", "ecb_right_x", "ecb_right_y",
                "shield_strength", "hitlag_left", "jumpsquat_frames_left", "invulnerability_left",
                "x", "y", "percent", "stock", "on_ground", "facing_right",
            ]}

        def g(name, default=0):
            return getattr(p, name, default)

        def ecb_pt(name: str) -> tuple[float, float]:
            pt = g(name, (0.0, 0.0))
            if hasattr(pt, "x"):
                return float(pt.x), float(pt.y)
            return (float(pt[0]) if len(pt) > 0 else 0.0, float(pt[1]) if len(pt) > 1 else 0.0)

        air_x = float(g("speed_air_x_self", 0))
        ground_x = float(g("speed_ground_x_self", 0))
        y_self = float(g("speed_y_self", 0))
        x_attack = float(g("speed_x_attack", 0))
        y_attack = float(g("speed_y_attack", 0))
        t, b, l, r = ecb_pt("ecb_top"), ecb_pt("ecb_bottom"), ecb_pt("ecb_left"), ecb_pt("ecb_right")

        return {
            f"{prefix}speed_air_x_self": _norm_vel(air_x),
            f"{prefix}speed_ground_x_self": _norm_vel(ground_x),
            f"{prefix}speed_y_self": _norm_vel(y_self),
            f"{prefix}speed_x_attack": _norm_vel(x_attack),
            f"{prefix}speed_y_attack": _norm_vel(y_attack),
            f"{prefix}ecb_top_x": _norm_ecb(t[0]), f"{prefix}ecb_top_y": _norm_ecb(t[1]),
            f"{prefix}ecb_bottom_x": _norm_ecb(b[0]), f"{prefix}ecb_bottom_y": _norm_ecb(b[1]),
            f"{prefix}ecb_left_x": _norm_ecb(l[0]), f"{prefix}ecb_left_y": _norm_ecb(l[1]),
            f"{prefix}ecb_right_x": _norm_ecb(r[0]), f"{prefix}ecb_right_y": _norm_ecb(r[1]),
            f"{prefix}shield_strength": _norm_shield(float(g("shield_strength", 60.0))),
            f"{prefix}hitlag_left": _norm_timer(int(g("hitlag_left", 0))),
            f"{prefix}jumpsquat_frames_left": _norm_timer(int(g("jumpsquat_frames_left", 0))),
            f"{prefix}invulnerability_left": _norm_timer(int(g("invulnerability_left", 0))),
            f"{prefix}x": _norm_pos(float(p.position.x) if hasattr(p.position, "x") else p.position[0]),
            f"{prefix}y": _norm_pos(float(p.position.y) if hasattr(p.position, "y") else p.position[1]),
            f"{prefix}percent": _norm_percent(float(g("percent", 0))),
            f"{prefix}stock": max(-1.0, min(1.0, (float(g("stock", 4)) / 4.0) * 2.0 - 1.0)),
            f"{prefix}on_ground": 1.0 if g("on_ground", True) else -1.0,
            f"{prefix}facing_right": 1.0 if g("facing", True) else -1.0,
        }

    def _gamestate_to_obs_tensordict(self, gamestate) -> TensorDict:
        """Convert libmelee GameState to TensorDict matching observation_spec.
        Returns TensorDict(p1=..., p2=...) for use as observation.
        """
        p1 = gamestate.players.get(1) if gamestate else None
        p2 = gamestate.players.get(2) if gamestate else None

        obs1 = self._extract_player_obs(p1, "p1.")
        obs2 = self._extract_player_obs(p2, "p2.")

        p1_dict = {}
        p2_dict = {}
        for k, v in obs1.items():
            key = k.replace("p1.", "")
            t = torch.tensor(v, dtype=torch.float32, device=self.device)
            if self.batch_size:
                t = t.expand(*self.batch_size)
            p1_dict[key] = t
        for k, v in obs2.items():
            key = k.replace("p2.", "")
            t = torch.tensor(v, dtype=torch.float32, device=self.device)
            if self.batch_size:
                t = t.expand(*self.batch_size)
            p2_dict[key] = t

        return TensorDict(
            {"p1": p1_dict, "p2": p2_dict},
            batch_size=self.batch_size,
            device=self.device,
        )

    def _action_tensordict_to_controller(self, action_td: TensorDict) -> None:
        """Apply action from TensorDict to libmelee Controller.
        action_td is the action Composite: main_stick, c_stick, buttons.
        """
        main = action_td.get("main_stick", None)
        c_stick = action_td.get("c_stick", None)
        buttons = action_td.get("buttons", None)

        if main is not None:
            x = main.get("x", torch.tensor(0.0))
            y = main.get("y", torch.tensor(0.0))
            if hasattr(x, "item"):
                x, y = x.item(), y.item()
            else:
                x, y = float(x.flatten()[0]), float(y.flatten()[0])
            self._controller.tilt_analog_unit(Button.BUTTON_MAIN, x, y)
        else:
            self._controller.tilt_analog_unit(Button.BUTTON_MAIN, 0.0, 0.0)

        if c_stick is not None:
            x = c_stick.get("x", torch.tensor(0.0))
            y = c_stick.get("y", torch.tensor(0.0))
            if hasattr(x, "item"):
                x, y = x.item(), y.item()
            else:
                x, y = float(x.flatten()[0]), float(y.flatten()[0])
            self._controller.tilt_analog_unit(Button.BUTTON_C, x, y)
        else:
            self._controller.tilt_analog_unit(Button.BUTTON_C, 0.0, 0.0)

        btn_map = [(Button.BUTTON_A, 0), (Button.BUTTON_B, 1), (Button.BUTTON_X, 2),
                   (Button.BUTTON_Y, 3), (Button.BUTTON_Z, 4)]
        if buttons is not None:
            b = buttons.flatten()
            for i, (btn, idx) in enumerate(btn_map):
                if idx < len(b) and b[idx].item() > 0:
                    self._controller.press_button(btn)
                else:
                    self._controller.release_button(btn)
            # Start (index 5) - typically not used in-match
            start_btn = getattr(Button, "BUTTON_START", None)
            if start_btn is not None and len(b) > 5 and b[5].item() > 0:
                self._controller.press_button(start_btn)
            elif start_btn is not None:
                self._controller.release_button(start_btn)
        else:
            for btn, _ in btn_map:
                self._controller.release_button(btn)
            start_btn = getattr(Button, "BUTTON_START", None)
            if start_btn is not None:
                self._controller.release_button(start_btn)

        # L/R for shield
        self._controller.release_button(Button.BUTTON_L)
        self._controller.release_button(Button.BUTTON_R)

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        if self._skip_emulator_init or self._console is None:
            raise RuntimeError("MeleeTorchEnv: cannot reset without emulator. Construct with skip_emulator_init=False and valid DOLPHIN_PATH.")
        self._action_buffer.clear()
        self._reward_calc.reset()

        self._menu_helper = MenuHelper()
        self._cpu_menu_helper = MenuHelper()

        for _ in range(3600):
            gamestate = self._console.step()
            if gamestate is None:
                continue
            if gamestate.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                obs_td = self._gamestate_to_obs_tensordict(gamestate)
                done_t = torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)
                obs_td["done"] = done_t
                obs_td["terminated"] = done_t.clone()
                return obs_td

            self._menu_helper.menu_helper_simple(
                gamestate=gamestate,
                controller=self._controller,
                character_selected=self._character,
                stage_selected=self._stage,
                connect_code="",
                cpu_level=0,
                costume=0,
                autostart=True,
                swag=False,
            )
            self._cpu_menu_helper.menu_helper_simple(
                gamestate=gamestate,
                controller=self._cpu_controller,
                character_selected=self._cpu_character,
                stage_selected=self._stage,
                connect_code="",
                cpu_level=self._cpu_level,
                costume=0,
                autostart=False,
                swag=False,
            )

        log.warning("MeleeTorchEnv reset: timed out waiting for match")
        return self._gamestate_to_obs_tensordict(self._console.step())

    def _step(self, tensordict: TensorDict) -> TensorDict:
        if self._skip_emulator_init or self._console is None:
            raise RuntimeError("MeleeTorchEnv: cannot step without emulator. Construct with skip_emulator_init=False and valid DOLPHIN_PATH.")
        action_td = tensordict.get("action", tensordict)
        if isinstance(action_td, TensorDict):
            self._action_buffer.append(action_td.clone())
        else:
            self._action_buffer.append(action_td)

        # Send action from 2 frames ago (or neutral if buffer not full)
        if len(self._action_buffer) >= 2:
            delayed_action = self._action_buffer[0]
            self._action_tensordict_to_controller(delayed_action)
        else:
            # Neutral until buffer fills
            self._controller.tilt_analog_unit(Button.BUTTON_MAIN, 0.0, 0.0)
            self._controller.tilt_analog_unit(Button.BUTTON_C, 0.0, 0.0)
            for btn in [Button.BUTTON_A, Button.BUTTON_B, Button.BUTTON_X, Button.BUTTON_Y, Button.BUTTON_Z]:
                self._controller.release_button(btn)

        gamestate = self._console.step()

        done = False
        if gamestate is None:
            done = True
            log.warning("MeleeTorchEnv: gamestate None (desync)")
        elif gamestate.menu_state == melee.Menu.POSTGAME_SCORES:
            done = True

        p1 = gamestate.players.get(1) if gamestate else None
        p2 = gamestate.players.get(2) if gamestate else None

        # Determine winner index (0-based for CompetitiveMeleeReward)
        winner = None
        if done and gamestate:
            w = getattr(gamestate, "winner", None)
            if w is not None:
                winner = 0 if w == 1 else 1  # port 1 -> idx 0, port 2 -> idx 1

        reward, reward_info = self._reward_calc.step(p1, p2, done=done, winner=winner)

        obs_td = self._gamestate_to_obs_tensordict(gamestate) if gamestate else self.observation_spec.zero()
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.bool, device=self.device)
        if self.batch_size:
            reward_t = reward_t.expand(*self.batch_size, 1)
            done_t = done_t.expand(*self.batch_size, 1)
        else:
            reward_t = reward_t.unsqueeze(-1)
            done_t = done_t.unsqueeze(-1)

        out = TensorDict(
            {
                "observation": obs_td,
                "reward": reward_t,
                "done": done_t,
                "terminated": done_t.clone(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        for k, v in reward_info.items():
            out[k] = torch.tensor(v, device=self.device)

        return out

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            torch.manual_seed(seed)

    def close(self) -> None:
        """Kill Dolphin process to prevent memory leaks."""
        if self._connected and self._console is not None:
            self._connected = False
            try:
                self._console.stop()
            except Exception:
                log.warning("Error stopping Dolphin", exc_info=True)
            self._console = None
            self._controller = None
            self._cpu_controller = None
