"""Deterministic frame-by-frame Melee-like simulator.

Public interface:
    sim = Simulator()
    state = sim.reset()
    state = sim.step(state, [action_p1, action_p2])

Each ``action`` is a dict with keys:
    stick_x : float in [-1, 1]  (negative = left, positive = right)
    stick_y : float in [-1, 1]  (negative = down, positive = up)
    jump    : bool
    attack  : bool               (jab if grounded neutral, smash if stick != 0)
    grab    : bool
    special : bool               (down + special = Rest)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from physics.constants import (
    Action,
    CHAR,
    GRAB_HOLD_FRAMES,
    GRAB_PUMMEL_DAMAGE,
    MAX_FRAMES,
    MOVES,
    STAGE,
    THROW_ANGLE,
    THROW_BKB,
    THROW_DAMAGE,
    THROW_KBG,
)
from physics.melee_physics import (
    apply_air_friction,
    apply_gravity,
    apply_traction,
    compute_hitstun,
    compute_knockback,
    decay_attack_velocity,
    knockback_to_velocity,
)
from physics.state import CharacterState, GameState, MoveData, Stage


NEUTRAL_ACTION: Dict = {
    "stick_x": 0.0,
    "stick_y": 0.0,
    "jump": False,
    "attack": False,
    "grab": False,
    "special": False,
}


def _move_data(name: str) -> MoveData:
    return MoveData(**MOVES[name])


class Simulator:
    """Stateless simulator — all mutable data lives in GameState."""

    def reset(self, seed: Optional[int] = None) -> GameState:
        max_j = CHAR["max_jumps"]
        return GameState(
            frame=0,
            players=[
                CharacterState(x=-10.0, y=0.0, facing_right=True,
                               on_ground=True, jumps_left=max_j),
                CharacterState(x=10.0, y=0.0, facing_right=False,
                               on_ground=True, jumps_left=max_j),
            ],
            stage=Stage(),
            done=False,
            winner=None,
        )

    def step(self, state: GameState, actions: List[Dict]) -> GameState:
        """Advance one frame. Returns a *new* GameState (does not mutate input)."""
        gs = state.copy()
        gs.frame += 1

        for i, p in enumerate(gs.players):
            act = actions[i] if i < len(actions) else NEUTRAL_ACTION
            other = gs.players[1 - i]
            self._process_player(p, act, other, i, gs)

        self._resolve_hitboxes(gs)

        for p in gs.players:
            self._apply_physics(p, gs.stage)
            self._stage_collision(p, gs.stage)

        self._check_deaths(gs)

        if gs.frame >= MAX_FRAMES:
            gs.done = True

        return gs

    # ------------------------------------------------------------------
    # Per-player action processing
    # ------------------------------------------------------------------

    def _process_player(
        self,
        p: CharacterState,
        act: Dict,
        other: CharacterState,
        idx: int,
        gs: GameState,
    ) -> None:
        if p.action == Action.DEAD:
            return
        if p.action == Action.RESPAWN_INVULN:
            p.action_frame += 1
            if p.action_frame >= CHAR["respawn_invuln_frames"]:
                p.action = Action.AIRBORNE
                p.invulnerable_frames_left = 0
            else:
                p.invulnerable_frames_left = (
                    CHAR["respawn_invuln_frames"] - p.action_frame
                )
            return

        if p.invulnerable_frames_left > 0:
            p.invulnerable_frames_left -= 1

        if p.grabbed_by is not None:
            p.grab_timer -= 1
            if p.grab_timer <= 0:
                self._release_grab(p, gs)
            return

        if p.hitstun_frames_left > 0:
            p.hitstun_frames_left -= 1
            if p.hitstun_frames_left <= 0:
                p.action = Action.AIRBORNE if not p.on_ground else Action.IDLE
            return

        if p.action == Action.REST_SLEEP:
            p.action_frame += 1
            move_name: str = getattr(p, "_current_move_name", "rest")
            md = _move_data(move_name)
            if p.action_frame >= md.endlag_frames:
                p.action = Action.IDLE if p.on_ground else Action.AIRBORNE
                p.action_frame = 0
            return

        if p.action in (
            Action.ATTACK_STARTUP,
            Action.ATTACK_ACTIVE,
            Action.ATTACK_ENDLAG,
            Action.GRAB_STARTUP,
            Action.GRAB_ACTIVE,
            Action.GRAB_ENDLAG,
            Action.THROW,
        ):
            self._advance_attack_state(p)
            return

        if p.action == Action.JUMPSQUAT:
            p.action_frame += 1
            if p.action_frame >= CHAR["jumpsquat_frames"]:
                p.on_ground = False
                full_hop = act.get("stick_y", 0) > 0.0 or act.get("jump", False)
                p.speed_y_self = (
                    CHAR["jump_velocity"] if full_hop else CHAR["short_hop_velocity"]
                )
                p.action = Action.AIRBORNE
                p.action_frame = 0
                p.jumps_left -= 1
            return

        if p.action == Action.LANDING:
            p.action_frame += 1
            if p.action_frame >= CHAR["landing_frames"]:
                p.action = Action.IDLE
                p.action_frame = 0
            return

        # --- Actionable states: IDLE, WALK, RUN, AIRBORNE ---

        sx = act.get("stick_x", 0.0)
        sy = act.get("stick_y", 0.0)
        jump = act.get("jump", False)
        attack = act.get("attack", False)
        grab = act.get("grab", False)
        special = act.get("special", False)

        # Only allow Rest when close enough to potentially hit
        REST_PROXIMITY = 5.0
        dist = abs(p.x - other.x) + abs(p.y - other.y)
        if dist > REST_PROXIMITY:
            special = False

        if p.on_ground:
            self._ground_actions(p, sx, sy, jump, attack, grab, special)
        else:
            self._air_actions(p, sx, sy, jump, attack, grab, special)

    # ------------------------------------------------------------------
    # Ground actions
    # ------------------------------------------------------------------

    def _ground_actions(
        self,
        p: CharacterState,
        sx: float,
        sy: float,
        jump: bool,
        attack: bool,
        grab: bool,
        special: bool = False,
    ) -> None:
        if special and sy < -0.3:
            self._start_rest(p)
            return
        if grab:
            self._start_move(p, "grab")
            return
        if attack:
            move_name = "smash" if abs(sx) > 0.5 or sy > 0.5 else "jab"
            self._start_move(p, move_name)
            return
        if jump:
            p.action = Action.JUMPSQUAT
            p.action_frame = 0
            return

        if abs(sx) > 0.6:
            p.action = Action.RUN
            p.facing_right = sx > 0
            target = CHAR["run_speed"] * (1.0 if sx > 0 else -1.0)
            p.speed_x_self += (target - p.speed_x_self) * 0.3
        elif abs(sx) > 0.2:
            p.action = Action.WALK
            p.facing_right = sx > 0
            target = CHAR["walk_speed"] * (1.0 if sx > 0 else -1.0)
            p.speed_x_self += (target - p.speed_x_self) * 0.25
        else:
            p.action = Action.IDLE

    # ------------------------------------------------------------------
    # Air actions
    # ------------------------------------------------------------------

    def _air_actions(
        self,
        p: CharacterState,
        sx: float,
        sy: float,
        jump: bool,
        attack: bool,
        grab: bool,
        special: bool = False,
    ) -> None:
        if special and sy < -0.3:
            self._start_rest(p)
            return
        if attack:
            move_name = "smash" if abs(sx) > 0.5 or sy > 0.5 else "jab"
            self._start_move(p, move_name)
            return

        if jump and p.jumps_left > 0:
            p.speed_y_self = CHAR["double_jump_velocity"]
            p.jumps_left -= 1

        if abs(sx) > 0.1:
            p.facing_right = sx > 0
            target = CHAR["air_speed"] * (1.0 if sx > 0 else -1.0)
            diff = target - p.speed_x_self
            p.speed_x_self += math.copysign(
                min(abs(diff), CHAR["air_accel"]), diff
            )

    # ------------------------------------------------------------------
    # Attack state machine
    # ------------------------------------------------------------------

    def _start_move(self, p: CharacterState, name: str) -> None:
        md = _move_data(name)
        p._current_move_name = name  # type: ignore[attr-defined]
        p.attack_connected = False
        if md.is_grab:
            p.action = Action.GRAB_STARTUP
        else:
            p.action = Action.ATTACK_STARTUP
        p.action_frame = 0

    def _start_rest(self, p: CharacterState) -> None:
        """Rest: frame-1 hitbox, then 240 frames of sleep."""
        p._current_move_name = "rest"  # type: ignore[attr-defined]
        p.attack_connected = False
        p.action = Action.ATTACK_ACTIVE
        p.action_frame = 0
        p.speed_x_self = 0.0

    def _advance_attack_state(self, p: CharacterState) -> None:
        name: str = getattr(p, "_current_move_name", "jab")
        md = _move_data(name)
        p.action_frame += 1

        if p.action in (Action.ATTACK_STARTUP, Action.GRAB_STARTUP):
            if p.action_frame >= md.startup_frames:
                p.action = (
                    Action.GRAB_ACTIVE if md.is_grab else Action.ATTACK_ACTIVE
                )
                p.action_frame = 0
        elif p.action in (Action.ATTACK_ACTIVE, Action.GRAB_ACTIVE):
            if p.action_frame >= md.active_frames:
                if name == "rest":
                    p.action = Action.REST_SLEEP
                    p.action_frame = 0
                else:
                    p.action = (
                        Action.GRAB_ENDLAG if md.is_grab else Action.ATTACK_ENDLAG
                    )
                    p.action_frame = 0
        elif p.action in (Action.ATTACK_ENDLAG, Action.GRAB_ENDLAG):
            if p.action_frame >= md.endlag_frames:
                p.action = Action.IDLE if p.on_ground else Action.AIRBORNE
                p.action_frame = 0
        elif p.action == Action.THROW:
            if p.action_frame >= 10:
                p.action = Action.IDLE if p.on_ground else Action.AIRBORNE
                p.action_frame = 0

    # ------------------------------------------------------------------
    # Hitbox resolution (runs once per frame after all players processed)
    # ------------------------------------------------------------------

    def _resolve_hitboxes(self, gs: GameState) -> None:
        for i, attacker in enumerate(gs.players):
            if attacker.action not in (Action.ATTACK_ACTIVE, Action.GRAB_ACTIVE):
                continue
            if attacker.attack_connected:
                continue

            name: str = getattr(attacker, "_current_move_name", "jab")
            md = _move_data(name)
            defender = gs.players[1 - i]

            if defender.invulnerable_frames_left > 0:
                continue
            if defender.action == Action.DEAD:
                continue

            if self._hitbox_check(attacker, defender, md):
                attacker.attack_connected = True
                if md.is_grab:
                    self._apply_grab(attacker, defender, i, gs)
                else:
                    self._apply_hit(attacker, defender, md)

    def _hitbox_check(
        self, attacker: CharacterState, defender: CharacterState, md: MoveData
    ) -> bool:
        hx = attacker.x + attacker.facing_sign * md.hitbox_x_range * 0.5
        dx = abs(defender.x - hx)
        dy = abs(defender.y - attacker.y)
        return dx <= md.hitbox_x_range * 0.5 and dy <= md.hitbox_y_range

    def _apply_hit(
        self,
        attacker: CharacterState,
        defender: CharacterState,
        md: MoveData,
    ) -> None:
        defender.percent += md.damage
        kb = compute_knockback(
            target_percent=defender.percent,
            damage=md.damage,
            weight=CHAR["weight"],
            base_knockback=md.base_knockback,
            knockback_growth=md.knockback_growth,
        )
        launch_sign = -attacker.facing_sign
        vx, vy = knockback_to_velocity(kb, md.angle, launch_sign)
        defender.speed_x_attack = vx
        defender.speed_y_attack = vy
        defender.hitstun_frames_left = compute_hitstun(kb)
        defender.action = Action.HITSTUN
        defender.action_frame = 0
        defender.on_ground = False

    def _apply_grab(
        self,
        attacker: CharacterState,
        defender: CharacterState,
        attacker_idx: int,
        gs: GameState,
    ) -> None:
        if defender.hitstun_frames_left > 0 or defender.grabbed_by is not None:
            return
        defender.grabbed_by = attacker_idx
        defender.action = Action.GRABBED
        defender.action_frame = 0
        escape_frames = GRAB_HOLD_FRAMES + int(defender.percent * 0.3)
        defender.grab_timer = escape_frames

        attacker.action = Action.THROW
        attacker.action_frame = 0
        defender.percent += GRAB_PUMMEL_DAMAGE

    def _release_grab(self, defender: CharacterState, gs: GameState) -> None:
        grabber_idx = defender.grabbed_by
        if grabber_idx is not None:
            grabber = gs.players[grabber_idx]
            defender.percent += THROW_DAMAGE
            kb = compute_knockback(
                target_percent=defender.percent,
                damage=THROW_DAMAGE,
                weight=CHAR["weight"],
                base_knockback=THROW_BKB,
                knockback_growth=THROW_KBG,
            )
            launch_sign = -grabber.facing_sign
            vx, vy = knockback_to_velocity(kb, THROW_ANGLE, launch_sign)
            defender.speed_x_attack = vx
            defender.speed_y_attack = vy
            defender.hitstun_frames_left = compute_hitstun(kb)
            defender.on_ground = False

        defender.grabbed_by = None
        defender.grab_timer = 0
        defender.action = Action.HITSTUN
        defender.action_frame = 0

    # ------------------------------------------------------------------
    # Physics integration (per player)
    # ------------------------------------------------------------------

    def _apply_physics(self, p: CharacterState, stage: Stage) -> None:
        if p.action in (Action.DEAD, Action.RESPAWN_INVULN):
            return

        if not p.on_ground:
            p.speed_y_self = apply_gravity(p.speed_y_self)
            if p.hitstun_frames_left <= 0:
                p.speed_x_self = apply_air_friction(p.speed_x_self)
        else:
            p.speed_y_self = 0.0
            if p.hitstun_frames_left <= 0 and p.action in (
                Action.IDLE, Action.LANDING,
                Action.ATTACK_STARTUP, Action.ATTACK_ACTIVE, Action.ATTACK_ENDLAG,
                Action.GRAB_STARTUP, Action.GRAB_ACTIVE, Action.GRAB_ENDLAG,
                Action.THROW, Action.REST_SLEEP,
            ):
                p.speed_x_self = apply_traction(p.speed_x_self)

        p.speed_x_attack, p.speed_y_attack = decay_attack_velocity(
            p.speed_x_attack, p.speed_y_attack
        )

        p.x += p.speed_x_self + p.speed_x_attack
        p.y += p.speed_y_self + p.speed_y_attack

    # ------------------------------------------------------------------
    # Stage collision
    # ------------------------------------------------------------------

    def _stage_collision(self, p: CharacterState, stage: Stage) -> None:
        if p.action == Action.DEAD:
            return

        if p.y <= stage.floor_y and p.speed_y_self <= 0:
            if stage.left_edge <= p.x <= stage.right_edge:
                p.y = stage.floor_y
                p.speed_y_self = 0.0
                p.speed_y_attack = 0.0
                was_airborne = not p.on_ground
                p.on_ground = True
                p.jumps_left = CHAR["max_jumps"]
                if was_airborne and p.hitstun_frames_left <= 0:
                    if p.action not in (
                        Action.ATTACK_STARTUP,
                        Action.ATTACK_ACTIVE,
                        Action.ATTACK_ENDLAG,
                    ):
                        p.action = Action.LANDING
                        p.action_frame = 0
            else:
                p.on_ground = False
        elif p.y > stage.floor_y:
            p.on_ground = False

    # ------------------------------------------------------------------
    # Death + respawn
    # ------------------------------------------------------------------

    def _check_deaths(self, gs: GameState) -> None:
        for i, p in enumerate(gs.players):
            if p.action == Action.DEAD or p.action == Action.RESPAWN_INVULN:
                continue

            s = gs.stage
            if (
                p.x < s.left_blastzone
                or p.x > s.right_blastzone
                or p.y > s.top_blastzone
                or p.y < s.bottom_blastzone
            ):
                p.stock -= 1
                if p.stock <= 0:
                    p.action = Action.DEAD
                    gs.done = True
                    gs.winner = 1 - i
                else:
                    self._respawn(p)

    def _respawn(self, p: CharacterState) -> None:
        p.x = CHAR["respawn_x"]
        p.y = CHAR["respawn_y"]
        p.percent = 0.0
        p.speed_x_self = 0.0
        p.speed_y_self = 0.0
        p.speed_x_attack = 0.0
        p.speed_y_attack = 0.0
        p.hitstun_frames_left = 0
        p.on_ground = False
        p.action = Action.RESPAWN_INVULN
        p.action_frame = 0
        p.invulnerable_frames_left = CHAR["respawn_invuln_frames"]
        p.jumps_left = CHAR["max_jumps"]
        p.grabbed_by = None
        p.grab_timer = 0
        p.attack_connected = False
