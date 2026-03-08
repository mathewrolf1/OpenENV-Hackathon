"""Policy inference for Fox (P1) and Puff (P2) checkpoints.

Converts SmashObservation <-> 26-dim sim obs and MultiDiscrete action <-> SmashAction.
Shared by the emulator server (puff_final.pt on P2) and train_emulator client (Fox policy on P1).
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .models import SmashAction, SmashObservation

log = logging.getLogger(__name__)

# Sim constants (mirror train_emulator / mango_trainer)
OBS_DIM = 26
STICK_X_BINS = [-1.0, -0.6, 0.0, 0.6, 1.0]
STICK_Y_BINS = [-1.0, 0.0, 0.5, 1.0]
ACTION_NVEC = [5, 4, 2, 2, 2, 2]
NUM_ACTIONS_FLAT = sum(ACTION_NVEC)


class Action(IntEnum):
    """Sim action states (mirrors physics.constants.Action)."""

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
    REST_SLEEP = 25
    DEAD = 30
    RESPAWN_INVULN = 31
    NUM_ACTIONS = 32


# Action state mapping (libmelee string -> sim Action)
_ACTION_STATE_MAP: Dict[str, int] = {}


def _register(sim_action: int, *prefixes: str) -> None:
    for p in prefixes:
        _ACTION_STATE_MAP[p] = sim_action


_register(Action.IDLE, "STANDING", "WAIT", "TURNING", "TURNING_RUN", "CROUCH", "CROUCHING", "SQUAT", "EDGE", "CLIFF", "TECH", "NEUTRAL_GET_UP", "GROUND_GET_UP", "PLATFORM_DROP", "SHIELD", "GUARD", "ESCAPE", "ROLL", "SPOT_DODGE")
_register(Action.WALK, "WALK", "SLOW_WALK")
_register(Action.RUN, "RUNNING", "RUN", "DASH", "DASHING")
_register(Action.JUMPSQUAT, "KNEE_BEND")
_register(Action.AIRBORNE, "JUMPING", "JUMP", "FALL", "AERIAL", "FALLING", "DOUBLE_JUMP", "WALL_JUMP", "MIDAIR", "PASS")
_register(Action.LANDING, "LANDING", "LAND")
_register(Action.ATTACK_ACTIVE, "ATTACK", "NEUTRAL_B", "SIDE_B", "DOWN_B", "UP_B", "FSMASH", "DSMASH", "USMASH", "JAB", "FTILT", "DTILT", "UTILT", "NAIR", "FAIR", "BAIR", "DAIR", "UAIR", "SWORD_DANCE", "SMASH", "LOOPING_ATTACK", "GETUP_ATTACK", "EDGE_ATTACK", "NEUTRAL_SPECIAL", "SIDE_SPECIAL", "DOWN_SPECIAL", "UP_SPECIAL", "SING", "REST", "ROLLOUT", "POUND")
_register(Action.GRAB_STARTUP, "GRAB_PULLING", "GRAB_RUNNING", "GRAB_WAIT")
_register(Action.GRAB_ACTIVE, "GRAB", "CATCH", "GRABBING")
_register(Action.GRAB_ENDLAG, "GRAB_PUMMEL")
_register(Action.GRABBED, "CAPTURED", "GRAB_PUMMELED")
_register(Action.THROW, "THROW", "FORWARD_THROW", "BACK_THROW", "UP_THROW", "DOWN_THROW")
_register(Action.HITSTUN, "DAMAGE", "KNOCKBACK", "FLYING_BACK", "THROWN")
_register(Action.TUMBLE, "TUMBLING", "TUMBLE")
_register(Action.REST_SLEEP, "REST_WAIT", "FURA_SLEEP")
_register(Action.DEAD, "DEAD", "DYING", "ON_HALO", "WAIT_HALO")
_register(Action.RESPAWN_INVULN, "REBIRTH", "ENTRY", "REBORN")


def map_action_state(state_str: str) -> int:
    upper = (state_str or "").upper()
    if upper in _ACTION_STATE_MAP:
        return _ACTION_STATE_MAP[upper]
    for prefix, action in _ACTION_STATE_MAP.items():
        if upper.startswith(prefix):
            return action
    return Action.IDLE


def _player_vec(
    x: float, y: float,
    speed_x: float, speed_y: float,
    damage: float, stocks: int,
    on_ground: bool, facing_right: bool,
    action_state: str, hitstun: float,
) -> np.ndarray:
    """Build 13-dim player vector. Positions and speeds are expected normalized in [-1,1] from OpenEnv server."""
    act = map_action_state(action_state)
    return np.array([
        x, y,
        speed_x, speed_y, 0.0, 0.0,
        damage / 200.0, stocks / 4.0,
        float(on_ground), float(facing_right),
        float(act) / float(Action.NUM_ACTIONS), 0.0,
        hitstun / 60.0,
    ], dtype=np.float32)


def obs_to_vector(obs: SmashObservation, player_idx: int = 0) -> np.ndarray:
    """Convert SmashObservation to 26-dim vector.

    player_idx=0: P1 perspective (player first, opponent second)
    player_idx=1: P2 perspective (opponent first, player second) for mango on port 2.
    """
    p1_vec = _player_vec(
        obs.player_x, obs.player_y,
        getattr(obs, "player_speed_x", 0.0), getattr(obs, "player_speed_y", 0.0),
        obs.player_damage, obs.player_stocks,
        getattr(obs, "player_on_ground", True), getattr(obs, "player_facing_right", True),
        obs.player_action_state, float(getattr(obs, "player_hitstun_left", 0)),
    )
    p2_vec = _player_vec(
        obs.opponent_x, obs.opponent_y,
        getattr(obs, "opponent_speed_x", 0.0), getattr(obs, "opponent_speed_y", 0.0),
        obs.opponent_damage, obs.opponent_stocks,
        getattr(obs, "opponent_on_ground", True), getattr(obs, "opponent_facing_right", True),
        obs.opponent_action_state, float(getattr(obs, "opponent_hitstun_left", 0)),
    )
    if player_idx == 0:
        return np.concatenate([p1_vec, p2_vec])
    return np.concatenate([p2_vec, p1_vec])


def action_to_smash(action_indices: np.ndarray) -> SmashAction:
    """MultiDiscrete action -> SmashAction."""
    return SmashAction(
        stick_x=STICK_X_BINS[int(action_indices[0])],
        stick_y=STICK_Y_BINS[int(action_indices[1])],
        c_stick_x=0.0,
        c_stick_y=0.0,
        button_a=bool(action_indices[3]),
        button_b=bool(action_indices[5]),
        button_x=bool(action_indices[2]),
        button_y=False,
        button_z=bool(action_indices[4]),
        button_l=False,
        button_r=False,
    )


class ActorCriticMLP(nn.Module):
    """Inference-only mirror of mango_trainer.ActorCriticMLP."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.action_nvec = tuple(ACTION_NVEC)
        layers_list: list = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers_list.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers_list)
        self.actor_head = nn.Linear(hidden_dim, NUM_ACTIONS_FLAT)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self.backbone(obs))


def load_model(checkpoint_path: str, device: str = "cpu") -> ActorCriticMLP:
    model = ActorCriticMLP(obs_dim=OBS_DIM)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Loaded checkpoint: %s", checkpoint_path)
    return model


@torch.no_grad()
def get_action(model: ActorCriticMLP, obs_vector: np.ndarray, deterministic: bool = True) -> np.ndarray:
    obs_t = torch.from_numpy(obs_vector).float().unsqueeze(0)
    logits_flat = model(obs_t)
    actions = []
    offset = 0
    for n in ACTION_NVEC:
        logits = logits_flat[:, offset : offset + n]
        a = logits.argmax(dim=-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        actions.append(a.item())
        offset += n
    return np.array(actions, dtype=np.int64)
