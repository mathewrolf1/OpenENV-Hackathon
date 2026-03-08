#!/usr/bin/env python3
"""Run a trained Puff model against real Melee via the OpenEnv emulator bridge.

Loads a checkpoint (default: ../checkpoints/puff_final.pt), connects to the
emulator server over WebSocket, and runs inference-only evaluation episodes.

The script bridges two incompatible interfaces:
  - Sim model: 26-dim float32 observation, MultiDiscrete([5,4,2,2,2,2]) action
  - Emulator:  SmashObservation (Pydantic), SmashAction (GameCube controller)

Translation layers handle the conversion in both directions.

Prerequisites:
    1. Install dependencies (once):
           cd emulator_env
           uv sync

    2. Start the emulator server (Dolphin + OpenEnv) in a separate terminal:
           cd emulator_env
           uv run --project . server

    3. Run this script:
           cd emulator_env
           uv run python train_emulator.py
           uv run python train_emulator.py --checkpoint ../checkpoints/puff_final.pt
           uv run python train_emulator.py --server-url http://remote-host:8000
           uv run python train_emulator.py --deterministic --num-episodes 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from emulator_env import EmulatorEnv
from emulator_env.models import SmashAction, SmashObservation

log = logging.getLogger(__name__)


# ===================================================================
# INLINED CONSTANTS  (from physics/constants.py and envs/melee_sim_env.py)
#
# These are copied here so the script can run inside the emulator_env
# venv without needing gymnasium or the project-root packages.
# Keep in sync if the sim's observation format ever changes.
# ===================================================================


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

    NUM_ACTIONS = 32  # sentinel for observation encoding


# Observation: 13 features per player, 2 players = 26
OBS_DIM = 26

# Discrete stick bins (must match envs/melee_sim_env.py)
STICK_X_BINS = [-1.0, -0.6, 0.0, 0.6, 1.0]
STICK_Y_BINS = [-1.0, 0.0, 0.5, 1.0]

# Action head sizes
ACTION_NVEC = [5, 4, 2, 2, 2, 2]
NUM_ACTIONS_FLAT = sum(ACTION_NVEC)


# ===================================================================
# 1. ACTION STATE MAPPING  (libmelee string -> sim Action IntEnum)
# ===================================================================

_ACTION_STATE_MAP: Dict[str, int] = {}


def _register(sim_action: int, *prefixes: str) -> None:
    """Register multiple libmelee name prefixes to one sim Action."""
    for p in prefixes:
        _ACTION_STATE_MAP[p] = sim_action


# Ground neutral
_register(
    Action.IDLE,
    "STANDING",
    "WAIT",
    "TURNING",
    "TURNING_RUN",
    "CROUCH",
    "CROUCHING",
    "SQUAT",
    "EDGE",
    "CLIFF",
    "TECH",
    "NEUTRAL_GET_UP",
    "GROUND_GET_UP",
    "PLATFORM_DROP",
    "SHIELD",
    "GUARD",
    "ESCAPE",
    "ROLL",
    "SPOT_DODGE",
)

# Locomotion
_register(Action.WALK, "WALK", "SLOW_WALK")
_register(Action.RUN, "RUNNING", "RUN", "DASH", "DASHING")

# Aerial states
_register(Action.JUMPSQUAT, "KNEE_BEND")
_register(
    Action.AIRBORNE,
    "JUMPING",
    "JUMP",
    "FALL",
    "AERIAL",
    "FALLING",
    "DOUBLE_JUMP",
    "WALL_JUMP",
    "MIDAIR",
    "PASS",
)
_register(Action.LANDING, "LANDING", "LAND")

# Attacks -- mapped to ATTACK_ACTIVE (we can't tell startup vs endlag
# from the state string alone)
_register(
    Action.ATTACK_ACTIVE,
    "ATTACK",
    "NEUTRAL_B",
    "SIDE_B",
    "DOWN_B",
    "UP_B",
    "FSMASH",
    "DSMASH",
    "USMASH",
    "JAB",
    "FTILT",
    "DTILT",
    "UTILT",
    "NAIR",
    "FAIR",
    "BAIR",
    "DAIR",
    "UAIR",
    "SWORD_DANCE",
    "SMASH",
    "LOOPING_ATTACK",
    "GETUP_ATTACK",
    "EDGE_ATTACK",
    "NEUTRAL_SPECIAL",
    "SIDE_SPECIAL",
    "DOWN_SPECIAL",
    "UP_SPECIAL",
    "SING",
    "REST",
    "ROLLOUT",
    "POUND",
)

# Grab / throw
_register(Action.GRAB_STARTUP, "GRAB_PULLING", "GRAB_RUNNING", "GRAB_WAIT")
_register(Action.GRAB_ACTIVE, "GRAB", "CATCH", "GRABBING")
_register(Action.GRAB_ENDLAG, "GRAB_PUMMEL")
_register(Action.GRABBED, "CAPTURED", "GRAB_PUMMELED")
_register(
    Action.THROW,
    "THROW",
    "FORWARD_THROW",
    "BACK_THROW",
    "UP_THROW",
    "DOWN_THROW",
)

# Hitstun / tumble
_register(Action.HITSTUN, "DAMAGE", "KNOCKBACK", "FLYING_BACK", "THROWN")
_register(Action.TUMBLE, "TUMBLING", "TUMBLE")

# Rest sleep (Puff-specific endlag)
_register(Action.REST_SLEEP, "REST_WAIT", "FURA_SLEEP")

# Dead / respawn
_register(Action.DEAD, "DEAD", "DYING", "ON_HALO", "WAIT_HALO")
_register(Action.RESPAWN_INVULN, "REBIRTH", "ENTRY", "REBORN")


def map_action_state(state_str: str) -> int:
    """Map a libmelee action state string to a sim Action int.

    Tries exact match first, then prefix match.
    Falls back to IDLE for unknown states.
    """
    upper = state_str.upper()

    # Exact match
    if upper in _ACTION_STATE_MAP:
        return _ACTION_STATE_MAP[upper]

    # Prefix match (e.g. "WALK_SLOW" -> WALK, "DASH_ATTACK" -> DASH)
    for prefix, action in _ACTION_STATE_MAP.items():
        if upper.startswith(prefix):
            return action

    return Action.IDLE


# ===================================================================
# 2. OBSERVATION ADAPTER  (SmashObservation -> 26-dim float32 vector)
# ===================================================================


def obs_to_vector(obs: SmashObservation) -> np.ndarray:
    """Convert a SmashObservation into the same 26-dim normalized vector
    that the sim's _build_obs() produces.

    Per-player features (13 each, agent then opponent):
        x/100, y/100,
        speed_x_self/5, speed_y_self/5,
        speed_x_attack/5, speed_y_attack/5,
        percent/200, stocks/4,
        on_ground, facing_right,
        action_enum/32, action_frame/60,
        hitstun/60

    Approximations:
    - Emulator combines self+attack velocity; we use total as self, zero attack.
    - action_frame not exposed by SmashObservation; defaults to 0.
    """
    player_action = map_action_state(obs.player_action_state)
    opp_action = map_action_state(obs.opponent_action_state)

    player_vec = np.array(
        [
            obs.player_x / 100.0,
            obs.player_y / 100.0,
            obs.player_speed_x / 5.0,
            obs.player_speed_y / 5.0,
            0.0,  # attack vel x (not split)
            0.0,  # attack vel y
            obs.player_damage / 200.0,
            obs.player_stocks / 4.0,
            float(obs.player_on_ground),
            float(obs.player_facing_right),
            float(player_action) / float(Action.NUM_ACTIONS),
            0.0,  # action_frame (unavailable)
            obs.player_hitstun_left / 60.0,
        ],
        dtype=np.float32,
    )

    opp_vec = np.array(
        [
            obs.opponent_x / 100.0,
            obs.opponent_y / 100.0,
            obs.opponent_speed_x / 5.0,
            obs.opponent_speed_y / 5.0,
            0.0,
            0.0,
            obs.opponent_damage / 200.0,
            obs.opponent_stocks / 4.0,
            float(obs.opponent_on_ground),
            float(obs.opponent_facing_right),
            float(opp_action) / float(Action.NUM_ACTIONS),
            0.0,
            obs.opponent_hitstun_left / 60.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate([player_vec, opp_vec])


# ===================================================================
# 3. ACTION ADAPTER  (MultiDiscrete[6] -> SmashAction)
# ===================================================================


def action_to_smash(action_indices: np.ndarray) -> SmashAction:
    """Convert a MultiDiscrete([5,4,2,2,2,2]) action array to a SmashAction.

    Mapping:
        [0] stick_x index -> STICK_X_BINS  -> stick_x
        [1] stick_y index -> STICK_Y_BINS  -> stick_y
        [2] jump    (0/1)                  -> button_x
        [3] attack  (0/1)                  -> button_a
        [4] grab    (0/1)                  -> button_z
        [5] special (0/1)                  -> button_b
    """
    return SmashAction(
        stick_x=STICK_X_BINS[int(action_indices[0])],
        stick_y=STICK_Y_BINS[int(action_indices[1])],
        c_stick_x=0.0,
        c_stick_y=0.0,
        button_a=bool(action_indices[3]),  # attack
        button_b=bool(action_indices[5]),  # special
        button_x=bool(action_indices[2]),  # jump
        button_y=False,
        button_z=bool(action_indices[4]),  # grab
        button_l=False,
        button_r=False,
    )


# ===================================================================
# 4. MODEL  (self-contained ActorCriticMLP -- no project-root imports)
# ===================================================================


class ActorCriticMLP(nn.Module):
    """Inference-only mirror of mango_trainer.ActorCriticMLP."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.action_nvec = tuple(ACTION_NVEC)
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_dim, NUM_ACTIONS_FLAT)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return self.actor_head(features)


def load_model(checkpoint_path: str, device: str = "cpu") -> ActorCriticMLP:
    """Load a .pt checkpoint into the ActorCriticMLP model."""
    model = ActorCriticMLP(obs_dim=OBS_DIM)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    total_frames = ckpt.get("total_frames", "unknown")
    log.info(
        "Loaded checkpoint: %s (trained for %s frames)", checkpoint_path, total_frames
    )
    return model


@torch.no_grad()
def get_action(
    model: ActorCriticMLP,
    obs_vector: np.ndarray,
    deterministic: bool = False,
) -> np.ndarray:
    """Run model inference and return a MultiDiscrete action array."""
    obs_t = torch.from_numpy(obs_vector).float().unsqueeze(0)
    logits_flat = model(obs_t)

    actions = []
    offset = 0
    for n in ACTION_NVEC:
        logits = logits_flat[:, offset : offset + n]
        if deterministic:
            a = logits.argmax(dim=-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        actions.append(a.item())
        offset += n

    return np.array(actions, dtype=np.int64)


# ===================================================================
# 5. EPISODE METRICS
# ===================================================================


@dataclass
class EpisodeMetrics:
    """Accumulates per-episode statistics."""

    episode: int = 0
    frames: int = 0
    total_reward: float = 0.0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    stocks_taken: int = 0
    stocks_lost: int = 0
    wall_time_s: float = 0.0

    _prev_player_damage: float = field(default=0.0, repr=False)
    _prev_opponent_damage: float = field(default=0.0, repr=False)
    _prev_player_stocks: int = field(default=4, repr=False)
    _prev_opponent_stocks: int = field(default=4, repr=False)

    def update(self, obs: SmashObservation, reward: float) -> None:
        self.frames += 1
        self.total_reward += reward or 0.0

        dmg_dealt = max(0.0, obs.opponent_damage - self._prev_opponent_damage)
        dmg_taken = max(0.0, obs.player_damage - self._prev_player_damage)
        stk_taken = max(0, self._prev_opponent_stocks - obs.opponent_stocks)
        stk_lost = max(0, self._prev_player_stocks - obs.player_stocks)

        self.damage_dealt += dmg_dealt
        self.damage_taken += dmg_taken
        self.stocks_taken += stk_taken
        self.stocks_lost += stk_lost

        self._prev_player_damage = obs.player_damage
        self._prev_opponent_damage = obs.opponent_damage
        self._prev_player_stocks = obs.player_stocks
        self._prev_opponent_stocks = obs.opponent_stocks

    def summary(self) -> str:
        fps = self.frames / self.wall_time_s if self.wall_time_s > 0 else 0
        return (
            f"Episode {self.episode:3d} | "
            f"{self.frames:5d} frames ({self.wall_time_s:.1f}s, {fps:.0f} fps) | "
            f"reward={self.total_reward:+8.3f} | "
            f"dmg dealt={self.damage_dealt:6.1f}  taken={self.damage_taken:6.1f} | "
            f"stocks +{self.stocks_taken} -{self.stocks_lost}"
        )


# ===================================================================
# 6. MAIN EVAL LOOP
# ===================================================================

MAX_EPISODE_FRAMES = 8 * 60 * 60  # 8 minutes at 60fps


def run_episode(
    client: EmulatorEnv,
    model: ActorCriticMLP,
    episode_num: int,
    deterministic: bool = False,
) -> EpisodeMetrics:
    """Run one full episode: reset -> step until done -> return metrics."""
    metrics = EpisodeMetrics(episode=episode_num)

    log.info("Episode %d: resetting environment (navigating menus)...", episode_num)
    result = client.reset()
    obs = result.observation

    metrics._prev_player_damage = obs.player_damage
    metrics._prev_opponent_damage = obs.opponent_damage
    metrics._prev_player_stocks = obs.player_stocks
    metrics._prev_opponent_stocks = obs.opponent_stocks

    start_time = time.time()

    for frame in range(MAX_EPISODE_FRAMES):
        # 1. Observation -> vector
        obs_vec = obs_to_vector(obs)

        # 2. Model inference
        action_indices = get_action(model, obs_vec, deterministic=deterministic)

        # 3. Action -> SmashAction and step
        smash_action = action_to_smash(action_indices)
        result = client.step(smash_action)
        obs = result.observation

        # 4. Track metrics
        metrics.update(obs, result.reward)

        # 5. Done?
        if result.done:
            log.info("Episode %d: match ended at frame %d", episode_num, frame + 1)
            break

        # Progress log every ~10 seconds (600 frames)
        if (frame + 1) % 600 == 0:
            elapsed = time.time() - start_time
            log.info(
                "  frame %5d | reward=%+.3f | P1 %d%% (%d stk)  P2 %d%% (%d stk) | %.1fs",
                frame + 1,
                metrics.total_reward,
                obs.player_damage,
                obs.player_stocks,
                obs.opponent_damage,
                obs.opponent_stocks,
                elapsed,
            )
    else:
        log.warning("Episode %d: hit frame limit (%d)", episode_num, MAX_EPISODE_FRAMES)

    metrics.wall_time_s = time.time() - start_time
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trained Puff model against real Melee via OpenEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints/puff_final.pt",
        help="Path to the .pt model checkpoint",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the OpenEnv emulator server",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes to run",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax (greedy) actions instead of sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for model inference (cpu or cuda)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Validate checkpoint ---
    if not os.path.isfile(args.checkpoint):
        log.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    # --- Load model ---
    log.info("Loading model from %s ...", args.checkpoint)
    model = load_model(args.checkpoint, device=args.device)
    param_count = sum(p.numel() for p in model.parameters())
    log.info("Model loaded: %d parameters", param_count)

    # --- Connect to emulator server ---
    log.info("Connecting to emulator server at %s ...", args.server_url)
    try:
        client = EmulatorEnv(base_url=args.server_url)
        client.connect()
    except ConnectionError as e:
        log.error("Failed to connect: %s", e)
        log.error("Is the server running?  Start it with:")
        log.error("    uv run --project . server")
        sys.exit(1)

    log.info("Connected. Running %d episode(s).", args.num_episodes)
    log.info(
        "Mode: %s",
        "deterministic (argmax)" if args.deterministic else "stochastic (sampling)",
    )
    log.info("-" * 70)

    # --- Run episodes ---
    all_metrics: List[EpisodeMetrics] = []

    try:
        for ep in range(1, args.num_episodes + 1):
            metrics = run_episode(client, model, ep, deterministic=args.deterministic)
            all_metrics.append(metrics)
            log.info(metrics.summary())
            log.info("-" * 70)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        client.close()
        log.info("Client closed.")

    # --- Aggregate stats ---
    if all_metrics:
        n = len(all_metrics)
        avg_reward = sum(m.total_reward for m in all_metrics) / n
        avg_dmg_dealt = sum(m.damage_dealt for m in all_metrics) / n
        avg_dmg_taken = sum(m.damage_taken for m in all_metrics) / n
        total_stk_taken = sum(m.stocks_taken for m in all_metrics)
        total_stk_lost = sum(m.stocks_lost for m in all_metrics)
        total_frames = sum(m.frames for m in all_metrics)
        total_time = sum(m.wall_time_s for m in all_metrics)

        log.info("=" * 70)
        log.info("RESULTS  (%d episodes)", n)
        log.info("  Total frames:   %d (%.1f seconds)", total_frames, total_time)
        log.info("  Avg reward:     %+.3f", avg_reward)
        log.info("  Avg dmg dealt:  %.1f", avg_dmg_dealt)
        log.info("  Avg dmg taken:  %.1f", avg_dmg_taken)
        log.info("  Stocks taken:   %d   lost: %d", total_stk_taken, total_stk_lost)
        log.info("=" * 70)


if __name__ == "__main__":
    main()
