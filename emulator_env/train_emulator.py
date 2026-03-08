#!/usr/bin/env python3
"""Run jiggly.pt (P1 Jigglypuff) against mango.pt (P2 Fox) via the OpenEnv emulator.

Port 1: Jigglypuff, controlled by this client using jiggly.pt.
Port 2: Fox (Mango), controlled by mango.pt on the server.

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
           uv run python train_emulator.py --checkpoint ../checkpoints/jiggly.pt
           uv run python train_emulator.py --deterministic --num-episodes 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field

from emulator_env import EmulatorEnv
from emulator_env.models import SmashAction, SmashObservation
from emulator_env.policy_runner import (
    ActorCriticMLP,
    action_to_smash,
    get_action,
    load_model,
    obs_to_vector,
)

log = logging.getLogger(__name__)


# ===================================================================
# EPISODE METRICS
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
        # 1. Observation -> vector (P1 perspective)
        obs_vec = obs_to_vector(obs, player_idx=0)

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
        description="Run jiggly.pt (P1 Jigglypuff) vs mango.pt (P2 Fox) via OpenEnv emulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints/puff_final.pt",
        help="Path to P1 Jigglypuff policy (e.g. puff_final.pt)",
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
