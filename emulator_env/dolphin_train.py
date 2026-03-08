#!/usr/bin/env python3
"""PPO fine-tuning client for Dolphin via OpenEnv.

Connects to the emulator server, loads a sim-trained .pt checkpoint, and
runs PPO training against real Melee gameplay.  Port 2 (opponent) is
controlled by a frozen model on the server side.

Usage:
    # Start Dolphin server first:
    cd emulator_env && uv run --project . server

    # Fine-tune Puff (default):
    cd emulator_env && uv run python dolphin_train.py --agent puff

    # Fine-tune with Mango reward shaping:
    cd emulator_env && uv run python dolphin_train.py --agent mango

    # Custom checkpoint + frames:
    cd emulator_env && uv run python dolphin_train.py \
        --checkpoint ../checkpoints/puff_final.pt \
        --agent puff --total-frames 100000
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from emulator_env import EmulatorEnv
from emulator_env.models import SmashAction, SmashObservation
from emulator_env.policy_runner import (
    ACTION_NVEC,
    ActorCriticMLP,
    OBS_DIM,
    action_to_smash,
    load_model,
    obs_to_vector,
)
from rewards.competitive import CompetitiveMeleeReward
from rewards.puff import PuffReward, PuffWeights

log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("../checkpoints")

# ---------------------------------------------------------------------------
# Lightweight player-like object for reward calculators
# ---------------------------------------------------------------------------

class _ObsProxy:
    """Wraps SmashObservation fields into a player-like object that
    PuffReward/CompetitiveMeleeReward can read via getattr."""

    def __init__(self, obs: SmashObservation, prefix: str):
        self._obs = obs
        self._prefix = prefix

    def __getattr__(self, name: str):
        prefixed = f"{self._prefix}_{name}"
        if hasattr(self._obs, prefixed):
            return getattr(self._obs, prefixed)
        if hasattr(self._obs, name):
            return getattr(self._obs, name)
        if name == "x":
            return getattr(self._obs, f"{self._prefix}_x", 0.0)
        if name == "y":
            return getattr(self._obs, f"{self._prefix}_y", 0.0)
        if name == "percent":
            return getattr(self._obs, f"{self._prefix}_damage", 0)
        if name == "stock":
            return getattr(self._obs, f"{self._prefix}_stocks", 4)
        if name == "action":
            return getattr(self._obs, f"{self._prefix}_action_state", "IDLE")
        if name == "on_ground":
            return getattr(self._obs, f"{self._prefix}_on_ground", True)
        if name == "facing_right":
            return getattr(self._obs, f"{self._prefix}_facing_right", True)
        if name == "hitstun_frames_left":
            return getattr(self._obs, f"{self._prefix}_hitstun_left", 0)
        if name == "speed_x_self":
            return getattr(self._obs, f"{self._prefix}_speed_ground_x_self", 0.0)
        if name == "speed_y_self":
            return getattr(self._obs, f"{self._prefix}_speed_y_self", 0.0)
        if name == "speed_x_attack":
            return getattr(self._obs, f"{self._prefix}_speed_x_attack", 0.0)
        if name == "speed_y_attack":
            return getattr(self._obs, f"{self._prefix}_speed_y_attack", 0.0)
        if name == "shield_strength":
            return getattr(self._obs, f"{self._prefix}_shield_strength", 60.0)
        if name == "hitlag_left":
            return getattr(self._obs, f"{self._prefix}_hitlag_left", 0)
        if name == "attack_connected":
            return False
        if name == "off_stage":
            # player_x is normalized by FD_HALF_WIDTH_X (224.0); stage edge
            # is at ±68.4 raw units → ±0.305 normalized.
            x = getattr(self._obs, f"{self._prefix}_x", 0.0)
            return abs(x) > 0.305
        if name == "jumps_left":
            # SmashObservation does not expose jumps_left; return a sentinel
            # so recovery logic in puff.py can still use it for the sim.
            return getattr(self._obs, f"{self._prefix}_jumps_left", -1)
        return 0


def _make_proxies(obs: SmashObservation):
    return _ObsProxy(obs, "player"), _ObsProxy(obs, "opponent")


# ---------------------------------------------------------------------------
# Live training stats
# ---------------------------------------------------------------------------

@dataclass
class TrainingStats:
    """Tracks and displays live training metrics."""

    start_time: float = field(default_factory=time.time)
    total_frames: int = 0
    target_frames: int = 0
    episode_count: int = 0

    _recent_ep_rewards: deque = field(default_factory=lambda: deque(maxlen=20))
    _recent_ep_dmg_dealt: deque = field(default_factory=lambda: deque(maxlen=20))
    _recent_ep_dmg_taken: deque = field(default_factory=lambda: deque(maxlen=20))
    _recent_ep_stocks_taken: deque = field(default_factory=lambda: deque(maxlen=20))
    _recent_ep_stocks_lost: deque = field(default_factory=lambda: deque(maxlen=20))
    _recent_ep_lengths: deque = field(default_factory=lambda: deque(maxlen=20))
    _recent_wins: deque = field(default_factory=lambda: deque(maxlen=20))
    _recent_recovery_success: deque = field(default_factory=lambda: deque(maxlen=20))

    _ep_dmg_dealt: float = 0.0
    _ep_dmg_taken: float = 0.0
    _ep_stocks_taken: int = 0
    _ep_stocks_lost: int = 0
    _ep_frames: int = 0
    _prev_opp_dmg: float = 0.0
    _prev_self_dmg: float = 0.0
    _prev_opp_stocks: int = 4
    _prev_self_stocks: int = 4

    last_policy_loss: float = 0.0
    last_value_loss: float = 0.0
    last_entropy: float = 0.0
    last_batch_reward: float = 0.0

    def reset_episode(self) -> None:
        self._ep_dmg_dealt = 0.0
        self._ep_dmg_taken = 0.0
        self._ep_stocks_taken = 0
        self._ep_stocks_lost = 0
        self._ep_frames = 0
        self._prev_opp_dmg = 0.0
        self._prev_self_dmg = 0.0
        self._prev_opp_stocks = 4
        self._prev_self_stocks = 4

    def update_frame(self, obs: SmashObservation) -> None:
        self._ep_frames += 1
        dmg_dealt = max(0.0, obs.opponent_damage - self._prev_opp_dmg)
        dmg_taken = max(0.0, obs.player_damage - self._prev_self_dmg)
        stk_taken = max(0, self._prev_opp_stocks - obs.opponent_stocks)
        stk_lost = max(0, self._prev_self_stocks - obs.player_stocks)

        self._ep_dmg_dealt += dmg_dealt
        self._ep_dmg_taken += dmg_taken
        self._ep_stocks_taken += stk_taken
        self._ep_stocks_lost += stk_lost

        self._prev_opp_dmg = obs.opponent_damage
        self._prev_self_dmg = obs.player_damage
        self._prev_opp_stocks = obs.opponent_stocks
        self._prev_self_stocks = obs.player_stocks

    def end_episode(
        self,
        ep_reward: float,
        winner: Optional[int],
        recovery_success: Optional[bool] = None,
    ) -> None:
        self.episode_count += 1
        self._recent_ep_rewards.append(ep_reward)
        self._recent_ep_dmg_dealt.append(self._ep_dmg_dealt)
        self._recent_ep_dmg_taken.append(self._ep_dmg_taken)
        self._recent_ep_stocks_taken.append(self._ep_stocks_taken)
        self._recent_ep_stocks_lost.append(self._ep_stocks_lost)
        self._recent_ep_lengths.append(self._ep_frames)
        self._recent_wins.append(1.0 if winner == 0 else 0.0)
        if recovery_success is not None:
            self._recent_recovery_success.append(1.0 if recovery_success else 0.0)

    def _avg(self, d: deque) -> float:
        return sum(d) / len(d) if d else 0.0

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def fps(self) -> float:
        e = self.elapsed()
        return self.total_frames / e if e > 0 else 0.0

    def eta_str(self) -> str:
        f = self.fps()
        if f <= 0:
            return "??:??"
        remaining = max(0, self.target_frames - self.total_frames)
        secs = remaining / f
        m, s = divmod(int(secs), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    def progress_pct(self) -> float:
        if self.target_frames <= 0:
            return 0.0
        return 100.0 * self.total_frames / self.target_frames

    def print_episode(self, obs: SmashObservation, ep_reward: float, winner: Optional[int]) -> None:
        result = "WIN " if winner == 0 else "LOSS" if winner == 1 else "TIME"
        e = self.elapsed()
        m, s = divmod(int(e), 60)

        sys.stdout.write(
            f"\r\033[K"
            f"[{m:02d}:{s:02d}] Ep {self.episode_count:3d} {result} | "
            f"reward={ep_reward:+7.2f} | "
            f"dmg {self._ep_dmg_dealt:5.0f}/{self._ep_dmg_taken:5.0f} | "
            f"stocks +{self._ep_stocks_taken}-{self._ep_stocks_lost} | "
            f"P1 {obs.player_damage:3.0f}% ({obs.player_stocks}stk) "
            f"P2 {obs.opponent_damage:3.0f}% ({obs.opponent_stocks}stk)\n"
        )
        sys.stdout.flush()

    def print_batch(self) -> None:
        avg_r = self._avg(self._recent_ep_rewards)
        avg_dmg = self._avg(self._recent_ep_dmg_dealt)
        avg_taken = self._avg(self._recent_ep_dmg_taken)
        avg_stk = self._avg(self._recent_ep_stocks_taken)
        wr = self._avg(self._recent_wins) * 100

        bar_len = 30
        filled = int(bar_len * self.progress_pct() / 100)
        bar = "=" * filled + ">" + " " * (bar_len - filled - 1)

        sys.stdout.write(
            f"\r\033[K"
            f"[{bar}] {self.progress_pct():5.1f}% | "
            f"{self.total_frames:,}/{self.target_frames:,} frames | "
            f"{self.fps():.0f} fps | ETA {self.eta_str()}\n"
            f"  Episodes: {self.episode_count} | "
            f"PPO: policy={self.last_policy_loss:.4f} "
            f"value={self.last_value_loss:.4f} "
            f"entropy={self.last_entropy:.2f} | "
            f"batch_reward={self.last_batch_reward:.4f}\n"
            f"  Avg(20ep): reward={avg_r:+.2f} | "
            f"dmg={avg_dmg:.0f}/{avg_taken:.0f} | "
            f"stocks={avg_stk:.1f} | "
            f"win_rate={wr:.0f}%\n"
        )
        sys.stdout.flush()

    def print_live_frame(self, obs: SmashObservation) -> None:
        pct = self.progress_pct()
        e = self.elapsed()
        m, s = divmod(int(e), 60)
        sys.stdout.write(
            f"\r\033[K"
            f"[{m:02d}:{s:02d}] {pct:5.1f}% | "
            f"frame {self.total_frames:,} | "
            f"P1 {obs.player_damage:3.0f}% ({obs.player_stocks}stk) "
            f"P2 {obs.opponent_damage:3.0f}% ({obs.opponent_stocks}stk) | "
            f"{self.fps():.0f} fps"
        )
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# PPO helpers
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lmbda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0
    nonterminal = 1.0 - dones.float()
    delta = rewards + gamma * next_values * nonterminal - values
    for t in reversed(range(len(rewards))):
        lastgaelam = delta[t] + gamma * lmbda * nonterminal[t] * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # Load model
    ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path):
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    log.info("Loading model from %s", ckpt_path)
    model = ActorCriticMLP(obs_dim=OBS_DIM).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()
    log.info("Model loaded: %d params", sum(p.numel() for p in model.parameters()))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if "optim_state_dict" in ckpt:
        optim.load_state_dict(ckpt["optim_state_dict"])

    # Reward calculator
    if args.agent == "puff":
        reward_calc = PuffReward(weights=PuffWeights.dolphin())
    else:
        reward_calc = CompetitiveMeleeReward()
    log.info("Reward shaping: %s", args.agent)

    # Connect to Dolphin
    log.info("Connecting to %s ...", args.server_url)
    client = EmulatorEnv(base_url=args.server_url)
    client.connect()
    log.info("Connected.")

    # Graceful exit
    save_requested = False
    def _signal_handler(signum, frame):
        nonlocal save_requested
        save_requested = True
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    total_frames = 0
    episode_count = 0
    batch_size = args.batch_size

    stats = TrainingStats(target_frames=args.total_frames)
    training_mode = getattr(args, "training_mode", "NORMAL")
    recovery_success_rate_threshold = 0.8
    # Normalized stage x bounds (server sends positions in [-1, 1]; stage ~ ±0.305)
    STAGE_NORM_X = 68.4 / 224.0  # ~0.305

    def _recovery_success(observation: SmashObservation) -> bool:
        """True if agent reached stage (on_ground and |x| within stage)."""
        return (
            getattr(observation, "player_on_ground", False)
            and abs(observation.player_x) <= STAGE_NORM_X
        )

    def _effective_training_mode() -> str:
        if training_mode != "RECOVERY":
            return "NORMAL"
        if len(stats._recent_recovery_success) < 5:
            return "RECOVERY"
        rate = sum(stats._recent_recovery_success) / len(stats._recent_recovery_success)
        return "NORMAL" if rate >= recovery_success_rate_threshold else "RECOVERY"

    print()
    print("=" * 70)
    print(f"  DOLPHIN PPO FINE-TUNING — {args.agent.upper()}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Batch: {batch_size} frames | PPO epochs: {args.ppo_epochs} | LR: {args.lr}")
    print(f"  Target: {args.total_frames:,} frames | Training mode: {training_mode}")
    print("=" * 70)
    print()

    try:
        while total_frames < args.total_frames and not save_requested:
            # ---- Collect one batch ----
            batch_obs = []
            batch_actions = []
            batch_log_probs = []
            batch_rewards = []
            batch_values = []
            batch_dones = []

            reward_calc.reset()
            effective_mode = _effective_training_mode()
            result = client.reset(training_mode=effective_mode)
            obs = result.observation
            ep_recovery_success = False  # set True if agent reaches stage this episode (RECOVERY only)

            batch_start = time.time()
            frames_in_batch = 0
            ep_reward = 0.0

            while frames_in_batch < batch_size:
                if save_requested:
                    break

                obs_vec = obs_to_vector(obs, player_idx=0)
                obs_t = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    logits_flat = model(obs_t)
                    # Also get value from critic
                    features = model.backbone(obs_t)
                    value = model.critic_head(features).squeeze(-1)

                # Sample action from each head
                actions = []
                log_prob_total = 0.0
                offset = 0
                for n in ACTION_NVEC:
                    logits = logits_flat[:, offset : offset + n]
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    log_prob_total = log_prob_total + dist.log_prob(a).item()
                    actions.append(a.item())
                    offset += n

                action_indices = np.array(actions, dtype=np.int64)
                smash_action = action_to_smash(action_indices)

                # Step in Dolphin
                result = client.step(smash_action)
                next_obs = result.observation
                done = result.done

                # Compute shaped reward
                me_proxy, opp_proxy = _make_proxies(next_obs)
                winner = None
                if done:
                    if next_obs.opponent_stocks <= 0:
                        winner = 0
                    elif next_obs.player_stocks <= 0:
                        winner = 1
                reward, reward_info = reward_calc.step(
                    me_proxy, opp_proxy, done=done, winner=winner, action=smash_action,
                )

                # Store transition
                batch_obs.append(obs_vec)
                batch_actions.append(action_indices)
                batch_log_probs.append(log_prob_total)
                batch_rewards.append(reward)
                batch_values.append(value.item())
                batch_dones.append(float(done))

                ep_reward += reward
                frames_in_batch += 1
                total_frames += 1
                stats.total_frames = total_frames
                stats.update_frame(next_obs)

                if training_mode == "RECOVERY" and _recovery_success(next_obs):
                    ep_recovery_success = True

                # Live frame counter (update every 60 frames = ~1 sec of game)
                if frames_in_batch % 60 == 0:
                    stats.print_live_frame(next_obs)

                if done:
                    episode_count += 1
                    stats.episode_count = episode_count
                    winner_id = None
                    if next_obs.opponent_stocks <= 0:
                        winner_id = 0
                    elif next_obs.player_stocks <= 0:
                        winner_id = 1
                    stats.end_episode(
                        ep_reward, winner_id,
                        recovery_success=ep_recovery_success if training_mode == "RECOVERY" else None,
                    )
                    stats.print_episode(next_obs, ep_reward, winner_id)
                    stats.reset_episode()

                    reward_calc.reset()
                    effective_mode = _effective_training_mode()
                    result = client.reset(training_mode=effective_mode)
                    next_obs = result.observation
                    ep_reward = 0.0
                    ep_recovery_success = False

                obs = next_obs

            if save_requested or len(batch_obs) < 2:
                break

            # ---- PPO Update ----
            b_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=device)
            b_actions = torch.tensor(np.array(batch_actions), dtype=torch.long, device=device)
            b_old_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32, device=device)
            b_rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            b_values = torch.tensor(batch_values, dtype=torch.float32, device=device)
            b_dones = torch.tensor(batch_dones, dtype=torch.float32, device=device)

            with torch.no_grad():
                features = model.backbone(b_obs)
                b_next_values = model.critic_head(features).squeeze(-1)

            advantages, returns = compute_gae(
                b_rewards, b_values, b_next_values, b_dones,
                gamma=args.gamma, lmbda=args.lmbda,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            batch_policy_loss = 0.0
            batch_value_loss = 0.0
            batch_entropy = 0.0
            n_updates = 0

            for _ in range(args.ppo_epochs):
                indices = torch.randperm(len(b_obs), device=device)
                for start in range(0, len(b_obs), args.mini_batch_size):
                    end = min(start + args.mini_batch_size, len(b_obs))
                    idx = indices[start:end]

                    obs_mb = b_obs[idx]
                    act_mb = b_actions[idx]
                    old_lp_mb = b_old_log_probs[idx]
                    adv_mb = advantages[idx]
                    ret_mb = returns[idx]

                    logits_flat = model(obs_mb)
                    features = model.backbone(obs_mb)
                    value = model.critic_head(features).squeeze(-1)

                    log_prob = torch.zeros(len(idx), device=device)
                    entropy = torch.zeros(len(idx), device=device)
                    offset = 0
                    for j, n in enumerate(ACTION_NVEC):
                        logits = logits_flat[:, offset : offset + n]
                        dist = torch.distributions.Categorical(logits=logits)
                        a = act_mb[:, j].long().clamp(0, n - 1)
                        log_prob = log_prob + dist.log_prob(a)
                        entropy = entropy + dist.entropy()
                        offset += n

                    ratio = torch.exp(log_prob - old_lp_mb)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(
                        ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon
                    ) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.functional.mse_loss(value, ret_mb)
                    ent_loss = -entropy.mean()

                    # Linear entropy decay: start high (exploration), settle to purposeful movement after decay_frames
                    decay_frames = getattr(args, "entropy_decay_frames", 500_000)
                    entropy_min = getattr(args, "entropy_coef_min", 0.001)
                    if total_frames >= decay_frames:
                        entropy_coef = entropy_min
                    else:
                        t = total_frames / max(1, decay_frames)
                        entropy_coef = args.entropy_coef + (entropy_min - args.entropy_coef) * t
                    loss = policy_loss + 0.5 * value_loss + entropy_coef * ent_loss

                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optim.step()

                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_entropy += entropy.mean().item()
                    n_updates += 1

            stats.last_policy_loss = batch_policy_loss / max(n_updates, 1)
            stats.last_value_loss = batch_value_loss / max(n_updates, 1)
            stats.last_entropy = batch_entropy / max(n_updates, 1)
            stats.last_batch_reward = b_rewards.mean().item()
            print()
            stats.print_batch()

            # Periodic checkpoint
            if total_frames % args.checkpoint_interval < batch_size:
                _save(model, optim, total_frames, final=False)

    except Exception:
        log.exception("Training error")
    finally:
        _save(model, optim, total_frames, final=True)
        client.close()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        e = stats.elapsed()
        m, s = divmod(int(e), 60)
        h, m2 = divmod(m, 60)
        print()
        print("=" * 70)
        print(f"  TRAINING COMPLETE")
        print(f"  Frames: {total_frames:,} | Episodes: {episode_count}")
        print(f"  Wall time: {h}h {m2:02d}m {s:02d}s | Avg fps: {stats.fps():.0f}")
        if stats._recent_ep_rewards:
            print(f"  Last 20 ep avg reward: {stats._avg(stats._recent_ep_rewards):+.2f}")
            print(f"  Last 20 ep win rate:   {stats._avg(stats._recent_wins) * 100:.0f}%")
        print("=" * 70)


def _save(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    total_frames: int,
    final: bool = False,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    name = "dolphin_puff_final.pt" if final else f"dolphin_puff_{total_frames}.pt"
    path = CHECKPOINT_DIR / name
    torch.save({
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "total_frames": total_frames,
    }, path)
    log.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO fine-tuning against real Melee via Dolphin OpenEnv server",
    )
    parser.add_argument(
        "--agent", type=str, default="puff", choices=("puff", "mango"),
        help="Which reward shaping to use",
    )
    parser.add_argument(
        "--training-mode", type=str, default="NORMAL", choices=("NORMAL", "RECOVERY"),
        help="NORMAL or RECOVERY. RECOVERY: spawn P1 off-stage each reset; switch to NORMAL when recovery success rate > 80%%",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="../checkpoints/puff_final.pt",
        help="Path to sim-trained .pt checkpoint to fine-tune",
    )
    parser.add_argument(
        "--server-url", type=str, default="http://localhost:8000",
    )
    parser.add_argument("--total-frames", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Initial entropy coefficient (decays linearly over --entropy-decay-frames)")
    parser.add_argument("--entropy-coef-min", type=float, default=0.001, help="Minimum entropy coefficient after decay")
    parser.add_argument("--entropy-decay-frames", type=int, default=500_000, help="Frames over which entropy_coef decays to entropy_coef_min")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    train(args)


if __name__ == "__main__":
    main()
