#!/usr/bin/env python3
"""Puff-style PPO trainer for Melee sim (TorchRL).

Defensive reward shaping to encourage patient play and Rest kills:
- Approach shaping, proximity bonus, damage amplification,
  Rest kill bonus, missed Rest penalty.

Usage:
    # Train vs dummy
    python3 train.py --total-frames 500000

    # Train vs Mango
    python3 train.py --opponent mango --total-frames 2000000

    # Evaluate
    python3 train.py --eval
    python3 train.py --eval-vs mango

    # Sanity / stability checks
    python3 train.py --sanity
    python3 train.py --stability
"""

from __future__ import annotations

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("PYOPENGL_PLATFORM", "")
os.environ.setdefault("RL_WARNINGS", "0")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchrl.envs.libs.gym")

import argparse
import math
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from envs.melee_sim_env import MeleeSimEnv, OBS_DIM, _build_obs, _decode_action
from opponents import load_opponent
from physics.constants import Action, STAGE
from physics.state import GameState

# Reuse the same network architecture as Mango
from mango_trainer import (
    ActorCriticMLP, ACTION_NVEC, NUM_ACTIONS, ACTION_FLAT,
    encode_flat, decode_flat, _ActionConverterWrapper,
)


# ---------------------------------------------------------------------------
# PuffRewardWrapper
# ---------------------------------------------------------------------------

# Sweet-spot spacing: Puff orbits the opponent at this distance
SPACING_IDEAL = 10.0   # units — just inside jab/bair range
SPACING_INNER = 5.0    # too close (Rest range — reserved for Rest)
SPACING_OUTER = 20.0   # too far — no spacing bonus
SPACING_BONUS = 0.003  # per frame in the sweet spot
SPACING_COEF = 0.005   # gradient toward sweet spot when outside it

DAMAGE_BONUS = 0.04    # on top of base env's 0.01 -> total 0.05
REST_KILL_BONUS = 0.5
REST_MISS_PENALTY = 0.002  # per frame of sleep after a whiff

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_INTERVAL_FRAMES = 1_000_000


class PuffRewardWrapper(gym.Wrapper):
    """Wraps MeleeSimEnv with Puff-style weave-in/out reward shaping.

    Puff is incentivized to orbit the opponent at jab/bair spacing (~10 units),
    darting in to attack then backing out — rather than always running toward
    them. Rest is still rewarded heavily when it kills.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_distance: float = 20.0
        self._prev_opp_percent: float = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_distance = 20.0
        self._prev_opp_percent = 0.0
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = self._compute_puff_bonus()
        return obs, reward + bonus, terminated, truncated, info

    def _compute_puff_bonus(self) -> float:
        bonus = 0.0
        state = getattr(self.env, "_state", None)
        if state is None:
            return bonus

        me = state.players[0]
        opp = state.players[1]

        distance_now = abs(me.x - opp.x) + abs(me.y - opp.y)

        # Spacing reward: bonus for orbiting at the sweet-spot distance
        if SPACING_INNER <= distance_now <= SPACING_OUTER:
            # Gaussian-shaped bonus peaked at SPACING_IDEAL
            deviation = distance_now - SPACING_IDEAL
            spacing_reward = SPACING_BONUS * math.exp(-0.5 * (deviation / 5.0) ** 2)
            bonus += spacing_reward
        elif distance_now > SPACING_OUTER:
            # Too far — gradient toward sweet spot (like old approach reward)
            distance_change = self._prev_distance - distance_now
            bonus += distance_change * SPACING_COEF

        self._prev_distance = distance_now

        # Extra damage amplification (base env gives 0.01, we add 0.04 -> 0.05 total)
        damage_dealt = opp.percent - self._prev_opp_percent
        if damage_dealt > 0:
            bonus += damage_dealt * DAMAGE_BONUS
        self._prev_opp_percent = opp.percent

        # Rest kill bonus (only fires the frame a stock is taken via Rest)
        if getattr(me, "_current_move_name", "") == "rest":
            if me.attack_connected and me.action == Action.REST_SLEEP:
                bonus += REST_KILL_BONUS

        # Missed Rest penalty (per frame of sleep after whiff)
        if (me.action == Action.REST_SLEEP
                and not me.attack_connected):
            bonus -= REST_MISS_PENALTY

        return bonus


# ---------------------------------------------------------------------------
# Env creation
# ---------------------------------------------------------------------------

def _create_env(max_frames: int = 3600, opponent_fn=None) -> gym.Env:
    base = MeleeSimEnv(opponent_fn=opponent_fn, max_frames=max_frames)
    return PuffRewardWrapper(base)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

_train_state: Optional[Dict[str, Any]] = None


def _save_checkpoint(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    total_frames: int,
    scaler=None,
    *,
    final: bool = False,
) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / "puff_final.pt" if final else CHECKPOINT_DIR / f"puff_ppo_{total_frames}.pt"
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "total_frames": total_frames,
    }
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    torch.save(ckpt, path)
    return path


def _load_latest_checkpoint() -> Optional[Dict[str, Any]]:
    if not CHECKPOINT_DIR.exists():
        return None
    pts = list(CHECKPOINT_DIR.glob("puff_ppo_*.pt"))
    if not pts:
        return None
    latest = max(pts, key=lambda p: int(p.stem.split("_")[-1].replace(".pt", "")))
    return torch.load(latest, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_ppo(args: argparse.Namespace) -> None:
    global _train_state
    try:
        from torchrl.collectors import SyncDataCollector
        from torchrl.envs import ParallelEnv
        from torchrl.envs.libs.gym import GymWrapper
        from torchrl.data.replay_buffers import ReplayBuffer
        from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
        from torchrl.data.replay_buffers.storages import LazyTensorStorage
    except ImportError as e:
        raise ImportError(
            "TorchRL required for training. Install: pip install torch torchrl"
        ) from e

    device_str = getattr(args, "device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
        print("WARNING: CUDA not available, falling back to CPU")
    device = torch.device(device_str)
    use_amp = device.type == "cuda" and args.use_amp
    if use_amp:
        print("AMP enabled for Tensor Core throughput")

    opponent_fn = None
    opponent_name = getattr(args, "opponent", None)
    if opponent_name:
        opponent_fn = load_opponent(opponent_name)
        print(f"Cross-play: training vs {opponent_name}")

    num_workers = args.num_workers or max(1, (os.cpu_count() or 4) - 1)
    if opponent_fn:
        num_workers = 1
        print("Using 1 worker (cross-play opponent must stay in-process)")
    print(f"Using {num_workers} parallel env workers")

    def make_env():
        return GymWrapper(
            _ActionConverterWrapper(_create_env(
                max_frames=args.max_frames, opponent_fn=opponent_fn
            )),
            device="cpu",
        )

    base_env = ParallelEnv(
        num_workers,
        make_env,
        serial_for_single=True,
    )

    model = ActorCriticMLP(
        obs_dim=OBS_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_frames = 0
    if args.resume:
        ckpt = _load_latest_checkpoint()
        if ckpt is None:
            final = CHECKPOINT_DIR / "puff_final.pt"
            if final.exists():
                ckpt = torch.load(final, map_location="cpu", weights_only=False)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optim_state_dict" in ckpt:
                optim.load_state_dict(ckpt["optim_state_dict"])
            start_frames = ckpt.get("total_frames", 0)
            if scaler and "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            print(f"Resumed from checkpoint: {start_frames} frames")
        else:
            print("No checkpoint found, starting fresh")

    def policy_fn(td):
        with torch.no_grad():
            obs = td["observation"].float()
            action_6, log_prob = model.get_action_and_log_prob(obs)
            _, value = model(obs)
        td["action"] = encode_flat(action_6)
        td["sample_log_prob"] = log_prob
        td["state_value"] = value
        return td

    remaining_frames = max(0, args.total_frames - start_frames)
    collector = SyncDataCollector(
        base_env,
        policy_fn,
        frames_per_batch=args.frames_per_batch,
        total_frames=remaining_frames,
        device=device,
        split_trajs=False,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    _train_state = {
        "model": model,
        "optim": optim,
        "scaler": scaler,
        "total_frames": start_frames,
        "save_requested": False,
    }

    def _signal_handler(signum, frame):
        _train_state["save_requested"] = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.use_wandb:
        try:
            import wandb
            wandb.init(project="puff-melee-ppo", config=vars(args))
        except ImportError:
            print("WARNING: wandb not installed, --use-wandb ignored")
            args.use_wandb = False

    def compute_gae(tensordict, values, values_next, dones):
        rewards = tensordict["next", "reward"].squeeze(-1)
        next_nonterminal = 1.0 - dones.float()
        delta = rewards + args.gamma * values_next * next_nonterminal - values
        advantages = torch.zeros_like(delta, device=delta.device)
        lastgaelam = 0
        for t in reversed(range(delta.shape[0])):
            lastgaelam = (
                delta[t]
                + args.gamma * args.lmbda * lastgaelam * next_nonterminal[t]
            )
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x

    mode = f"CROSS-PLAY vs {opponent_name}" if opponent_name else "DUMMY"
    print(f"\nPuff PPO — {remaining_frames:,} frames | {num_workers} workers | mode: {mode}")
    print(f"Episodes: {args.max_frames} frames ({args.max_frames / 60:.0f}s)")
    print(f"Entropy coef: {args.entropy_coef}")
    print(f"Checkpoints: {CHECKPOINT_DIR}/")
    print()

    pbar = tqdm(total=args.total_frames, initial=start_frames, desc="Puff PPO")

    for i, batch in enumerate(collector):
        if _train_state["save_requested"]:
            tf = _train_state["total_frames"]
            path = _save_checkpoint(model, optim, tf, scaler)
            final_path = _save_checkpoint(model, optim, tf, scaler, final=True)
            print(f"\nGraceful exit: saved to {path} and {final_path}")
            break

        batch = batch.to(device)
        obs = batch["observation"].float()
        obs_next = batch["next", "observation"].float()
        with torch.no_grad():
            _, values = model(obs)
            _, values_next = model(obs_next)
            values = values.squeeze(-1)
            values_next = values_next.squeeze(-1)
            dones = batch["next", "done"].squeeze(-1)
        advantages, returns = compute_gae(batch, values, values_next, dones)
        batch["advantage"] = advantages
        batch["value_target"] = returns

        batch_policy_loss = 0.0
        batch_value_loss = 0.0
        batch_entropy = 0.0
        step_count = 0

        for _ in range(args.ppo_epochs):
            data_view = batch.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(args.frames_per_batch // args.sub_batch_size):
                sub = replay_buffer.sample(args.sub_batch_size).to(device)
                obs_sub = sub["observation"].float()
                action_sub = decode_flat(sub["action"])  # (batch, 6) from flat Discrete(320)
                old_log_prob = sub["sample_log_prob"]
                adv = sub["advantage"]
                ret = sub["value_target"]

                optim.zero_grad()
                logits_flat, value = model(obs_sub)
                offset = 0
                log_prob = 0.0
                for j, n in enumerate(ACTION_NVEC):
                    logits = logits_flat[:, offset : offset + n]
                    dist = torch.distributions.Categorical(logits=logits)
                    a = action_sub[..., j].long().clamp(0, n - 1)
                    log_prob = log_prob + dist.log_prob(a)
                    offset += n
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value.squeeze(-1), ret)
                entropy = 0.0
                offset = 0
                for n in ACTION_NVEC:
                    logits = logits_flat[:, offset : offset + n]
                    dist = torch.distributions.Categorical(logits=logits)
                    entropy = entropy + dist.entropy().mean()
                    offset += n
                loss = policy_loss + 0.5 * value_loss - args.entropy_coef * entropy
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()

                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
                batch_entropy += entropy.item()
                step_count += 1

        _train_state["total_frames"] += batch.numel()
        total_frames = _train_state["total_frames"]
        mean_reward = batch["next", "reward"].mean().item()
        mean_policy_loss = batch_policy_loss / max(step_count, 1)
        mean_value_loss = batch_value_loss / max(step_count, 1)
        mean_entropy = batch_entropy / max(step_count, 1)

        pbar.update(batch.numel())
        pbar.set_postfix(reward=f"{mean_reward:.4f}")

        if total_frames > 0 and total_frames % args.checkpoint_interval == 0:
            path = _save_checkpoint(model, optim, total_frames, scaler)
            print(f"\nCheckpoint saved: {path}")

        if args.use_wandb:
            import wandb
            wandb.log({
                "reward": mean_reward,
                "policy_loss": mean_policy_loss,
                "value_loss": mean_value_loss,
                "entropy": mean_entropy,
                "total_frames": total_frames,
            })

    pbar.close()
    total_frames = _train_state["total_frames"]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    final_path = _save_checkpoint(model, optim, total_frames, scaler, final=True)
    print(f"Saved final model: {final_path}")
    _train_state = None
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    print("Training complete.")


# ---------------------------------------------------------------------------
# Sanity / stability checks
# ---------------------------------------------------------------------------

def _run_sanity_rollout(max_steps: int = 2000) -> None:
    env = _create_env(max_frames=max_steps)
    obs, _ = env.reset(seed=42)
    total_reward = 0.0
    steps = 0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    print(f"Sanity rollout: steps={steps}, total_reward={total_reward:.4f}")
    print(f"  Final info: {info}")


def _run_stability_test() -> None:
    try:
        from torchrl.collectors import SyncDataCollector
        from torchrl.envs.libs.gym import GymWrapper
        from torchrl.data.replay_buffers import ReplayBuffer
        from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
        from torchrl.data.replay_buffers.storages import LazyTensorStorage
    except ImportError as e:
        raise ImportError("TorchRL required. Install: pip install torch torchrl") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("STABILITY TEST (Puff trainer)")
    print("  Device:", device)
    print("=" * 60)

    frames_per_batch = 512
    total_frames = 10_000
    ppo_epochs = 4
    sub_batch_size = 64

    base_env = GymWrapper(
        _ActionConverterWrapper(_create_env(max_frames=3000)),
        device=device,
    )

    model = ActorCriticMLP(obs_dim=OBS_DIM, hidden_dim=128, num_layers=2).to(device)

    def policy_fn(td):
        with torch.no_grad():
            obs = td["observation"].float()
            action_6, log_prob = model.get_action_and_log_prob(obs)
            _, value = model(obs)
        td["action"] = encode_flat(action_6)
        td["sample_log_prob"] = log_prob
        td["state_value"] = value
        return td

    collector = SyncDataCollector(
        base_env, policy_fn,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device, split_trajs=False,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    gamma, lmbda, clip_epsilon, entropy_coef = 0.99, 0.95, 0.2, 0.01

    def compute_gae(tensordict, values, values_next, dones):
        rewards = tensordict["next", "reward"].squeeze(-1)
        next_nonterminal = 1.0 - dones.float()
        delta = rewards + gamma * values_next * next_nonterminal - values
        advantages = torch.zeros_like(delta, device=delta.device)
        lastgaelam = 0
        for t in reversed(range(delta.shape[0])):
            lastgaelam = delta[t] + gamma * lmbda * lastgaelam * next_nonterminal[t]
            advantages[t] = lastgaelam
        return advantages, advantages + values

    batch_count = 0
    for batch in collector:
        batch = batch.to(device)
        obs = batch["observation"].float()
        obs_next = batch["next", "observation"].float()
        with torch.no_grad():
            _, values = model(obs)
            _, values_next = model(obs_next)
            values = values.squeeze(-1)
            values_next = values_next.squeeze(-1)
            dones = batch["next", "done"].squeeze(-1)
        advantages, returns = compute_gae(batch, values, values_next, dones)
        batch["advantage"] = advantages
        batch["value_target"] = returns

        total_loss = 0.0
        n = 0
        for _ in range(ppo_epochs):
            data_view = batch.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                sub = replay_buffer.sample(sub_batch_size).to(device)
                action_sub = decode_flat(sub["action"])  # (batch, 6) from flat Discrete(320)
                logits_flat, value = model(sub["observation"].float())
                offset = 0
                log_prob = 0.0
                for j, nv in enumerate(ACTION_NVEC):
                    logits = logits_flat[:, offset : offset + nv]
                    dist = torch.distributions.Categorical(logits=logits)
                    a = action_sub[..., j].long().clamp(0, nv - 1)
                    log_prob = log_prob + dist.log_prob(a)
                    offset += nv
                ratio = torch.exp(log_prob - sub["sample_log_prob"])
                adv = sub["advantage"]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value.squeeze(-1), sub["value_target"])
                ent = 0.0
                offset = 0
                for nv in ACTION_NVEC:
                    logits = logits_flat[:, offset : offset + nv]
                    dist = torch.distributions.Categorical(logits=logits)
                    ent = ent + dist.entropy().mean()
                    offset += nv
                loss = policy_loss + 0.5 * value_loss - entropy_coef * ent

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"\n!!! NaN/Inf at batch {batch_count} !!!")
                    sys.exit(1)

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                total_loss += loss.item()
                n += 1

        mean_reward = batch["next", "reward"].mean().item()
        print(f"batch {batch_count:3d} | loss={total_loss / max(n, 1):.4f} reward={mean_reward:.4f}")
        sys.stdout.flush()
        batch_count += 1

    print("=" * 60)
    print(f"STABILITY TEST PASSED ({batch_count} batches, {total_frames} frames)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(opponent_name: Optional[str], num_episodes: int = 5) -> None:
    ckpt_path = CHECKPOINT_DIR / "puff_final.pt"
    if not ckpt_path.exists():
        ckpt = _load_latest_checkpoint()
        if ckpt is None:
            print("No Puff checkpoint found. Train first.")
            return
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = ActorCriticMLP(obs_dim=OBS_DIM, hidden_dim=256, num_layers=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    opp_fn = None
    if opponent_name:
        opp_fn = load_opponent(opponent_name)
    env = MeleeSimEnv(opponent_fn=opp_fn, max_frames=3600)

    label = f"Puff vs {opponent_name}" if opponent_name else "Puff vs dummy"
    print(f"Evaluating: {label}\n")

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0

        for step in range(3600):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                logits_flat, _ = model(obs_t)
            actions = []
            offset = 0
            for n in ACTION_NVEC:
                logits = logits_flat[:, offset : offset + n]
                a = logits.argmax(dim=-1)
                actions.append(a.item())
                offset += n
            action_6 = torch.tensor([actions], dtype=torch.long)
            action = int(encode_flat(action_6).item())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        metrics = info.get("episode_metrics", {})
        winner = info.get("winner", None)
        result = "WIN" if winner == 0 else "LOSS" if winner == 1 else "TIME"

        print(f"  Episode {ep}: {result} | reward={total_reward:+.3f} "
              f"| stocks {info['p0_stocks']}v{info['p1_stocks']} "
              f"| dmg_dealt={metrics.get('damage_dealt', 0):.0f} "
              f"dmg_taken={metrics.get('damage_taken', 0):.0f} "
              f"| rest: {metrics.get('rest_hits', 0)}/{metrics.get('rest_attempts', 0)} hits "
              f"({metrics.get('rest_kills', 0)} kills)")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Puff-style Melee PPO trainer (TorchRL)")
    parser.add_argument("--sanity", action="store_true",
                        help="Run one random rollout and print reward")
    parser.add_argument("--stability", action="store_true",
                        help="Mini-training: print loss curve, check NaN")
    parser.add_argument("--eval", action="store_true", help="Evaluate vs dummy")
    parser.add_argument("--eval-vs", type=str, default=None, metavar="OPPONENT",
                        help="Evaluate vs named opponent (e.g. 'mango', or .pt/.zip path)")
    parser.add_argument("--opponent", type=str, default=None, metavar="OPPONENT",
                        help="Train vs named opponent (e.g. 'mango', or .pt/.zip path)")
    parser.add_argument("--max-frames", type=int, default=3600)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Parallel env workers (default: cpu_count - 1)")
    parser.add_argument("--frames-per-batch", type=int, default=8192)
    parser.add_argument("--total-frames", type=int, default=500_000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--sub-batch-size", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL_FRAMES)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Load latest Puff checkpoint and continue training")
    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()
    args.use_amp = args.use_amp and not args.no_amp

    if args.sanity:
        _run_sanity_rollout()
    elif args.stability:
        _run_stability_test()
    elif args.eval:
        _evaluate(opponent_name=None, num_episodes=args.eval_episodes)
    elif args.eval_vs:
        _evaluate(opponent_name=args.eval_vs, num_episodes=args.eval_episodes)
    else:
        _train_ppo(args)


if __name__ == "__main__":
    main()
