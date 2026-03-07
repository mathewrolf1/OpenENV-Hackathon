#!/usr/bin/env python3
"""
Mang0-style PPO trainer for Melee sim.

Heuristic reward shaping to encourage aggressive, high-risk play:
- Proximity bonus, shield pressure, movement incentive, high-risk recovery.
"""

from __future__ import annotations

# Headless mode: prevent GUI/OpenGL windows on SSH (must run before any display libs)
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("PYOPENGL_PLATFORM", "")
os.environ.setdefault("RL_WARNINGS", "0")  # Suppress TorchRL deprecation spam

# Suppress noisy warnings for headless deployment
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

from envs.melee_sim_env import MeleeSimEnv, OBS_DIM
from physics.constants import Action, STAGE
from physics.state import CharacterState, GameState, Stage


# ---------------------------------------------------------------------------
# MangoRewardWrapper
# ---------------------------------------------------------------------------

PROXIMITY_DISTANCE = 30.0
PROXIMITY_BONUS = 0.05
SHIELD_PRESSURE_BONUS = 2.0
MOVEMENT_BONUS = 0.02
RECOVERY_BONUS = 1.0
DISADVANTAGE_Y_THRESHOLD = 5.0
DISADVANTAGE_X_THRESHOLD = 55.0

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_INTERVAL_FRAMES = 1_000_000


class MangoRewardWrapper(gym.Wrapper):
    """Wraps MeleeSimEnv with Mang0-style heuristic reward shaping."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_p1_percent: float = 0.0
        self._was_in_disadvantage: bool = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_p1_percent = 0.0
        self._was_in_disadvantage = False
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = self._compute_mango_bonus(terminated, truncated)
        return obs, reward + bonus, terminated, truncated, info

    def _is_disadvantage(self, gs: GameState) -> bool:
        """Agent in disadvantage: low/far off-stage or in hitstun."""
        me = gs.players[0]
        if me.action == Action.DEAD or me.action == Action.RESPAWN_INVULN:
            return False
        low_or_far = (
            (not me.on_ground and me.y < DISADVANTAGE_Y_THRESHOLD)
            or abs(me.x) > DISADVANTAGE_X_THRESHOLD
        )
        in_hitstun = me.hitstun_frames_left > 0
        return low_or_far or in_hitstun

    def _returned_to_stage(self, gs: GameState) -> bool:
        """Agent just returned to stage from disadvantage."""
        me = gs.players[0]
        on_stage = me.on_ground and me.y >= 0 and abs(me.x) <= STAGE["right_edge"]
        return self._was_in_disadvantage and on_stage and me.hitstun_frames_left <= 0

    def _compute_mango_bonus(self, terminated: bool, truncated: bool) -> float:
        bonus = 0.0
        state = getattr(self.env, "_state", None)
        if state is None:
            return bonus

        me = state.players[0]
        opp = state.players[1]

        # Proximity: +0.05 per frame when close to opponent
        dx = me.x - opp.x
        dy = me.y - opp.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= PROXIMITY_DISTANCE:
            bonus += PROXIMITY_BONUS

        # Shield pressure: +2.0 when dealing damage (sim has no shields; proxy)
        damage_dealt = opp.percent - self._prev_p1_percent
        if damage_dealt > 0:
            bonus += SHIELD_PRESSURE_BONUS
        self._prev_p1_percent = opp.percent

        # Movement incentive: small reward for RUN or AIRBORNE (anti-camping)
        if me.action == Action.RUN or me.action == Action.AIRBORNE:
            bonus += MOVEMENT_BONUS

        # High-risk recovery: bonus for returning to stage from disadvantage
        if self._returned_to_stage(state):
            bonus += RECOVERY_BONUS

        self._was_in_disadvantage = self._is_disadvantage(state)
        return bonus


# ---------------------------------------------------------------------------
# Policy architecture (MLP for MultiDiscrete)
# ---------------------------------------------------------------------------

ACTION_NVEC = [5, 4, 2, 2, 2, 2]
NUM_ACTIONS = sum(ACTION_NVEC)


class ActorCriticMLP(nn.Module):
    """MLP actor-critic for MultiDiscrete action space."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        action_nvec: Tuple[int, ...] = tuple(ACTION_NVEC),
    ):
        super().__init__()
        self.action_nvec = action_nvec
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        logits_flat = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits_flat, value

    def get_action_and_log_prob(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits_flat, _ = self.forward(obs)
        actions = []
        log_probs = []
        offset = 0
        for n in self.action_nvec:
            logits = logits_flat[:, offset : offset + n]
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            lp = dist.log_prob(a)
            actions.append(a)
            log_probs.append(lp)
            offset += n
        action = torch.stack(actions, dim=-1)
        log_prob = sum(log_probs)
        return action, log_prob


# ---------------------------------------------------------------------------
# TorchRL integration
# ---------------------------------------------------------------------------

def _create_env(max_frames: int = 5000) -> gym.Env:
    base = MeleeSimEnv(max_frames=max_frames)
    return MangoRewardWrapper(base)


class _ActionConverterWrapper(gym.Wrapper):
    """Ensures action is np.ndarray shape (6,) for MultiDiscrete (TorchRL compat)."""

    def step(self, action):
        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        action = np.asarray(action, dtype=np.int64)
        if action.ndim > 1:
            action = action.squeeze()
        if action.ndim == 0:
            # Flattened index from TorchRL: decode [5,4,2,2,2,2]
            idx = int(action)
            decoded = []
            for n in reversed(ACTION_NVEC):
                decoded.append(idx % n)
                idx //= n
            action = np.array(decoded[::-1], dtype=np.int64)
        return self.env.step(action)


def _run_sanity_rollout(max_steps: int = 2000) -> None:
    """Run one rollout with random actions; print total_reward for sanity check."""
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


# Global for signal handler (set by _train_ppo)
_train_state: Optional[Dict[str, Any]] = None


def _save_checkpoint(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    total_frames: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    *,
    final: bool = False,
) -> Path:
    """Save model, optimizer, and optionally scaler to checkpoints/."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / "mango_final.pt" if final else CHECKPOINT_DIR / f"mango_ppo_{total_frames}.pt"
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
    """Load latest checkpoint from checkpoints/ if any exist."""
    if not CHECKPOINT_DIR.exists():
        return None
    pts = list(CHECKPOINT_DIR.glob("mango_ppo_*.pt"))
    if not pts:
        return None
    latest = max(pts, key=lambda p: int(p.stem.split("_")[-1].replace(".pt", "")))
    return torch.load(latest, map_location="cpu", weights_only=False)


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
        print("AMP enabled for H100 Tensor Core throughput")

    num_workers = args.num_workers or max(1, (os.cpu_count() or 4) - 1)
    print(f"Using {num_workers} parallel env workers")

    def make_env():
        return GymWrapper(
            _ActionConverterWrapper(_create_env(max_frames=args.max_frames)),
            device="cpu",
        )

    base_env = ParallelEnv(
        num_workers,
        make_env,
        serial_for_single=True,
    )

    obs_dim = OBS_DIM
    model = ActorCriticMLP(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_frames = 0
    if args.resume:
        ckpt = _load_latest_checkpoint()
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
            action, log_prob = model.get_action_and_log_prob(obs)
            _, value = model(obs)
        td["action"] = action
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
            wandb.init(project="mango-melee-ppo", config=vars(args))
        except ImportError:
            print("WARNING: wandb not installed, --use_wandb ignored")
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

    pbar = tqdm(total=args.total_frames, initial=start_frames, desc="PPO")

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
                action_sub = sub["action"]
                old_log_prob = sub["sample_log_prob"]
                adv = sub["advantage"]
                ret = sub["value_target"]

                optim.zero_grad()
                if use_amp and scaler is not None:
                    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
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
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optim)
                    scaler.update()
                else:
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
            log_dict = {
                "reward": mean_reward,
                "policy_loss": mean_policy_loss,
                "value_loss": mean_value_loss,
                "entropy": mean_entropy,
                "total_frames": total_frames,
            }
            if device.type == "cuda":
                log_dict["gpu_memory_allocated_mb"] = (
                    torch.cuda.memory_allocated() / 1024 / 1024
                )
                log_dict["gpu_memory_reserved_mb"] = (
                    torch.cuda.memory_reserved() / 1024 / 1024
                )
            wandb.log(log_dict)

    pbar.close()
    total_frames = _train_state["total_frames"]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    final_path = _save_checkpoint(model, optim, total_frames, scaler, final=True)
    print(f"Saved final model: {final_path}")
    _train_state = None
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    print("Training complete.")


def _run_stability_test() -> None:
    """
    Mini-training session for local debugging: detect memory leaks, NaN gradients.
    Prints loss curve to terminal. Watch Mac RAM in Activity Monitor.
    """
    try:
        from torchrl.collectors import SyncDataCollector
        from torchrl.envs import ParallelEnv
        from torchrl.envs.libs.gym import GymWrapper
        from torchrl.data.replay_buffers import ReplayBuffer
        from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
        from torchrl.data.replay_buffers.storages import LazyTensorStorage
    except ImportError as e:
        raise ImportError("TorchRL required. Install: pip install torch torchrl") from e

    try:
        import psutil
    except ImportError:
        psutil = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("STABILITY TEST (Local Debug)")
    print("  Device:", device)
    print("  Watch RAM in Activity Monitor; loss curve below.")
    print("=" * 60)

    # Mac-friendly: single env (avoids multiprocessing/shared-memory on macOS)
    num_envs = 1
    frames_per_batch = 512
    total_frames = 10_000
    ppo_epochs = 4
    sub_batch_size = 64
    max_frames = 3000

    # Single env only (avoids ParallelEnv shared-memory on macOS)
    base_env = GymWrapper(
        _ActionConverterWrapper(_create_env(max_frames=max_frames)),
        device=device,
    )

    model = ActorCriticMLP(
        obs_dim=OBS_DIM,
        hidden_dim=128,
        num_layers=2,
    ).to(device)

    def policy_fn(td):
        with torch.no_grad():
            obs = td["observation"].float()
            action, log_prob = model.get_action_and_log_prob(obs)
            _, value = model(obs)
        td["action"] = action
        td["sample_log_prob"] = log_prob
        td["state_value"] = value
        return td

    collector = SyncDataCollector(
        base_env,
        policy_fn,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        split_trajs=False,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    gamma, lmbda, clip_epsilon, entropy_coef = 0.99, 0.95, 0.2, 1e-4

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

    def _get_ram_mb() -> Optional[float]:
        if psutil is not None:
            return psutil.Process().memory_info().rss / 1024 / 1024
        return None

    loss_history: List[Dict[str, float]] = []
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

        batch_policy_loss = 0.0
        batch_value_loss = 0.0
        batch_entropy = 0.0
        batch_loss = 0.0
        step_count = 0

        for _ in range(ppo_epochs):
            data_view = batch.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                sub = replay_buffer.sample(sub_batch_size).to(device)
                obs_sub = sub["observation"].float()
                action_sub = sub["action"]
                old_log_prob = sub["sample_log_prob"]
                adv = sub["advantage"]
                ret = sub["value_target"]

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
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value.squeeze(-1), ret)
                entropy = 0.0
                offset = 0
                for n in ACTION_NVEC:
                    logits = logits_flat[:, offset : offset + n]
                    dist = torch.distributions.Categorical(logits=logits)
                    entropy = entropy + dist.entropy().mean()
                    offset += n
                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

                # NaN check
                for name, t in [
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("loss", loss),
                ]:
                    if torch.isnan(t).any() or torch.isinf(t).any():
                        print(f"\n!!! NaN/Inf detected in {name} at batch {batch_count} !!!")
                        sys.exit(1)

                optim.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"\n!!! NaN/Inf in grad_norm at batch {batch_count} !!!")
                    sys.exit(1)
                optim.step()

                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
                batch_entropy += entropy.item()
                batch_loss += loss.item()
                step_count += 1

        mean_reward = batch["next", "reward"].mean().item()
        n = max(step_count, 1)
        rec = {
            "batch": batch_count,
            "policy_loss": batch_policy_loss / n,
            "value_loss": batch_value_loss / n,
            "entropy": batch_entropy / n,
            "loss": batch_loss / n,
            "reward": mean_reward,
        }
        if psutil is not None:
            rec["ram_mb"] = _get_ram_mb()
        loss_history.append(rec)

        ram_str = f" RAM={rec['ram_mb']:.0f}MB" if "ram_mb" in rec else ""
        print(
            f"batch {batch_count:3d} | loss={rec['loss']:.4f} "
            f"policy={rec['policy_loss']:.4f} value={rec['value_loss']:.4f} "
            f"entropy={rec['entropy']:.4f} reward={mean_reward:.4f}{ram_str}"
        )
        sys.stdout.flush()

        batch_count += 1

    print("=" * 60)
    print("STABILITY TEST PASSED")
    print(f"  Completed {batch_count} batches, {total_frames} frames")
    if loss_history:
        losses = [h["loss"] for h in loss_history]
        print(f"  Loss range: {min(losses):.4f} .. {max(losses):.4f}")
        if "ram_mb" in loss_history[0]:
            rams = [h["ram_mb"] for h in loss_history]
            print(f"  RAM range: {min(rams):.0f}MB .. {max(rams):.0f}MB")
    print("=" * 60)


def test_reward_logic() -> None:
    """
    Test reward multipliers for 3 scenarios: Near Opponent, Far From Opponent,
    Self-Destruct. Manually sets game state and prints reward for verification.
    """
    env = MangoRewardWrapper(MeleeSimEnv(max_frames=5000))
    env.reset(seed=42)
    inner = env.env
    stage = Stage()
    neutral_action = np.array([2, 1, 0, 0, 0, 0], dtype=np.int64)

    def run_scenario(name: str, game_state: GameState) -> float:
        original_step = inner.sim.step

        def mock_step(state, actions):
            return game_state.copy()

        inner.sim.step = mock_step
        env._prev_p1_percent = game_state.players[1].percent
        env._was_in_disadvantage = False
        inner._prev_percent_self = game_state.players[0].percent
        inner._prev_percent_opp = game_state.players[1].percent
        inner._prev_opp_stocks = game_state.players[1].stock
        _, reward, _, _, _ = env.step(neutral_action)
        inner.sim.step = original_step
        return reward

    # 1. Near Opponent: both players close (dist < 30) -> expect PROXIMITY_BONUS
    near_state = GameState(
        frame=100,
        players=[
            CharacterState(x=0.0, y=0.0, percent=0, stock=4, action=Action.IDLE),
            CharacterState(x=15.0, y=0.0, percent=0, stock=4, action=Action.IDLE),
        ],
        stage=stage,
        done=False,
        winner=None,
    )
    near_reward = run_scenario("Near Opponent", near_state)
    print(f"Near Opponent (dist=15): reward={near_reward:.4f} (expect +{PROXIMITY_BONUS} proximity)")

    # 2. Far From Opponent: players far apart -> no proximity bonus
    far_state = GameState(
        frame=100,
        players=[
            CharacterState(x=-50.0, y=0.0, percent=0, stock=4, action=Action.IDLE),
            CharacterState(x=50.0, y=0.0, percent=0, stock=4, action=Action.IDLE),
        ],
        stage=stage,
        done=False,
        winner=None,
    )
    far_reward = run_scenario("Far From Opponent", far_state)
    print(f"Far From Opponent (dist=100): reward={far_reward:.4f} (expect no proximity)")

    # 3. Self-Destruct: agent dead (lost) -> negative base reward
    self_destruct_state = GameState(
        frame=100,
        players=[
            CharacterState(x=-60.0, y=0.0, percent=50, stock=0, action=Action.DEAD),
            CharacterState(x=60.0, y=0.0, percent=20, stock=4, action=Action.IDLE),
        ],
        stage=stage,
        done=True,
        winner=1,
    )
    self_destruct_reward = run_scenario("Self-Destruct", self_destruct_state)
    print(f"Self-Destruct (agent dead, winner=opp): reward={self_destruct_reward:.4f} (expect -1.0 loss)")

    print()
    print("Multipliers: PROXIMITY=+0.05, SHIELD_PRESSURE=+2.0, MOVEMENT=+0.02, RECOVERY=+1.0")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mang0-style Melee PPO trainer")
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run one random rollout and print total_reward (no training)",
    )
    parser.add_argument(
        "--stability",
        action="store_true",
        help="Mini-training on Mac: print loss curve, check NaN, monitor RAM",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test_reward_logic: verify reward multipliers for 3 scenarios",
    )
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel env workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--frames-per-batch",
        type=int,
        default=8192,
        help="Frames per batch (default: 8192 for H100)",
    )
    parser.add_argument("--total-frames", type=int, default=500_000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--sub-batch-size", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use Automatic Mixed Precision (default: True on CUDA)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL_FRAMES,
        help=f"Save checkpoint every N frames (default: {CHECKPOINT_INTERVAL_FRAMES})",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log to Weights & Biases (reward, policy_loss, GPU memory)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load latest checkpoint from checkpoints/ and continue training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device for training (default: cuda for H100)",
    )
    args = parser.parse_args()
    args.use_amp = args.use_amp and not args.no_amp

    if args.sanity:
        _run_sanity_rollout(max_steps=2000)
    elif args.stability:
        _run_stability_test()
    elif args.test:
        test_reward_logic()
    else:
        _train_ppo(args)


if __name__ == "__main__":
    main()
