#!/usr/bin/env python3
"""
Mang0-style PPO trainer for Melee sim.

Heuristic reward shaping to encourage aggressive, high-risk play:
- Proximity bonus, shield pressure, movement incentive, high-risk recovery.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from envs.melee_sim_env import MeleeSimEnv, OBS_DIM
from physics.constants import Action, STAGE
from physics.state import GameState


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


def _train_ppo(args: argparse.Namespace) -> None:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA not available, training on CPU")

    def make_env() -> gym.Env:
        return _create_env(max_frames=args.max_frames)

    base_env = GymWrapper(
        ParallelEnv(
            args.num_envs,
            make_env,
            serial_for_single=True,
        ),
        device=device,
    )

    obs_dim = OBS_DIM
    model = ActorCriticMLP(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
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
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        device=device,
        split_trajs=False,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

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

    pbar = tqdm(total=args.total_frames, desc="PPO")

    for i, batch in enumerate(collector):
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

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()

        mean_reward = batch["next", "reward"].mean().item()
        pbar.update(batch.numel())
        pbar.set_postfix(reward=f"{mean_reward:.4f}")

    pbar.close()
    print("Training complete.")


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
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--frames-per-batch", type=int, default=4096)
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
    args = parser.parse_args()

    if args.sanity:
        _run_sanity_rollout(max_steps=2000)
    else:
        _train_ppo(args)


if __name__ == "__main__":
    main()
