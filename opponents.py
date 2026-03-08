"""Universal opponent loaders for cross-play between Puff and Mango trainers.

Both trainers now use the same ActorCriticMLP architecture and .pt checkpoint
format. Each class implements ``__call__(state, idx) -> action_dict`` compatible
with MeleeSimEnv's ``opponent_fn`` interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from envs.melee_sim_env import OBS_DIM, _build_obs, _decode_action
from physics.state import GameState

# ---------------------------------------------------------------------------
# Shared constants (must match both trainers)
# ---------------------------------------------------------------------------

ACTION_NVEC = [5, 4, 2, 2, 2, 2]
NUM_ACTIONS = sum(ACTION_NVEC)


# ---------------------------------------------------------------------------
# ActorCriticMLP (inference-only mirror, avoids circular import)
# ---------------------------------------------------------------------------

class _ActorCriticMLP(nn.Module):
    """Matches mango_trainer.ActorCriticMLP / train.py's imported version."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.action_nvec = tuple(ACTION_NVEC)
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.backbone(obs)
        logits_flat = self.actor_head(features)
        return logits_flat


# ---------------------------------------------------------------------------
# TorchRLOpponent — loads any .pt checkpoint (Puff or Mango)
# ---------------------------------------------------------------------------

class TorchRLOpponent:
    """Load a .pt checkpoint (ActorCriticMLP) as an opponent."""

    def __init__(
        self,
        path: str,
        hidden_dim: int = 256,
        num_layers: int = 3,
        deterministic: bool = False,
    ):
        self._deterministic = deterministic
        self._model = _ActorCriticMLP(
            obs_dim=OBS_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

    @torch.no_grad()
    def __call__(self, state: GameState, idx: int) -> Dict:
        obs = _build_obs(state, idx)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        logits_flat = self._model(obs_t)

        actions = []
        offset = 0
        for n in ACTION_NVEC:
            logits = logits_flat[:, offset : offset + n]
            if self._deterministic:
                a = logits.argmax(dim=-1)
            else:
                a = torch.distributions.Categorical(logits=logits).sample()
            actions.append(a.item())
            offset += n

        action = np.array(actions, dtype=np.int64)
        return _decode_action(action)


# ---------------------------------------------------------------------------
# SB3Opponent — backward compat for .zip files
# ---------------------------------------------------------------------------

class SB3Opponent:
    """Load a legacy Stable-Baselines3 .zip checkpoint as an opponent."""

    def __init__(self, path: str = "puff_ppo.zip", deterministic: bool = False):
        from stable_baselines3 import PPO
        self._model = PPO.load(path)
        self._model.policy.set_training_mode(False)
        self._deterministic = deterministic

    def __call__(self, state: GameState, idx: int) -> Dict:
        obs = _build_obs(state, idx)
        action, _ = self._model.predict(obs, deterministic=self._deterministic)
        return _decode_action(action)


# ---------------------------------------------------------------------------
# Factory — load by name
# ---------------------------------------------------------------------------

PUFF_DEFAULT_PATH = "checkpoints/puff_final.pt"
MANGO_DEFAULT_PATH = "checkpoints/mango_final.pt"


def load_opponent(name: str, **kwargs) -> callable:
    """Load an opponent by name: 'puff', 'mango', or a file path.

    Both 'puff' and 'mango' use the same TorchRLOpponent loader since they
    now share the ActorCriticMLP architecture and .pt format.
    Legacy .zip files are still supported via SB3Opponent.
    """
    det = kwargs.pop("deterministic", False)

    if name == "puff":
        path = kwargs.pop("path", PUFF_DEFAULT_PATH)
        return TorchRLOpponent(path=path, deterministic=det, **kwargs)
    elif name == "mango":
        path = kwargs.pop("path", MANGO_DEFAULT_PATH)
        return TorchRLOpponent(path=path, deterministic=det, **kwargs)
    elif name.endswith(".zip"):
        return SB3Opponent(path=name, deterministic=det)
    elif name.endswith(".pt"):
        return TorchRLOpponent(path=name, deterministic=det, **kwargs)
    else:
        raise ValueError(
            f"Unknown opponent '{name}'. Use 'puff', 'mango', a .zip, or a .pt path."
        )
