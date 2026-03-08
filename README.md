# Melee RL — Fox vs Jigglypuff via Dolphin + OpenEnv

A reinforcement learning system that trains a **Fox** AI agent to play **Super Smash Bros. Melee** in real-time. The agent connects to [Slippi Dolphin](https://slippi.gg/) via [libmelee](https://github.com/altf4/libmelee) and learns through self-play using PPO, served over HTTP with [Meta's OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

## How It Works

```
Training Script (PPO)          OpenEnv Server              Slippi Dolphin
┌──────────────────┐     HTTP/WS      ┌──────────────┐      libmelee      ┌──────────────┐
│ dolphin_train.py │ ──SmashAction──> │ FastAPI app  │ ──controller───> │  Melee game  │
│                  │ <─SmashObs─────  │ (port 8000)  │ <─gamestate────  │  (60 FPS)    │
│  CompetitiveMelee│                  │ EmulatorEnv  │                  │  Fox vs Puff │
│  Reward shaping  │                  │ Server       │                  │  on FD       │
└──────────────────┘                  └──────────────┘                  └──────────────┘
```

1. **OpenEnv server** launches Dolphin, navigates menus, and exposes `reset()` / `step()` over HTTP
2. **Training client** sends controller inputs (stick, buttons) each frame and receives game state back
3. **Reward shaping** (`CompetitiveMeleeReward`) computes per-frame reward on the client side
4. **PPO** updates the policy every 2048 frames

## Training Pipeline

### Phase 1 — Physics Simulator (fast, offline)

Train base policies in a custom Melee physics engine — no emulator needed.

```bash
pip install -r requirements.txt

# Train Puff (defensive, spacing-based)
python3 train.py --total-frames 1500000 --device cpu

# Train Fox/Mango (aggressive, pressure-based)
python3 mango_trainer.py --total-frames 500000 --device cpu
```

Produces `checkpoints/puff_final.pt` and `checkpoints/mango_final.pt`.

### Phase 2 — Dolphin Fine-Tuning (real game)

Fine-tune the sim-trained Fox against real Melee running in Dolphin.

```bash
# Terminal 1: Start the OpenEnv server (launches Dolphin)
cd emulator_env && uv run --project . server

# Terminal 2: Train Fox via PPO against CPU or Puff model
cd emulator_env && uv run python dolphin_train.py --agent mango \
  --checkpoint ../checkpoints/mango_final.pt --total-frames 500000
```

## Configuration

All environment config is in `emulator_env/.env`:

```env
DOLPHIN_PATH=~/Library/Application Support/Slippi Launcher/netplay
ISO_PATH=~/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso
P1_CHARACTER=FOX
P2_CHARACTER=JIGGLYPUFF
CPU_LEVEL=7                # 0 = model-driven P2, 1-9 = Dolphin CPU AI
TRAINING_MODE=NORMAL       # NORMAL or RECOVERY
# P2_CHECKPOINT_PATH=../checkpoints/puff_final.pt  # uncomment when CPU_LEVEL=0
```

| Setting | Options | Description |
|---------|---------|-------------|
| `CPU_LEVEL` | `0` | P2 controlled by a trained model (`P2_CHECKPOINT_PATH`) |
| `CPU_LEVEL` | `1-9` | P2 controlled by Dolphin's built-in CPU AI |
| `TRAINING_MODE` | `NORMAL` | Standard matches |
| `TRAINING_MODE` | `RECOVERY` | 20% of resets spawn P1 off-stage for recovery training |

## Model Architecture

**ActorCriticMLP** — shared backbone with separate actor and critic heads.

| Component | Details |
|-----------|---------|
| Backbone | 3-layer MLP, 256 hidden units, Tanh activation |
| Actor head | Linear(256, 17) — MultiDiscrete `[5, 4, 2, 2, 2, 2]` |
| Critic head | Linear(256, 1) — value estimate |
| Observation | 26-dim vector (13 per player) |
| Action space | 320 discrete actions (stick × buttons) |

### Observation Vector (26-dim)

Per player (13 dims): position (x, y), velocities (x, y), damage (normalized), stocks (normalized), on_ground, facing, action_state, hitstun.

### Action Space

`MultiDiscrete([5, 4, 2, 2, 2, 2])` mapping to GameCube controller:

| Index | Bins | Control |
|-------|------|---------|
| 0 | 5 | Stick X: `[-1.0, -0.6, 0.0, 0.6, 1.0]` |
| 1 | 4 | Stick Y: `[-1.0, 0.0, 0.5, 1.0]` |
| 2 | 2 | X button (jump) |
| 3 | 2 | A button (attack) |
| 4 | 2 | Z button (grab) |
| 5 | 2 | B button (special) |

## Reward System

**CompetitiveMeleeReward** (`rewards/competitive.py`) — aggressive Fox / Mango style:

| Signal | Value | Purpose |
|--------|-------|---------|
| Damage dealt | +0.05 per % | Reward hitting opponent |
| Damage taken | -0.005 per % | Light penalty for getting hit |
| Stock taken | +1.0 | Reward kills |
| Stock lost | -5.0 | Heavy anti-SD penalty |
| Win / Loss | +2.0 / -1.0 | Terminal bonus |
| Approach | +0.008 per unit closed | Incentivize charging toward opponent |
| Proximity | +0.003/frame within 25 units | Reward staying close |
| Combo | +0.06 per hit in hitstun | Reward follow-up hits |
| Edgeguard | +0.08 per hit while opponent off-stage | Reward edgeguarding |
| Velocity | +0.02 × speed within 50 units | Reward active movement near opponent |
| Existence | +0.002/frame on-stage | Survival incentive |
| Off-stage | -0.015/frame | Penalize being past the ledge |
| Blastzone | -0.0005 × (excess)² | Exponential wall near blastzones |
| Shield pressure | +0.05 × shield drain | Reward pressuring shield |
| Recovery | +0.03 × edge_dist_closed | Reward recovering to stage |

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 2048 |
| Mini-batch size | 64 |
| PPO epochs | 4 |
| Learning rate | 1e-4 |
| Gamma | 0.99 |
| Lambda (GAE) | 0.95 |
| Clip epsilon | 0.2 |
| Entropy coef | 0.01 → 0.001 (linear decay over 500k frames) |
| Max grad norm | 1.0 |

## Tech Stack

| Tool | Role |
|------|------|
| [OpenEnv](https://github.com/meta-pytorch/OpenEnv) (Meta) | HTTP/WebSocket server framework for RL environments |
| [libmelee](https://github.com/altf4/libmelee) | Python API for reading game state and sending inputs to Dolphin |
| [Slippi Dolphin](https://slippi.gg/) | Modified GameCube emulator for Melee |
| [PyTorch](https://pytorch.org/) | Neural network + model inference |
| [TorchRL](https://github.com/pytorch/rl) | PPO implementation for sim training |
| [Gymnasium](https://gymnasium.farama.org/) | Standard RL environment interface (sim) |
| [FastAPI](https://fastapi.tiangolo.com/) | HTTP server (via OpenEnv) |
| [uv](https://docs.astral.sh/uv/) | Python package manager |

## Project Structure

```
├── physics/                    # Custom Melee physics simulator
│   ├── state.py                #   Game state dataclasses
│   ├── constants.py            #   Character + stage constants
│   ├── melee_physics.py        #   Knockback, hitstun, gravity
│   └── simulator.py            #   Deterministic step function
│
├── envs/                       # RL environments
│   ├── melee_sim_env.py        #   Gymnasium env (physics sim)
│   └── melee_torchrl_env.py    #   TorchRL env (alternative)
│
├── rewards/                    # Reward shaping
│   ├── competitive.py          #   CompetitiveMeleeReward (Fox/Mango)
│   └── puff.py                 #   PuffReward (Jigglypuff)
│
├── emulator_env/               # Dolphin integration (OpenEnv)
│   ├── server/
│   │   ├── app.py              #   FastAPI server (OpenEnv create_app)
│   │   └── emulator_env_environment.py  # Dolphin wrapper (reset/step)
│   ├── client.py               #   EmulatorEnv WebSocket client
│   ├── dolphin_train.py        #   PPO fine-tuning loop
│   ├── policy_runner.py        #   ActorCriticMLP + obs/action conversion
│   ├── models.py               #   SmashAction / SmashObservation (Pydantic)
│   ├── menu_nav.py             #   Automated menu navigation
│   └── melee_constants.py      #   Stage geometry + velocity limits
│
├── train.py                    # Puff PPO trainer (sim)
├── mango_trainer.py            # Fox/Mango PPO trainer (sim)
├── opponents.py                # Load frozen opponent checkpoints
│
└── checkpoints/                # Trained models
    ├── mango_final.pt          #   Sim-trained Fox
    ├── puff_final.pt           #   Sim-trained Puff
    └── dolphin_fox_final_*.pt  #   Dolphin fine-tuned Fox
```

## Checkpoints

| File | Description | Training |
|------|-------------|----------|
| `mango_final.pt` | Base Fox policy | Physics sim, 500k+ frames |
| `puff_final.pt` | Base Puff policy | Physics sim, 1.5M frames |
| `dolphin_fox_final_501760.pt` | Fine-tuned Fox | Dolphin, 500k frames vs CPU/Puff |

Checkpoint format: `{"model_state_dict": ..., "optim_state_dict": ..., "total_frames": int}`

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.11 recommended |
| [uv](https://docs.astral.sh/uv/) | `pip install uv` or `brew install uv` |
| Slippi Dolphin | Download from [slippi.gg](https://slippi.gg/) |
| Melee ISO | `Super Smash Bros. Melee (USA) (En,Ja) (v1.02)` |
