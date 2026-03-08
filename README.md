# Melee RL — Jigglypuff Rest Trainer

A Jigglypuff physics simulator wrapped in a
[Gymnasium](https://gymnasium.farama.org/) environment, built for reinforcement
learning with an emphasis on learning **when to land Rest** — Puff's iconic
frame-1 kill move with extreme risk/reward.

## Quick start

```bash
pip install -r requirements.txt

# Run sanity checks (Rest kill, Puff floaty physics, multi-jump, miss penalty)
python sanity_check.py

# Use the env in your own script
python -c "
from envs.melee_sim_env import MeleeSimEnv
env = MeleeSimEnv(max_frames=3600)
obs, info = env.reset(seed=0)
for _ in range(1000):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        break
print(info)
"
```

## Architecture

```
physics/
  state.py          CharacterState, Stage, GameState dataclasses
  constants.py      Jigglypuff constants, stage geometry, move data (incl. Rest)
  melee_physics.py  Knockback formula, hitstun, gravity, traction
  simulator.py      Deterministic step(state, actions) -> state

envs/
  melee_sim_env.py  Gymnasium env wrapping the simulator

train.py            Puff PPO trainer (TorchRL)
mango_trainer.py    Mango PPO trainer (TorchRL)
opponents.py        Load any checkpoint as a frozen opponent
frameReader.py      Dolphin/Slippi 60 FPS loop (libmelee)
```

## What's implemented

- **Jigglypuff** — lightest character, lowest gravity (very floaty), best air
  speed, 5 midair jumps, 6-frame jumpsquat
- **Final Destination** stage with blastzones
- **4 moves**: jab, grab + throw, forward smash, **Rest** (down + special)
- **Rest** — frame-1 active, tiny hitbox (must overlap opponent), massive
  knockback (kills around 50%), but **240 frames of sleep** if you miss
- **Melee-style knockback formula** with base knockback and growth scaling
- **Hitstun** (0.4x multiplier), attack velocity decay, traction during attacks

## Observation space

26-dimensional float vector (13 per player, agent-relative):
position, velocities (self + attack-induced), percent, stocks, on_ground,
facing, action, action_frame, hitstun.

## Action space

`MultiDiscrete([5, 4, 2, 2, 2, 2])` — stick_x (5 bins), stick_y (4 bins),
jump, attack, grab, special.

Rest is triggered by **special + stick_y < -0.3** (down + special), matching
the real Melee input.

## Trainers

Two agents train in this sim, both using the same `ActorCriticMLP` (3×256 Tanh
backbone) and `.pt` checkpoint format so their policies are interchangeable.

| Trainer | File | Style | Reward focus |
|---------|------|-------|--------------|
| **Puff** | `train.py` | Defensive, spacing-based | Orbit at jab range, Rest kills |
| **Mango** | `mango_trainer.py` | Aggressive, pressure-based | Proximity, movement, damage |

### Reward design

**Base env (shared):**

| Signal | Value |
|--------|-------|
| Damage dealt | +0.01 per % |
| Damage taken | -0.01 per % |
| Stock taken | +0.5 |
| Win / Loss | +1.0 / -1.0 |

**Puff wrapper adds:**

| Signal | Value |
|--------|-------|
| Spacing sweet-spot (Gaussian peak at 10 units) | +0.003/frame |
| Gradient toward sweet-spot when > 20 units | +0.005 per unit closed |
| Extra damage amplification | +0.04 per % (total 0.05) |
| Rest kill bonus | +0.5 |
| Missed Rest sleep penalty | -0.002/frame (×240f = -0.48 total) |

**Mango wrapper adds:**

| Signal | Value |
|--------|-------|
| Proximity (within 30 units) | +0.05/frame |
| Shield pressure (any damage dealt) | +2.0 flat |
| Movement (RUN or AIRBORNE) | +0.02/frame |
| Recovery (return from disadvantage) | +1.0 flat |

---

## Training pipeline

### Phase 1 — Solo sim training

Get each agent learning basic movement and dealing damage before they fight
each other. Mango converges faster due to denser reward; Puff needs more steps.

```bash
# Terminal 1 — train Puff
python3 train.py --total-frames 1500000 --device cpu

# Terminal 2 — train Mango in parallel
python3 mango_trainer.py --total-frames 500000 --device cpu
```

Check both are dealing damage before proceeding:

```bash
python3 train.py --eval
python3 mango_trainer.py --sanity
```

Puff eval should show `dmg_dealt > 0`. Mango sanity should show `total_reward > 50`.

### Phase 2 — Cross-play round 1

Each agent trains against the other's frozen policy.

```bash
# Puff learns to counter Mango's aggression
python3 train.py --opponent mango --total-frames 500000 --device cpu --resume

# Mango learns to counter Puff's spacing and Rest
python3 mango_trainer.py --opponent puff --total-frames 500000 --device cpu --resume
```

Evaluate head-to-head:

```bash
python3 train.py --eval-vs mango --eval-episodes 10
python3 mango_trainer.py --eval-vs puff --eval-episodes 10
```

### Phase 3 — Cross-play round 2 (iterate)

Repeat with updated models. Each round the opponent is harder.

```bash
python3 train.py --opponent mango --total-frames 500000 --device cpu --resume
python3 mango_trainer.py --opponent puff --total-frames 500000 --device cpu --resume

python3 train.py --eval-vs mango --eval-episodes 10
```

Repeat until win rates stabilize (neither agent consistently dominates).

---

## Hybrid training path

The observation and action interfaces are designed to match
[libmelee](https://github.com/altf4/libmelee) `PlayerState` / `Controller`,
so a policy trained in this simulator can be evaluated or fine-tuned against
real Melee via Dolphin, or combined with Slippi replay data for offline RL.
