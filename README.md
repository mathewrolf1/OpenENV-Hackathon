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

frameReader.py      Existing Dolphin/Slippi 60 FPS loop (libmelee)
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

## Reward design

| Signal | Value | Purpose |
|--------|-------|---------|
| Damage dealt | +0.01 per % | Encourage offense |
| Damage taken | -0.01 per % | Discourage getting hit |
| Stock taken | +0.5 | Reward kills |
| Rest kill bonus | +0.5 extra | Incentivize Rest when it kills |
| Missed Rest (per frame) | -0.002 | Punish whiffing (240f x 0.002 = -0.48 total) |
| Win | +1.0 | Win the game |
| Loss | -1.0 | Lose the game |

The agent learns the fundamental Rest tradeoff: massive reward if it kills
(+0.5 stock + 0.5 Rest bonus + 0.2 damage), massive punishment if it misses
(-0.48 sleep penalty + likely getting punished).

## Hybrid training path

The observation and action interfaces are designed to match
[libmelee](https://github.com/altf4/libmelee) `PlayerState` / `Controller`,
so a policy trained in this simulator can be evaluated or fine-tuned against
real Melee via Dolphin, or combined with Slippi replay data for offline RL.
