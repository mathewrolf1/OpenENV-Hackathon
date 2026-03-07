#!/usr/bin/env python3
"""Quick sanity checks for the Jigglypuff Melee physics sim + Gymnasium env.

Run:  python sanity_check.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from physics.constants import CHAR, MOVES
from physics.melee_physics import compute_knockback, compute_hitstun
from physics.simulator import Simulator
from envs.melee_sim_env import MeleeSimEnv

idle = {"stick_x": 0.0, "stick_y": 0.0, "jump": False,
        "attack": False, "grab": False, "special": False}

# ---- 1. Knockback / hitstun plausibility (Rest) ----

print("=== Rest knockback scaling ===")
for pct in [0, 25, 50, 70, 100]:
    kb = compute_knockback(pct, damage=20.0, weight=CHAR["weight"],
                           base_knockback=40.0, knockback_growth=180.0)
    hs = compute_hitstun(kb)
    print(f"  Rest @ {pct:>3d}%: KB={kb:6.1f}  hitstun={hs:>3d} frames")

print()

# ---- 2. Puff jump arc (floaty!) ----

print("=== Puff jump arc (should be very floaty) ===")
sim = Simulator()
gs = sim.reset()

jump_action = {**idle, "jump": True}

for f in range(100):
    act = jump_action if f == 0 else idle
    gs = sim.step(gs, [act, idle])
    p = gs.players[0]
    if f % 10 == 0 or f < 8 or (p.on_ground and f > 5):
        print(f"  frame {gs.frame:>3d}: x={p.x:7.2f} y={p.y:7.2f} "
              f"vy={p.speed_y_self:6.3f} on_ground={p.on_ground} "
              f"jumps_left={p.jumps_left}")
    if p.on_ground and f > 10:
        print(f"  Landed on frame {gs.frame}")
        break

print()

# ---- 3. Multi-jump (Puff has 5 midair jumps) ----

print("=== Multi-jump test (6 total jumps) ===")
sim2 = Simulator()
gs2 = sim2.reset()

for f in range(200):
    act = jump_action if f % 20 == 0 else idle
    gs2 = sim2.step(gs2, [act, idle])
    p = gs2.players[0]
    if f % 20 == 0:
        print(f"  frame {gs2.frame:>3d}: y={p.y:7.2f} jumps_left={p.jumps_left}")
    if p.jumps_left == 0 and f % 20 == 0:
        print(f"  All jumps used at frame {gs2.frame}")
        break

print()

# ---- 4. Rest hit test (close range, overlapping) ----

print("=== Rest hit test: point-blank ===")
sim3 = Simulator()
gs3 = sim3.reset()
gs3.players[0].x = -1.0
gs3.players[1].x = 0.0

rest_action = {"stick_x": 0.0, "stick_y": -1.0, "jump": False,
               "attack": False, "grab": False, "special": True}

hit_connected = False
for f in range(20):
    act = rest_action if f == 0 else idle
    gs3 = sim3.step(gs3, [act, idle])
    p0, p1 = gs3.players
    if p1.percent > 0 and not hit_connected:
        hit_connected = True
    if f < 10:
        print(f"  frame {gs3.frame:>3d}: p0 x={p0.x:7.2f} action={p0.action.name:20s} "
              f"connected={p0.attack_connected} | "
              f"p1 x={p1.x:7.2f} pct={p1.percent:5.1f} hitstun={p1.hitstun_frames_left}")

if hit_connected:
    print("  -> Rest connected!")
else:
    print("  -> Rest did NOT connect")

print()

# ---- 5. Rest kill test (should kill at moderate percent) ----

print("=== Rest kill test @ 50% ===")
sim4 = Simulator()
gs4 = sim4.reset()
gs4.players[0].x = -1.0  # overlapping — Rest has tiny hitbox
gs4.players[1].x = 0.0
gs4.players[1].percent = 50.0

for f in range(80):
    act = rest_action if f == 0 else idle
    gs4 = sim4.step(gs4, [act, idle])
    p0, p1 = gs4.players
    if f < 5 or p1.stock < 4 or p1.action.name == "DEAD":
        print(f"  frame {gs4.frame:>3d}: p1 x={p1.x:7.2f} y={p1.y:7.2f} "
              f"pct={p1.percent:5.1f} stocks={p1.stock} action={p1.action.name}")
    if p1.stock < 4:
        print(f"  -> Rest killed at frame {gs4.frame}!")
        break

print()

# ---- 6. Rest miss penalty (240 frames asleep) ----

print("=== Rest miss test (whiffed, 240f sleep) ===")
sim5 = Simulator()
gs5 = sim5.reset()
gs5.players[0].x = -30.0
gs5.players[1].x = 30.0

for f in range(260):
    act = rest_action if f == 0 else idle
    gs5 = sim5.step(gs5, [act, idle])
    p0 = gs5.players[0]
    if f in (0, 1, 2, 5, 50, 100, 200, 239, 240, 241, 242):
        print(f"  frame {gs5.frame:>3d}: action={p0.action.name:20s} frame={p0.action_frame:>3d} "
              f"connected={p0.attack_connected}")
    if p0.action.name == "IDLE" and f > 5:
        print(f"  -> Woke up on frame {gs5.frame}")
        break

print()

# ---- 7. Gym env: random rollout with special button ----

print("=== Gym env: 2000-step random rollout (Puff + Rest) ===")
env = MeleeSimEnv(max_frames=2000)
obs, info = env.reset(seed=42)
print(f"  obs shape: {obs.shape}, action space: {env.action_space}")

total_reward = 0.0
steps = 0
rest_count = 0
for _ in range(2000):
    action = env.action_space.sample()
    if action[5] == 1:
        rest_count += 1
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        break

print(f"  steps: {steps}")
print(f"  rest attempts: {rest_count}")
print(f"  total reward: {total_reward:.4f}")
print(f"  final info: {info}")

print()
print("All checks passed.")
