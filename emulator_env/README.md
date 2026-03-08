# Smash Melee RL Environment

A reinforcement learning environment that wraps **Super Smash Bros. Melee** running in [Slippi Dolphin](https://slippi.gg/) via [libmelee](https://github.com/altf4/libmelee). Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

The environment exposes the game as a standard RL interface (`reset` / `step`) over HTTP/WebSocket. An RL agent sends controller inputs each frame and receives game state observations back.

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | 3.11 recommended |
| **[uv](https://docs.astral.sh/uv/)** | Fast Python package manager (`pip install uv` or `brew install uv`) |
| **Slippi Dolphin** | Download from [slippi.gg](https://slippi.gg/). Launch it once so it creates its config directories |
| **Melee ISO** | A legally obtained copy of `Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso` |

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd emulator_env
uv sync
```

### 2. Configure environment variables

Copy the example file and fill in your paths:

```bash
cp .env.example .env
```

Then edit `.env` — uncomment and set the three variables for your platform (see [Platform Paths](#platform-paths) below).

| Variable | Required | Description |
|---|---|---|
| `DOLPHIN_PATH` | Yes | Directory containing the Slippi Dolphin executable |
| `ISO_PATH` | Yes | Full path to the Melee v1.02 ISO file |
| `DOLPHIN_HOME` | No | Dolphin user config directory. If set, uses your existing controller/graphics config instead of a temp directory |

The server loads `.env` automatically at startup via `python-dotenv`. You can also export the variables manually if you prefer.

### 3. Start the server

```bash
uv run --project . server
```

This will:
1. Read your `.env` (or exported env vars)
2. Launch Slippi Dolphin
3. Start the OpenEnv HTTP/WebSocket server on port 8000
4. Wait for a client to connect

### 4. Run the test client (in a separate terminal)

```bash
uv run --project . python test_client.py
```

The test client connects, resets the environment (navigates menus, starts a match), sends random inputs for a few frames, and prints observations.

## Platform Paths

### Windows

Slippi Launcher installs Dolphin to your `AppData\Roaming` directory:

```env
DOLPHIN_PATH=C:\Users\you\AppData\Roaming\Slippi Launcher\netplay
ISO_PATH=C:\Users\you\Downloads\Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso
DOLPHIN_HOME=C:\Users\you\AppData\Roaming\Slippi Launcher\netplay\User
```

The executable is `Slippi Dolphin.exe` inside the `DOLPHIN_PATH` directory.

### macOS — Mainline (netplay-beta)

If you installed Slippi Launcher recently, you likely have the mainline build:

```env
DOLPHIN_PATH=~/Library/Application Support/Slippi Launcher/netplay-beta
ISO_PATH=~/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso
DOLPHIN_HOME=~/Library/Application Support/com.project-slippi.dolphin/netplay-beta/User
```

The executable is `Slippi_Dolphin.app/Contents/MacOS/Slippi_Dolphin` (note: underscores).

### macOS — Ishiiruka / Legacy (netplay)

Older Slippi installations use the Ishiiruka build:

```env
DOLPHIN_PATH=~/Library/Application Support/Slippi Launcher/netplay
ISO_PATH=~/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso
DOLPHIN_HOME=~/Library/Application Support/com.project-slippi.dolphin/netplay/User
```

The executable is `Slippi Dolphin.app/Contents/MacOS/Slippi Dolphin` (note: spaces).

> **Tip:** Not sure which build you have? Check if `~/Library/Application Support/Slippi Launcher/netplay-beta` exists. If it does, use the mainline paths. libmelee auto-detects the variant based on whether `netplay-beta` appears in the path.

## Client Usage

```python
from emulator_env import EmulatorEnv, SmashAction

env = EmulatorEnv(base_url="http://localhost:8000")
env.connect()

# reset() navigates menus and starts a match (Fox vs CPU Falco on Final Destination)
obs = env.reset()
print(f"Match started! Player at ({obs.player_x}, {obs.player_y})")

# Send controller inputs, one per frame
result = env.step(SmashAction(stick_x=1.0, button_a=True))
print(f"Player damage: {result.observation.player_damage}%")
print(f"Opponent damage: {result.observation.opponent_damage}%")

env.close()
```

If the server is running on another machine, replace `localhost` with that machine's IP address.

## Action Space

`SmashAction` maps directly to a GameCube controller. All fields are optional (default: neutral).

| Field | Type | Default | Description |
|---|---|---|---|
| `stick_x` | float [-1, 1] | 0.0 | Main stick horizontal (-1 left, 1 right) |
| `stick_y` | float [-1, 1] | 0.0 | Main stick vertical (-1 down, 1 up) |
| `c_stick_x` | float [-1, 1] | 0.0 | C-stick horizontal (smash attacks, aerials) |
| `c_stick_y` | float [-1, 1] | 0.0 | C-stick vertical |
| `button_a` | bool | false | A — normals, smash attacks |
| `button_b` | bool | false | B — special moves |
| `button_x` | bool | false | X — jump |
| `button_y` | bool | false | Y — jump |
| `button_z` | bool | false | Z — grab (grounded), z-air (aerial) |
| `button_l` | bool | false | L — shield, tech, L-cancel |
| `button_r` | bool | false | R — shield, tech, L-cancel |

## Observation Space

`SmashObservation` — game state returned after each frame.

| Field | Type | Description |
|---|---|---|
| `player_x` | float | Player X position (world units) |
| `player_y` | float | Player Y position (world units) |
| `player_damage` | int (0-999) | Player damage percent |
| `player_action_state` | string | Current animation (e.g. `STANDING`, `KNEE_BEND`, `DEAD_DOWN`) |
| `player_stocks` | int | Player remaining stocks |
| `opponent_x` | float | Opponent X position |
| `opponent_y` | float | Opponent Y position |
| `opponent_damage` | int (0-999) | Opponent damage percent |
| `opponent_action_state` | string | Opponent current animation |
| `opponent_stocks` | int | Opponent remaining stocks |
| `menu_state` | string | Game state (`IN_GAME`, `CHARACTER_SELECT`, `POSTGAME_SCORES`, etc.) |
| `reward` | float | Reward signal for this frame |
| `done` | bool | Whether the episode (match) has ended |

## Training Loop Example

```python
from emulator_env import EmulatorEnv, SmashAction
import random

env = EmulatorEnv(base_url="http://localhost:8000")
env.connect()

for episode in range(100):
    result = env.reset()
    total_reward = 0.0
    steps = 0

    while not result.done:
        # Replace this with your RL agent's policy
        action = SmashAction(
            stick_x=random.uniform(-1, 1),
            stick_y=random.uniform(-1, 1),
            button_a=random.random() > 0.7,
            button_b=random.random() > 0.9,
            button_l=random.random() > 0.95,
        )
        result = env.step(action)
        total_reward += result.reward
        steps += 1

    print(f"Episode {episode}: {steps} frames, total reward = {total_reward:.1f}")

env.close()
```

## Match Configuration

By default the environment runs:
- **Player 1 (agent):** Fox
- **Player 2 (CPU):** Falco, Level 3
- **Stage:** Final Destination

These are set in `server/emulator_env_environment.py`. To change them, edit:

```python
self._character = melee.Character.FOX          # your character
self._cpu_character = melee.Character.FALCO    # CPU character
self._stage = melee.Stage.FINAL_DESTINATION    # stage
self._cpu_level = 3                            # CPU difficulty (1-9)
```

**Characters:** `FOX`, `FALCO`, `MARTH`, `SHEIK`, `CPTFALCON`, `PEACH`, `JIGGLYPUFF`, `SAMUS`, `DK`, `LINK`, `YLINK`, `PIKACHU`, `MARIO`, `LUIGI`, `DOC`, `GANONDORF`, `ROY`, `MEWTWO`, `NESS`, `YOSHI`, `KIRBY`, `BOWSER`, `PICHU`, `GAMEANDWATCH`, `ZELDA`, `POPO`

**Stages:** `FINAL_DESTINATION`, `BATTLEFIELD`, `DREAMLAND`, `YOSHIS_STORY`, `FOUNTAIN_OF_DREAMS`, `POKEMON_STADIUM`

## Project Structure

```
emulator_env/
├── __init__.py                # Package exports
├── models.py                  # SmashAction & SmashObservation (Pydantic)
├── client.py                  # EmulatorEnv WebSocket client
├── test_client.py             # Smoke test script
├── .env.example               # Environment variable template
├── openenv.yaml               # OpenEnv manifest
├── pyproject.toml             # Dependencies & project metadata
├── uv.lock                    # Locked dependency versions
├── README.md                  # This file
├── docker-entrypoint.sh       # Container startup (Xvfb + uvicorn)
├── .dockerignore              # Docker build exclusions
└── server/
    ├── __init__.py            # Server module exports
    ├── app.py                 # FastAPI application (HTTP + WebSocket)
    ├── emulator_env_environment.py  # Core environment (wraps libmelee)
    ├── requirements.txt       # Server-specific requirements
    └── Dockerfile             # Container image definition (see note below)
```

## Troubleshooting

**`reset()` times out**
- Menu navigation can take 10-30 seconds on the first run while Dolphin loads shaders.
- If it consistently times out, check the server terminal for errors. The most common cause is incorrect `DOLPHIN_PATH` or `ISO_PATH`.

**Server crashes on startup with `EnvironmentFactoryError`**
- The OpenEnv framework wraps `__init__` errors with a generic message. Look at the **server terminal output** for the actual traceback and pre-flight diagnostic info (env var values, directory checks).

**Dolphin opens but the match never starts**
- Make sure you launched Slippi Dolphin at least once manually so it creates its config directories.
- Verify `DOLPHIN_HOME` points to the correct `User` directory containing `Config/` and `GameSettings/`.

**Client can't connect**
- Verify the server is running: `curl http://localhost:8000/health`
- If running on a remote machine, make sure port 8000 is open.

**Wrong ISO version**
- libmelee requires `Super Smash Bros. Melee (USA) (En,Ja) (v1.02)`. Other regions or versions will fail.

## Docker (Experimental)

A `server/Dockerfile` exists for containerized deployment, but it is **currently non-functional** on most platforms. The extracted Slippi Dolphin AppImage is missing shared library dependencies (glib, pango, fontconfig, harfbuzz, OpenGL, etc.) that are not yet installed in the image. Additionally, Slippi only provides x86_64 Linux builds, so ARM hosts (e.g., Apple Silicon Macs) would need emulation.

For now, **run the server locally** using the instructions above. The Dockerfile is retained for future work on headless/CI deployment.
