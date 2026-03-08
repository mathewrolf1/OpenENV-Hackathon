"""Quick smoke test for the EmulatorEnv client.

Usage:
    1. Start the server in a separate terminal:
           $env:DOLPHIN_PATH = "C:\\path\\to\\Slippi Dolphin.exe"
           uv run --project . server

    2. Run this script:
           uv run --project . python test_client.py
"""

import sys
from emulator_env import SmashAction, SmashObservation, EmulatorEnv


def print_obs(label: str, result):
    """Pretty-print a StepResult."""
    obs = result.observation
    print(f"  [{label}]")
    print(f"    menu_state           = {obs.menu_state}")
    print(
        f"    player  x={obs.player_x:7.2f}  y={obs.player_y:7.2f}  "
        f"dmg={obs.player_damage:3d}%  stocks={obs.player_stocks}  "
        f"action={obs.player_action_state}"
    )
    print(
        f"    opponent x={obs.opponent_x:7.2f}  y={obs.opponent_y:7.2f}  "
        f"dmg={obs.opponent_damage:3d}%  stocks={obs.opponent_stocks}  "
        f"action={obs.opponent_action_state}"
    )
    print(f"    reward={result.reward}  done={result.done}")
    print()


def main():
    base_url = "http://localhost:8000"
    print(f"Connecting to {base_url} ...")

    try:
        env = EmulatorEnv(base_url=base_url)
        env.connect()
        print("Connected via WebSocket.\n")
    except ConnectionError as e:
        print(f"Failed to connect: {e}")
        print("Is the server running? Start it with: uv run --project . server")
        sys.exit(1)

    try:
        # --- reset (waits for menu navigation to finish) ---
        print("Calling reset() -- this will navigate menus and start a match ...")
        result = env.reset()
        print_obs("reset", result)

        # --- a few steps exercising different buttons ---
        actions = [
            SmashAction(stick_x=0.0, stick_y=0.0),  # neutral
            SmashAction(stick_x=1.0, stick_y=0.0, button_a=True),  # forward tilt
            SmashAction(stick_x=-1.0, stick_y=0.5, button_b=True),  # side-B
            SmashAction(button_y=True),  # jump
            SmashAction(c_stick_x=1.0),  # c-stick forward smash
            SmashAction(button_z=True),  # grab
            SmashAction(button_l=True),  # shield
            SmashAction(stick_x=0.0, stick_y=-1.0, button_a=True),  # down tilt
        ]

        for i, action in enumerate(actions):
            print(f"Step {i + 1}: {action.model_dump()}")
            result = env.step(action)
            print_obs(f"step {i + 1}", result)

        print("All steps completed successfully.")

    finally:
        env.close()
        print("Client closed.")


if __name__ == "__main__":
    main()
