"""Quick smoke test for the EmulatorEnv client.

Usage:
    1. Start the server in a separate terminal:
           $env:DOLPHIN_PATH = "C:\\path\\to\\Slippi Dolphin.exe"
           uv run --project . server

    2. Run this script:
           uv run --project . python test_client.py
"""

import logging
import sys

from emulator_env import SmashAction, SmashObservation, EmulatorEnv

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def log_obs(label: str, result) -> None:
    """Log a single-line summary of a StepResult."""
    obs = result.observation
    log.info(
        "%s | P1(%6.1f,%6.1f) %3d%% stk=%d  P2(%6.1f,%6.1f) %3d%% stk=%d | r=%+.4f done=%s",
        label,
        obs.player_x, obs.player_y, obs.player_damage, obs.player_stocks,
        obs.opponent_x, obs.opponent_y, obs.opponent_damage, obs.opponent_stocks,
        result.reward or 0.0, result.done,
    )


def main():
    base_url = "http://localhost:8000"

    try:
        env = EmulatorEnv(base_url=base_url)
        env.connect()
        log.info("Connected to %s", base_url)
    except ConnectionError as e:
        log.error("Connection failed: %s — is the server running?", e)
        sys.exit(1)

    try:
        result = env.reset()
        log_obs("reset  ", result)

        actions = [
            SmashAction(stick_x=0.0, stick_y=0.0),                           # neutral
            SmashAction(stick_x=1.0, stick_y=0.0, button_a=True),            # forward tilt
            SmashAction(stick_x=-1.0, stick_y=0.5, button_b=True),           # side-B
            SmashAction(button_y=True),                                       # jump
            SmashAction(c_stick_x=1.0),                                       # c-stick fsmash
            SmashAction(button_z=True),                                       # grab
            SmashAction(button_l=True),                                       # shield
            SmashAction(stick_x=0.0, stick_y=-1.0, button_a=True),           # down tilt
        ]

        for i, action in enumerate(actions):
            result = env.step(action)
            log_obs(f"step {i+1:2d}", result)

        log.info("All %d steps completed successfully.", len(actions))

    finally:
        env.close()
        log.info("Client closed.")


if __name__ == "__main__":
    main()
