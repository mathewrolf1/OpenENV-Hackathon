# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Emulator Env Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import SmashAction, SmashObservation


class EmulatorEnv(EnvClient[SmashAction, SmashObservation, State]):
    """
    Client for the Emulator Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with EmulatorEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.player_x)
        ...
        ...     result = client.step(SmashAction(stick_x=0.5, button_a=True))
        ...     print(result.observation.player_damage)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = EmulatorEnv.from_docker_image("emulator_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SmashAction(stick_x=1.0))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SmashAction) -> Dict:
        """
        Convert SmashAction to JSON payload for step message.

        Args:
            action: SmashAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "stick_x": action.stick_x,
            "stick_y": action.stick_y,
            "c_stick_x": action.c_stick_x,
            "c_stick_y": action.c_stick_y,
            "button_a": action.button_a,
            "button_b": action.button_b,
            "button_x": action.button_x,
            "button_y": action.button_y,
            "button_z": action.button_z,
            "button_l": action.button_l,
            "button_r": action.button_r,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SmashObservation]:
        """
        Parse server response into StepResult[SmashObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SmashObservation
        """
        obs_data = payload.get("observation", {})
        observation = SmashObservation(
            # Player 1 (agent)
            player_x=obs_data.get("player_x", 0.0),
            player_y=obs_data.get("player_y", 0.0),
            player_damage=obs_data.get("player_damage", 0),
            player_action_state=obs_data.get("player_action_state", "unknown"),
            player_stocks=obs_data.get("player_stocks", 0),
            # Player 1 physics
            player_speed_x=obs_data.get("player_speed_x", 0.0),
            player_speed_y=obs_data.get("player_speed_y", 0.0),
            player_on_ground=obs_data.get("player_on_ground", True),
            player_facing_right=obs_data.get("player_facing_right", True),
            player_hitstun_left=obs_data.get("player_hitstun_left", 0),
            # Player 2 (opponent)
            opponent_x=obs_data.get("opponent_x", 0.0),
            opponent_y=obs_data.get("opponent_y", 0.0),
            opponent_damage=obs_data.get("opponent_damage", 0),
            opponent_action_state=obs_data.get("opponent_action_state", "unknown"),
            opponent_stocks=obs_data.get("opponent_stocks", 0),
            # Player 2 physics
            opponent_speed_x=obs_data.get("opponent_speed_x", 0.0),
            opponent_speed_y=obs_data.get("opponent_speed_y", 0.0),
            opponent_on_ground=obs_data.get("opponent_on_ground", True),
            opponent_facing_right=obs_data.get("opponent_facing_right", True),
            opponent_hitstun_left=obs_data.get("opponent_hitstun_left", 0),
            # General
            frame=obs_data.get("frame", 0),
            menu_state=obs_data.get("menu_state", "unknown"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
