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
        _get = obs_data.get

        def _ecb(key: str):
            return _get(key, {"top": {"x": 0, "y": 0}, "bottom": {"x": 0, "y": 0}, "left": {"x": 0, "y": 0}, "right": {"x": 0, "y": 0}})

        observation = SmashObservation(
            # Player 1 (agent)
            player_x=_get("player_x", 0.0),
            player_y=_get("player_y", 0.0),
            player_damage=_get("player_damage", 0),
            player_action_state=_get("player_action_state", "unknown"),
            player_stocks=_get("player_stocks", 0),
            # Player 1 physics (5-speed + timers + shield + ECB)
            player_speed_air_x_self=_get("player_speed_air_x_self", 0.0),
            player_speed_ground_x_self=_get("player_speed_ground_x_self", 0.0),
            player_speed_y_self=_get("player_speed_y_self", 0.0),
            player_speed_x_attack=_get("player_speed_x_attack", 0.0),
            player_speed_y_attack=_get("player_speed_y_attack", 0.0),
            player_speed_x=_get("player_speed_x", 0.0),
            player_speed_y=_get("player_speed_y", 0.0),
            player_on_ground=_get("player_on_ground", True),
            player_facing_right=_get("player_facing_right", True),
            player_hitstun_left=_get("player_hitstun_left", 0),
            player_hitlag_left=_get("player_hitlag_left", 0),
            player_jumpsquat_frames_left=_get("player_jumpsquat_frames_left", 0),
            player_invulnerability_left=_get("player_invulnerability_left", 0),
            player_shield_strength=_get("player_shield_strength", 60.0),
            player_ecb=_ecb("player_ecb"),
            # Player 2 (opponent)
            opponent_x=_get("opponent_x", 0.0),
            opponent_y=_get("opponent_y", 0.0),
            opponent_damage=_get("opponent_damage", 0),
            opponent_action_state=_get("opponent_action_state", "unknown"),
            opponent_stocks=_get("opponent_stocks", 0),
            opponent_speed_air_x_self=_get("opponent_speed_air_x_self", 0.0),
            opponent_speed_ground_x_self=_get("opponent_speed_ground_x_self", 0.0),
            opponent_speed_y_self=_get("opponent_speed_y_self", 0.0),
            opponent_speed_x_attack=_get("opponent_speed_x_attack", 0.0),
            opponent_speed_y_attack=_get("opponent_speed_y_attack", 0.0),
            opponent_speed_x=_get("opponent_speed_x", 0.0),
            opponent_speed_y=_get("opponent_speed_y", 0.0),
            opponent_on_ground=_get("opponent_on_ground", True),
            opponent_facing_right=_get("opponent_facing_right", True),
            opponent_hitstun_left=_get("opponent_hitstun_left", 0),
            opponent_hitlag_left=_get("opponent_hitlag_left", 0),
            opponent_jumpsquat_frames_left=_get("opponent_jumpsquat_frames_left", 0),
            opponent_invulnerability_left=_get("opponent_invulnerability_left", 0),
            opponent_shield_strength=_get("opponent_shield_strength", 60.0),
            opponent_ecb=_ecb("opponent_ecb"),
            # General
            projectiles=_get("projectiles", []),
            frame=_get("frame", 0),
            menu_state=_get("menu_state", "unknown"),
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
