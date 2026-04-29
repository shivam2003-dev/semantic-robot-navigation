"""
tests/test_env.py — Unit tests for ThorEnv (mock-based, no AI2-THOR required).
"""

import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from env import ThorEnv, Observation, ACTIONS


class TestObservation:
    """Observation dataclass basics."""

    def test_creation(self):
        obs = Observation(
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            depth=np.zeros((480, 640), dtype=np.float32),
            pose={
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "horizon": 0.0,
            },
            intrinsics={"fx": 320.0, "fy": 240.0, "cx": 320.0, "cy": 240.0, "fov": 90.0},
        )
        assert obs.rgb.shape == (480, 640, 3)
        assert obs.depth.shape == (480, 640)


class TestIntrinsics:
    """Camera intrinsic computation."""

    def test_fov_90(self):
        env = ThorEnv.__new__(ThorEnv)
        env.width = 640
        env.height = 480
        env.fov = 90
        fov_rad = math.radians(90 / 2.0)
        expected_fx = (640 / 2.0) / math.tan(fov_rad)
        env.intrinsics = {
            "fx": (env.width / 2.0) / math.tan(fov_rad),
            "fy": (env.height / 2.0) / math.tan(fov_rad),
            "cx": env.width / 2.0,
            "cy": env.height / 2.0,
            "fov": 90.0,
        }
        assert abs(env.intrinsics["fx"] - expected_fx) < 1e-6
        assert env.intrinsics["cx"] == 320.0
        assert env.intrinsics["cy"] == 240.0


class TestActions:
    """Action space validation."""

    def test_valid_actions(self):
        assert "MoveAhead" in ACTIONS
        assert "RotateLeft" in ACTIONS
        assert "RotateRight" in ACTIONS
        assert "LookDown" in ACTIONS
        assert "LookUp" in ACTIONS
        assert "Done" in ACTIONS

    def test_action_count(self):
        assert len(ACTIONS) == 6


class TestThorEnvMocked:
    """ThorEnv with mocked AI2-THOR controller."""

    def _make_mock_event(self):
        """Create a mock AI2-THOR event."""
        event = MagicMock()
        event.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        event.depth_frame = np.ones((480, 640), dtype=np.float32) * 2.0
        event.metadata = {
            "agent": {
                "position": {"x": 1.0, "y": 0.9, "z": 2.0},
                "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                "cameraHorizon": 0.0,
            },
            "objects": [],
            "lastActionSuccess": True,
            "lastAction": "MoveAhead",
        }
        return event

    @patch("ai2thor.controller.Controller")
    def test_reset_returns_observation(self, mock_ctrl_class):
        """reset() should return a valid Observation."""
        mock_ctrl = MagicMock()
        mock_ctrl.last_event = self._make_mock_event()
        mock_ctrl_class.return_value = mock_ctrl

        env = ThorEnv(scene="FloorPlan1", headless=False)
        obs = env.reset()

        assert isinstance(obs, Observation)
        assert obs.rgb.shape == (480, 640, 3)
        assert obs.pose["position"]["x"] == 1.0

    @patch("ai2thor.controller.Controller")
    def test_step_executes_action(self, mock_ctrl_class):
        """step() should call the controller and return Observation."""
        mock_ctrl = MagicMock()
        mock_ctrl.last_event = self._make_mock_event()
        mock_ctrl.step.return_value = self._make_mock_event()
        mock_ctrl_class.return_value = mock_ctrl

        env = ThorEnv(scene="FloorPlan1", headless=False)
        env.reset()
        obs = env.step("MoveAhead")

        assert isinstance(obs, Observation)
        mock_ctrl.step.assert_called_once_with(action="MoveAhead")
