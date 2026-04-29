"""
env.py — AI2-THOR Environment Wrapper

Provides a clean interface to the AI2-THOR simulator with discrete actions,
returning structured Observation objects containing RGB, depth, agent pose,
and camera intrinsics.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class Observation:
    """Single observation from the environment."""
    rgb: np.ndarray            # (H, W, 3) uint8
    depth: np.ndarray          # (H, W) float32, meters
    pose: Dict[str, Any]       # {position: {x,y,z}, rotation: {x,y,z}, horizon: float}
    intrinsics: Dict[str, float]  # {fx, fy, cx, cy, fov}
    metadata: Dict[str, Any] = field(default_factory=dict)


# Valid discrete actions the agent can take
ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight", "LookDown", "LookUp", "Done"]


class ThorEnv:
    """Wrapper around AI2-THOR controller for semantic navigation."""

    def __init__(
        self,
        scene: str = "FloorPlan1",
        grid_size: float = 0.25,
        rotate_step: float = 45.0,
        fov: int = 90,
        width: int = 640,
        height: int = 480,
        headless: bool = True,
        visibility_distance: float = 1.5,
    ):
        self.scene = scene
        self.grid_size = grid_size
        self.rotate_step = rotate_step
        self.fov = fov
        self.width = width
        self.height = height
        self.headless = headless
        self.visibility_distance = visibility_distance
        self.controller = None

        # Compute camera intrinsics from FOV
        fov_rad = math.radians(fov / 2.0)
        self.intrinsics = {
            "fx": (width / 2.0) / math.tan(fov_rad),
            "fy": (height / 2.0) / math.tan(fov_rad),
            "cx": width / 2.0,
            "cy": height / 2.0,
            "fov": float(fov),
        }

    def _init_controller(self, scene: str):
        """Initialize or reinitialize the AI2-THOR controller."""
        from ai2thor.controller import Controller

        kwargs = dict(
            scene=scene,
            gridSize=self.grid_size,
            rotateStepDegrees=self.rotate_step,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            width=self.width,
            height=self.height,
            fieldOfView=self.fov,
            visibilityDistance=self.visibility_distance,
            snapToGrid=True,
        )

        # Use headless rendering if requested (requires GPU + Vulkan on Linux)
        if self.headless:
            try:
                from ai2thor.platform import CloudRendering
                kwargs["platform"] = CloudRendering
            except ImportError:
                pass  # Fall back to default X11 rendering

        if self.controller is not None:
            self.controller.stop()

        self.controller = Controller(**kwargs)

    def reset(self, scene: Optional[str] = None) -> Observation:
        """Reset to a scene and return the first observation."""
        scene = scene or self.scene
        self.scene = scene
        self._init_controller(scene)
        return self._make_observation(self.controller.last_event)

    def step(self, action: str) -> Observation:
        """Execute a discrete action and return the new observation."""
        assert self.controller is not None, "Call reset() before step()"
        assert action in ACTIONS, f"Invalid action: {action}. Choose from {ACTIONS}"

        event = self.controller.step(action=action)
        return self._make_observation(event)

    def get_object_positions(self) -> List[Dict[str, Any]]:
        """Return list of all objects with their positions and types."""
        assert self.controller is not None, "Call reset() first"
        objects = self.controller.last_event.metadata["objects"]
        return [
            {
                "objectId": obj["objectId"],
                "objectType": obj["objectType"],
                "name": obj["name"],
                "position": obj["position"],
                "visible": obj["visible"],
                "distance": obj.get("distance", None),
            }
            for obj in objects
        ]

    def get_reachable_positions(self) -> List[Dict[str, float]]:
        """Return list of all reachable grid positions in the scene."""
        assert self.controller is not None, "Call reset() first"
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata["actionReturn"]

    def _make_observation(self, event) -> Observation:
        """Convert an AI2-THOR event to an Observation."""
        agent = event.metadata["agent"]
        pose = {
            "position": agent["position"],      # {x, y, z}
            "rotation": agent["rotation"],       # {x, y, z} in degrees
            "horizon": agent.get("cameraHorizon", 0.0),
        }

        rgb = event.frame  # (H, W, 3) uint8
        depth = event.depth_frame if event.depth_frame is not None else np.zeros(
            (self.height, self.width), dtype=np.float32
        )

        return Observation(
            rgb=rgb,
            depth=depth,
            pose=pose,
            intrinsics=self.intrinsics,
            metadata={
                "objects": event.metadata.get("objects", []),
                "lastActionSuccess": event.metadata.get("lastActionSuccess", False),
                "lastAction": event.metadata.get("lastAction", ""),
            },
        )

    def close(self):
        """Shut down the controller."""
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    def __del__(self):
        if hasattr(self, "controller"):
            self.close()


# ---------------------------------------------------------------------------
# Sanity check: random walk and save frames
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import random
    from PIL import Image

    os.makedirs("debug", exist_ok=True)
    env = ThorEnv(scene="FloorPlan1", headless=True)
    obs = env.reset()

    for i in range(10):
        action = random.choice(["MoveAhead", "RotateLeft", "RotateRight"])
        obs = env.step(action)
        img = Image.fromarray(obs.rgb)
        img.save(f"debug/frame_{i:03d}.png")
        print(f"Step {i}: action={action}, "
              f"pos=({obs.pose['position']['x']:.2f}, "
              f"{obs.pose['position']['z']:.2f}), "
              f"yaw={obs.pose['rotation']['y']:.1f}")

    env.close()
    print("Saved 10 frames to debug/")
