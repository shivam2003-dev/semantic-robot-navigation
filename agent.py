"""
agent.py — Navigation Agent State Machine

Implements the core agent loop: EXPLORE → SEARCH_FRONTIER → GROUND → APPROACH → STOP.
Coordinates the environment, grounding, mapping, and language modules.
"""

import json
import math
import os
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple

from env import ThorEnv, Observation
from grounding import VLGrounder, Detection
from mapping import OccupancyMap
from language import parse, ParsedInstruction


class AgentState(Enum):
    EXPLORE = auto()
    SEARCH_FRONTIER = auto()
    GROUND = auto()
    APPROACH = auto()
    STOP = auto()


@dataclass
class EpisodeResult:
    """Result of a navigation episode."""
    success: bool
    steps: int
    path_length: float           # total distance traveled (meters)
    trajectory: List[Dict]       # per-step records
    final_distance: float = -1.0 # distance to target at end
    instruction: str = ""
    scene: str = ""


class NavigationAgent:
    """
    State-machine agent that navigates to objects described in natural language.

    States:
        EXPLORE        — Initial 360° scan to build the map.
        SEARCH_FRONTIER — Navigate toward unexplored frontiers, periodically grounding.
        GROUND         — Target detected; back-project to 3D and set as goal.
        APPROACH       — Follow path toward target; verify with grounding each step.
        STOP           — Target reached; call Done.
    """

    def __init__(
        self,
        env: ThorEnv,
        grounder: VLGrounder,
        mapper: Optional[OccupancyMap] = None,
        ground_every_n: int = 5,
        detection_threshold: float = 0.15,
        approach_distance: float = 1.0,
        max_steps: int = 250,
        log_dir: Optional[str] = None,
    ):
        self.env = env
        self.grounder = grounder
        self.mapper = mapper or OccupancyMap()
        self.ground_every_n = ground_every_n
        self.detection_threshold = detection_threshold
        self.approach_distance = approach_distance
        self.max_steps = max_steps

        # Logging
        if log_dir is None:
            log_dir = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Episode state
        self.state = AgentState.EXPLORE
        self.step_count = 0
        self.path_length = 0.0
        self.prev_pos = None
        self.trajectory = []
        self.current_path = []       # planned path (world coords)
        self.path_index = 0
        self.target_world = None     # (x, z) goal in world coords
        self.explore_rotations = 0   # count rotations during EXPLORE
        self.no_detection_count = 0  # steps without a detection
        self.last_detections = []
        self.query = ""

    def run(
        self,
        instruction: str,
        scene: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> EpisodeResult:
        """
        Run a full navigation episode.

        Args:
            instruction: Natural language navigation command.
            scene: Override scene (default: env's scene).
            max_steps: Override step budget.

        Returns:
            EpisodeResult with success/failure and trajectory.
        """
        max_steps = max_steps or self.max_steps

        # Parse instruction
        parsed = parse(instruction)
        self.query = parsed.query
        print(f"[agent] Instruction: '{instruction}'")
        print(f"[agent] Parsed → target='{parsed.target}', "
              f"attrs={parsed.attributes}, room={parsed.room_hint}, "
              f"query='{self.query}'")

        # Reset environment
        obs = self.env.reset(scene)
        self._reset_episode()

        # Open trace log
        trace_path = os.path.join(self.log_dir, "trace.jsonl")
        trace_file = open(trace_path, "w")

        try:
            for step in range(max_steps):
                self.step_count = step

                # Update map
                self.mapper.update(obs.depth, obs.pose, obs.intrinsics)

                # Track path length
                pos = obs.pose["position"]
                if self.prev_pos is not None:
                    dx = pos["x"] - self.prev_pos["x"]
                    dz = pos["z"] - self.prev_pos["z"]
                    self.path_length += math.sqrt(dx * dx + dz * dz)
                self.prev_pos = dict(pos)

                # Decide whether to run grounding this step
                should_ground = self._should_ground(step)
                detections = []
                if should_ground:
                    detections = self.grounder.score_frame(obs.rgb, self.query)
                    self.last_detections = detections

                # State machine transition
                action = self._decide_action(obs, detections, parsed)

                # Log step
                record = self._make_record(step, obs, action, detections)
                trace_file.write(json.dumps(record) + "\n")
                self.trajectory.append(record)

                # Check for stop
                if self.state == AgentState.STOP or action == "Done":
                    # Verify success
                    dist = self._distance_to_target_object(obs, parsed.target)
                    print(f"[agent] STOP at step {step}, distance={dist:.2f}m")
                    return EpisodeResult(
                        success=dist <= self.approach_distance,
                        steps=step + 1,
                        path_length=self.path_length,
                        trajectory=self.trajectory,
                        final_distance=dist,
                        instruction=instruction,
                        scene=self.env.scene,
                    )

                # Execute action
                obs = self.env.step(action)

        finally:
            trace_file.close()

        # Ran out of steps
        dist = self._distance_to_target_object(obs, parsed.target)
        print(f"[agent] Ran out of steps ({max_steps}), distance={dist:.2f}m")
        return EpisodeResult(
            success=False,
            steps=max_steps,
            path_length=self.path_length,
            trajectory=self.trajectory,
            final_distance=dist,
            instruction=instruction,
            scene=self.env.scene,
        )

    def _reset_episode(self):
        """Reset internal episode state."""
        self.state = AgentState.EXPLORE
        self.step_count = 0
        self.path_length = 0.0
        self.prev_pos = None
        self.trajectory = []
        self.current_path = []
        self.path_index = 0
        self.target_world = None
        self.explore_rotations = 0
        self.no_detection_count = 0
        self.last_detections = []

    def _should_ground(self, step: int) -> bool:
        """Decide whether to run the grounder this step."""
        if self.state in (AgentState.GROUND, AgentState.APPROACH):
            return True  # every step when approaching
        if self.state == AgentState.EXPLORE:
            return True  # check on each rotation
        # During frontier search, ground every N steps
        return step % self.ground_every_n == 0

    def _decide_action(
        self, obs: Observation, detections: List[Detection], parsed: ParsedInstruction
    ) -> str:
        """
        Core state machine: pick the next action based on current state
        and observations.
        """
        pos = obs.pose["position"]
        yaw = obs.pose["rotation"]["y"]

        # Check if we have a strong detection
        best_det = detections[0] if detections else None
        has_detection = (best_det is not None
                         and best_det.score >= self.detection_threshold)

        # ---------- STATE: EXPLORE ----------
        if self.state == AgentState.EXPLORE:
            if has_detection:
                self._transition_to_ground(obs, best_det)
                return self._approach_action(obs)

            # Rotate 360° (8 rotations at 45°)
            self.explore_rotations += 1
            if self.explore_rotations >= 8:
                print(f"[agent] Explore complete, switching to SEARCH_FRONTIER")
                self.state = AgentState.SEARCH_FRONTIER
                return self._frontier_action(obs)
            return "RotateRight"

        # ---------- STATE: SEARCH_FRONTIER ----------
        if self.state == AgentState.SEARCH_FRONTIER:
            if has_detection:
                self._transition_to_ground(obs, best_det)
                return self._approach_action(obs)

            self.no_detection_count += 1
            return self._frontier_action(obs)

        # ---------- STATE: GROUND ----------
        if self.state == AgentState.GROUND:
            if has_detection:
                # Update target position from latest detection
                self._transition_to_ground(obs, best_det)

            if self.target_world is not None:
                dist = math.sqrt(
                    (pos["x"] - self.target_world[0]) ** 2
                    + (pos["z"] - self.target_world[1]) ** 2
                )
                if dist <= self.approach_distance and has_detection:
                    self.state = AgentState.STOP
                    return "Done"
                self.state = AgentState.APPROACH
                return self._approach_action(obs)

            # Lost detection — back to frontier
            self.no_detection_count += 1
            if self.no_detection_count > 10:
                self.state = AgentState.SEARCH_FRONTIER
                self.no_detection_count = 0
            return self._frontier_action(obs)

        # ---------- STATE: APPROACH ----------
        if self.state == AgentState.APPROACH:
            if has_detection:
                # Update target
                self._transition_to_ground(obs, best_det)

                dist = math.sqrt(
                    (pos["x"] - self.target_world[0]) ** 2
                    + (pos["z"] - self.target_world[1]) ** 2
                )
                if dist <= self.approach_distance:
                    self.state = AgentState.STOP
                    return "Done"
            else:
                self.no_detection_count += 1
                if self.no_detection_count > 50:
                    print("[agent] Lost target, returning to SEARCH_FRONTIER")
                    self.state = AgentState.SEARCH_FRONTIER
                    self.target_world = None
                    self.no_detection_count = 0
                    return self._frontier_action(obs)

            return self._approach_action(obs)

        # ---------- STATE: STOP ----------
        return "Done"

    def _transition_to_ground(self, obs: Observation, det: Detection):
        """Back-project detection bbox center to world coordinates and set goal."""
        x1, y1, x2, y2 = det.bbox
        cx_pixel = (x1 + x2) / 2.0
        cy_pixel = (y1 + y2) / 2.0

        H, W = obs.depth.shape
        # Clamp to valid range
        px = int(min(max(cx_pixel, 0), W - 1))
        py = int(min(max(cy_pixel, 0), H - 1))

        depth_val = obs.depth[py, px]
        if depth_val <= 0 or depth_val > 10.0 or not np.isfinite(depth_val):
            # Try average depth in bbox region
            region = obs.depth[int(y1):int(y2), int(x1):int(x2)]
            valid = region[(region > 0) & (region < 10.0) & np.isfinite(region)]
            if len(valid) == 0:
                return  # can't get depth, skip
            depth_val = float(np.median(valid))

        # Back-project to camera frame
        intr = obs.intrinsics
        X_c = (cx_pixel - intr["cx"]) * depth_val / intr["fx"]
        Z_c = depth_val

        # Rotate by yaw to world frame
        yaw_rad = math.radians(obs.pose["rotation"]["y"])
        world_x = X_c * math.cos(yaw_rad) + Z_c * math.sin(yaw_rad) + obs.pose["position"]["x"]
        world_z = -X_c * math.sin(yaw_rad) + Z_c * math.cos(yaw_rad) + obs.pose["position"]["z"]

        self.target_world = (world_x, world_z)
        self.state = AgentState.GROUND
        self.no_detection_count = 0

        # Plan path to target
        start = (obs.pose["position"]["x"], obs.pose["position"]["z"])
        result = self.mapper.plan_path(start, self.target_world)
        if result.success and len(result.path) > 1:
            self.current_path = result.path
            self.path_index = 1  # skip start
        else:
            self.current_path = []
            self.path_index = 0

    def _approach_action(self, obs: Observation) -> str:
        """Pick action to follow the planned path or move toward target."""
        pos = obs.pose["position"]
        yaw = obs.pose["rotation"]["y"]

        # Determine goal point
        if self.current_path and self.path_index < len(self.current_path):
            goal = self.current_path[self.path_index]
            dist_to_wp = math.sqrt(
                (pos["x"] - goal[0]) ** 2 + (pos["z"] - goal[1]) ** 2
            )
            if dist_to_wp < 0.3:
                self.path_index += 1
                if self.path_index >= len(self.current_path):
                    goal = self.target_world or goal
                else:
                    goal = self.current_path[self.path_index]
        elif self.target_world is not None:
            goal = self.target_world
        else:
            return "MoveAhead"

        return self._action_toward(pos, yaw, goal)

    def _frontier_action(self, obs: Observation) -> str:
        """Pick action to explore: navigate toward nearest frontier."""
        pos = obs.pose["position"]
        yaw = obs.pose["rotation"]["y"]

        # If we have a path, follow it
        if self.current_path and self.path_index < len(self.current_path):
            goal = self.current_path[self.path_index]
            dist_to_wp = math.sqrt(
                (pos["x"] - goal[0]) ** 2 + (pos["z"] - goal[1]) ** 2
            )
            if dist_to_wp < 0.3:
                self.path_index += 1
                if self.path_index >= len(self.current_path):
                    self.current_path = []
                    self.path_index = 0
                else:
                    return self._action_toward(pos, yaw,
                                                self.current_path[self.path_index])
            return self._action_toward(pos, yaw, goal)

        # No path — find frontier and plan
        frontiers = self.mapper.get_frontiers()
        if not frontiers:
            # No frontiers: random exploration
            return "MoveAhead"

        # Pick nearest frontier
        start = (pos["x"], pos["z"])
        nearest = min(frontiers,
                      key=lambda f: (f[0] - start[0])**2 + (f[1] - start[1])**2)
        result = self.mapper.plan_path(start, nearest)

        if result.success and len(result.path) > 1:
            self.current_path = result.path
            self.path_index = 1
            return self._action_toward(pos, yaw, self.current_path[self.path_index])

        # Fallback: move forward or rotate
        return "RotateRight"

    def _action_toward(
        self, pos: dict, yaw: float, goal: Tuple[float, float]
    ) -> str:
        """
        Pick a discrete action to move toward goal (world_x, world_z).

        Computes the desired heading, and if the agent is roughly facing
        the goal, moves ahead; otherwise rotates.
        """
        dx = goal[0] - pos["x"]
        dz = goal[1] - pos["z"]

        # Desired heading in degrees (0 = +Z, 90 = +X)
        desired_yaw = math.degrees(math.atan2(dx, dz)) % 360
        diff = (desired_yaw - yaw + 180) % 360 - 180  # in [-180, 180]

        rotate_step = self.env.rotate_step

        if abs(diff) < rotate_step / 2:
            return "MoveAhead"
        elif diff > 0:
            return "RotateRight"
        else:
            return "RotateLeft"

    def _distance_to_target_object(
        self, obs: Observation, target_type: str
    ) -> float:
        """
        Compute distance from agent to the nearest matching object
        using AI2-THOR metadata (ground truth).
        """
        pos = obs.pose["position"]
        objects = obs.metadata.get("objects", [])

        min_dist = float("inf")
        target_lower = target_type.lower()

        for obj in objects:
            obj_type = obj.get("objectType", "").lower()
            obj_name = obj.get("name", "").lower()

            if target_lower in obj_type or target_lower in obj_name:
                op = obj["position"]
                d = math.sqrt(
                    (pos["x"] - op["x"]) ** 2
                    + (pos["z"] - op["z"]) ** 2
                )
                min_dist = min(min_dist, d)

        return min_dist

    def _make_record(
        self, step: int, obs: Observation, action: str,
        detections: List[Detection]
    ) -> Dict[str, Any]:
        """Build a log record for the trace file."""
        top3 = [
            {"bbox": list(d.bbox), "score": round(d.score, 4), "label": d.label}
            for d in detections[:3]
        ]
        return {
            "step": step,
            "state": self.state.name,
            "action": action,
            "pose": {
                "x": round(obs.pose["position"]["x"], 3),
                "z": round(obs.pose["position"]["z"], 3),
                "yaw": round(obs.pose["rotation"]["y"], 1),
            },
            "query": self.query,
            "detections_top3": top3,
            "target_world": list(self.target_world) if self.target_world else None,
            "path_length": round(self.path_length, 3),
        }
