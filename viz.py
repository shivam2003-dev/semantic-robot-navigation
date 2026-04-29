"""
viz.py — Navigation Visualization

Provides a two-panel matplotlib display:
  Left:  RGB frame with bounding box overlay
  Right: Top-down occupancy grid with path, agent pose, and target
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend by default
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from typing import List, Dict, Optional, Tuple
import os
import math

from mapping import OccupancyMap, UNKNOWN, FREE, OCCUPIED


class NavigationVisualizer:
    """Live or replay visualization of navigation episodes."""

    def __init__(self, mapper: OccupancyMap, save_dir: str = "viz_frames"):
        self.mapper = mapper
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.frame_count = 0

    def render_frame(
        self,
        rgb: Optional[np.ndarray] = None,
        detections: Optional[List[Dict]] = None,
        agent_pos: Optional[Tuple[float, float]] = None,
        agent_yaw: float = 0.0,
        target_pos: Optional[Tuple[float, float]] = None,
        path: Optional[List[Tuple[float, float]]] = None,
        step: int = 0,
        state: str = "",
        save: bool = True,
    ) -> np.ndarray:
        """
        Render a single visualization frame.

        Args:
            rgb: (H, W, 3) RGB image, or None
            detections: list of {bbox: [x1,y1,x2,y2], score: float, label: str}
            agent_pos: (world_x, world_z)
            agent_yaw: degrees
            target_pos: (world_x, world_z)
            path: list of (world_x, world_z) waypoints
            step: current step number
            state: current agent state name
            save: whether to save frame to disk

        Returns:
            Rendered frame as numpy array (H, W, 3).
        """
        fig, (ax_rgb, ax_map) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Left panel: RGB with detections ---
        if rgb is not None:
            ax_rgb.imshow(rgb)
            if detections:
                for det in detections[:5]:
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    score = det.get("score", 0)
                    label = det.get("label", "")
                    x1, y1, x2, y2 = bbox
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor="lime", facecolor="none"
                    )
                    ax_rgb.add_patch(rect)
                    ax_rgb.text(
                        x1, y1 - 5, f"{label} {score:.2f}",
                        color="lime", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
                    )
        else:
            ax_rgb.text(0.5, 0.5, "No RGB", transform=ax_rgb.transAxes,
                       ha="center", va="center", fontsize=16)

        ax_rgb.set_title(f"Step {step} | State: {state}", fontsize=12)
        ax_rgb.axis("off")

        # --- Right panel: Occupancy map ---
        grid_img = self.mapper.get_grid_image()
        ax_map.imshow(grid_img, origin="lower")

        # Draw path
        if path:
            path_rows, path_cols = [], []
            for wx, wz in path:
                r, c = self.mapper.world_to_grid(wx, wz)
                path_rows.append(r)
                path_cols.append(c)
            ax_map.plot(path_cols, path_rows, "g-", linewidth=1.5, alpha=0.8)

        # Draw agent
        if agent_pos is not None:
            ar, ac = self.mapper.world_to_grid(agent_pos[0], agent_pos[1])
            ax_map.plot(ac, ar, "ro", markersize=8, zorder=5)
            # Arrow showing heading
            yaw_rad = math.radians(agent_yaw)
            dx = math.sin(yaw_rad) * 8
            dy = math.cos(yaw_rad) * 8
            ax_map.annotate("", xy=(ac + dx, ar + dy), xytext=(ac, ar),
                          arrowprops=dict(arrowstyle="->", color="red", lw=2))

        # Draw target
        if target_pos is not None:
            tr, tc = self.mapper.world_to_grid(target_pos[0], target_pos[1])
            ax_map.plot(tc, tr, "b*", markersize=15, zorder=5)

        # Zoom to relevant area
        occupied_cells = np.argwhere(self.mapper.grid != UNKNOWN)
        if len(occupied_cells) > 0:
            min_r, min_c = occupied_cells.min(axis=0)
            max_r, max_c = occupied_cells.max(axis=0)
            pad = 20
            ax_map.set_xlim(max(0, min_c - pad), min(self.mapper.map_size, max_c + pad))
            ax_map.set_ylim(max(0, min_r - pad), min(self.mapper.map_size, max_r + pad))

        ax_map.set_title("Occupancy Map", fontsize=12)
        ax_map.set_aspect("equal")

        plt.tight_layout()

        # Convert figure to array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if save:
            path_out = os.path.join(self.save_dir, f"frame_{self.frame_count:04d}.png")
            fig.savefig(path_out, dpi=100, bbox_inches="tight")
            self.frame_count += 1

        plt.close(fig)
        return img

    def replay_trajectory(
        self,
        trajectory: List[Dict],
        mapper: OccupancyMap,
    ):
        """Replay a trajectory and generate visualization frames."""
        self.mapper = mapper
        print(f"[viz] Rendering {len(trajectory)} frames to {self.save_dir}/")

        for record in trajectory:
            self.render_frame(
                rgb=None,  # RGB not stored in trace; render map only
                detections=record.get("detections_top3", []),
                agent_pos=(record["pose"]["x"], record["pose"]["z"]),
                agent_yaw=record["pose"]["yaw"],
                target_pos=tuple(record["target_world"]) if record.get("target_world") else None,
                step=record["step"],
                state=record.get("state", ""),
                save=True,
            )

        print(f"[viz] Saved {self.frame_count} frames.")

    def make_gif(self, output_path: str = "demo.gif", fps: int = 5):
        """Combine saved frames into an animated GIF."""
        import imageio

        frames = []
        for i in range(self.frame_count):
            path = os.path.join(self.save_dir, f"frame_{i:04d}.png")
            if os.path.exists(path):
                frames.append(imageio.imread(path))

        if frames:
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
            print(f"[viz] GIF saved to {output_path} ({len(frames)} frames, {fps} fps)")
        else:
            print("[viz] No frames to assemble into GIF.")


# ---------------------------------------------------------------------------
# Standalone usage: replay from trace file
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python viz.py <trace.jsonl> [output.gif]")
        sys.exit(1)

    trace_path = sys.argv[1]
    gif_path = sys.argv[2] if len(sys.argv) > 2 else "demo.gif"

    # Load trajectory
    trajectory = []
    with open(trace_path) as f:
        for line in f:
            trajectory.append(json.loads(line))

    # Create a mapper (won't have actual occupancy data, but that's OK for replay)
    mapper = OccupancyMap()

    viz = NavigationVisualizer(mapper, save_dir="viz_replay")
    viz.replay_trajectory(trajectory, mapper)
    viz.make_gif(gif_path)
