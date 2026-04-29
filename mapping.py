"""
mapping.py — Occupancy Grid Mapping and A* Path Planning

Builds a 2D top-down occupancy grid from depth observations, detects
exploration frontiers, and plans paths using A* with 8-connectivity.
"""

import math
import heapq
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.ndimage import binary_dilation


# Cell states
UNKNOWN = 0
FREE = 1
OCCUPIED = 2


@dataclass
class PathResult:
    """Result of path planning."""
    path: List[Tuple[float, float]]   # list of (world_x, world_z)
    cost: float
    success: bool


class OccupancyMap:
    """2D top-down occupancy grid built from depth observations."""

    def __init__(
        self,
        resolution: float = 0.05,     # meters per cell
        map_size: int = 400,           # cells per side → 20m x 20m
        agent_radius: float = 0.125,   # meters, for obstacle inflation
        height_min: float = 0.1,       # min height for obstacle detection
        height_max: float = 1.8,       # max height for obstacle detection
        floor_height: float = 0.05,    # below this → free space
    ):
        self.resolution = resolution
        self.map_size = map_size
        self.agent_radius = agent_radius
        self.height_min = height_min
        self.height_max = height_max
        self.floor_height = floor_height

        # Grid centered at origin; origin at map_size//2
        self.grid = np.zeros((map_size, map_size), dtype=np.uint8)  # UNKNOWN
        self.origin = map_size // 2  # grid index of world (0, 0)

        # Inflation radius in cells
        self.inflate_cells = max(1, int(math.ceil(agent_radius / resolution)))

    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        """Convert world coordinates (x, z) to grid indices (row, col)."""
        col = int(round(x / self.resolution)) + self.origin
        row = int(round(z / self.resolution)) + self.origin
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices (row, col) to world coordinates (x, z)."""
        x = (col - self.origin) * self.resolution
        z = (row - self.origin) * self.resolution
        return x, z

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.map_size and 0 <= col < self.map_size

    def update(self, depth: np.ndarray, pose: dict, intrinsics: dict):
        """
        Update the occupancy grid from a depth frame.

        Args:
            depth: (H, W) float32 depth in meters
            pose: {position: {x,y,z}, rotation: {x,y,z}, horizon: float}
            intrinsics: {fx, fy, cx, cy}
        """
        H, W = depth.shape
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        # Down-sample for speed: use every 4th pixel
        stride = 4
        us = np.arange(0, W, stride, dtype=np.float32)
        vs = np.arange(0, H, stride, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(us, vs)  # (H', W')

        d = depth[::stride, ::stride].copy()  # (H', W')

        # Filter out invalid depth
        valid = (d > 0.1) & (d < 10.0) & np.isfinite(d)
        if not np.any(valid):
            return

        # Back-project to camera frame: X_c right, Y_c down, Z_c forward
        X_c = (u_grid - cx) * d / fx
        Y_c = (v_grid - cy) * d / fy
        Z_c = d

        # Apply camera pitch (horizon)
        horizon_rad = math.radians(pose.get("horizon", 0.0))
        cos_p = math.cos(horizon_rad)
        sin_p = math.sin(horizon_rad)
        # Rotate around X axis (pitch): Y' = Y*cos - Z*sin, Z' = Y*sin + Z*cos
        Y_rot = Y_c * cos_p - Z_c * sin_p
        Z_rot = Y_c * sin_p + Z_c * cos_p

        # Apply yaw rotation to transform from camera to world frame
        # AI2-THOR: yaw 0 = +Z, 90 = +X (right-hand around Y-up)
        yaw_deg = pose["rotation"]["y"]
        yaw_rad = math.radians(yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        # World coordinates: rotate (X_c, Z_rot) by yaw, then translate
        world_x = X_c * cos_y + Z_rot * sin_y + pose["position"]["x"]
        world_z = -X_c * sin_y + Z_rot * cos_y + pose["position"]["z"]
        world_y = -Y_rot + pose["position"]["y"]  # camera Y is down, world Y is up

        # Classify points by height
        obstacle_mask = valid & (world_y > self.height_min) & (world_y < self.height_max)
        free_mask = valid & (world_y <= self.floor_height)

        # Update grid: obstacles
        if np.any(obstacle_mask):
            ox = world_x[obstacle_mask]
            oz = world_z[obstacle_mask]
            for xi, zi in zip(ox.ravel(), oz.ravel()):
                r, c = self.world_to_grid(xi, zi)
                if self._in_bounds(r, c):
                    self.grid[r, c] = OCCUPIED

        # Update grid: free space (don't overwrite obstacles)
        if np.any(free_mask):
            fx_pts = world_x[free_mask]
            fz_pts = world_z[free_mask]
            for xi, zi in zip(fx_pts.ravel(), fz_pts.ravel()):
                r, c = self.world_to_grid(xi, zi)
                if self._in_bounds(r, c) and self.grid[r, c] != OCCUPIED:
                    self.grid[r, c] = FREE

        # Mark agent position as free
        ar, ac = self.world_to_grid(pose["position"]["x"], pose["position"]["z"])
        if self._in_bounds(ar, ac):
            self.grid[ar, ac] = FREE

        # Raycast from agent to nearby free cells for connectivity
        self._mark_free_rays(pose["position"]["x"], pose["position"]["z"],
                             world_x[valid], world_z[valid])

    def _mark_free_rays(self, ax: float, az: float,
                        pts_x: np.ndarray, pts_z: np.ndarray):
        """Mark cells along rays from agent to observed free points as free."""
        ar, ac = self.world_to_grid(ax, az)

        # Sub-sample points for speed
        n = min(200, len(pts_x.ravel()))
        if n == 0:
            return
        indices = np.random.choice(len(pts_x.ravel()), n, replace=False)
        for idx in indices:
            er, ec = self.world_to_grid(pts_x.ravel()[idx], pts_z.ravel()[idx])
            # Bresenham-like ray
            for r, c in self._bresenham(ar, ac, er, ec):
                if not self._in_bounds(r, c):
                    break
                if self.grid[r, c] == OCCUPIED:
                    break  # stop at obstacle
                if self.grid[r, c] == UNKNOWN:
                    self.grid[r, c] = FREE

    @staticmethod
    def _bresenham(r0, c0, r1, c1):
        """Generate cells along a line from (r0,c0) to (r1,c1)."""
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            cells.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return cells

    def get_inflated_grid(self) -> np.ndarray:
        """Return a binary obstacle grid with inflation for planning."""
        obstacle = (self.grid == OCCUPIED)
        struct = np.ones((2 * self.inflate_cells + 1, 2 * self.inflate_cells + 1))
        inflated = binary_dilation(obstacle, structure=struct).astype(np.uint8)
        return inflated

    def get_frontiers(self, min_cluster_size: int = 3) -> List[Tuple[float, float]]:
        """
        Find frontier cells (free cells adjacent to unknown) and return
        cluster centroids in world coordinates.
        """
        free_mask = (self.grid == FREE)
        unknown_mask = (self.grid == UNKNOWN)

        # A free cell is a frontier if it has at least one unknown neighbor
        frontier_mask = np.zeros_like(free_mask)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(np.roll(unknown_mask, dr, axis=0), dc, axis=1)
                frontier_mask |= (free_mask & shifted)

        # Find frontier cell coordinates
        frontier_cells = np.argwhere(frontier_mask)  # (N, 2) -> (row, col)
        if len(frontier_cells) == 0:
            return []

        # Simple clustering: grid-based grouping
        from scipy.ndimage import label
        labeled, n_clusters = label(frontier_mask)

        centroids = []
        for i in range(1, n_clusters + 1):
            cluster_cells = np.argwhere(labeled == i)
            if len(cluster_cells) < min_cluster_size:
                continue
            centroid_r = cluster_cells[:, 0].mean()
            centroid_c = cluster_cells[:, 1].mean()
            wx, wz = self.grid_to_world(int(centroid_r), int(centroid_c))
            centroids.append((wx, wz))

        return centroids

    def plan_path(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> PathResult:
        """
        A* path planning from start to goal (world coordinates).

        Args:
            start: (world_x, world_z)
            goal: (world_x, world_z)

        Returns:
            PathResult with path in world coordinates.
        """
        sr, sc = self.world_to_grid(start[0], start[1])
        gr, gc = self.world_to_grid(goal[0], goal[1])

        # Clamp to grid bounds
        sr = max(0, min(self.map_size - 1, sr))
        sc = max(0, min(self.map_size - 1, sc))
        gr = max(0, min(self.map_size - 1, gr))
        gc = max(0, min(self.map_size - 1, gc))

        if (sr, sc) == (gr, gc):
            wx, wz = self.grid_to_world(sr, sc)
            return PathResult(path=[(wx, wz)], cost=0.0, success=True)

        inflated = self.get_inflated_grid()

        # If goal is in obstacle, find nearest free cell
        if inflated[gr, gc]:
            gr, gc = self._nearest_free(gr, gc, inflated)
            if gr is None:
                return PathResult(path=[], cost=float("inf"), success=False)

        # If start is in obstacle, find nearest free cell
        if inflated[sr, sc]:
            sr, sc = self._nearest_free(sr, sc, inflated)
            if sr is None:
                return PathResult(path=[], cost=float("inf"), success=False)

        path_cells, cost = self._astar(sr, sc, gr, gc, inflated)

        if path_cells is None:
            return PathResult(path=[], cost=float("inf"), success=False)

        # Convert to world coordinates
        path_world = [self.grid_to_world(r, c) for r, c in path_cells]
        return PathResult(path=path_world, cost=cost, success=True)

    def _nearest_free(self, r, c, inflated, max_radius=20):
        """Find nearest non-inflated cell."""
        for radius in range(1, max_radius):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if self._in_bounds(nr, nc) and not inflated[nr, nc]:
                        return nr, nc
        return None, None

    def _astar(
        self, sr: int, sc: int, gr: int, gc: int, inflated: np.ndarray
    ) -> Tuple[Optional[List[Tuple[int, int]]], float]:
        """
        A* search with 8-connectivity on the grid.

        Returns (path_cells, cost) or (None, inf) if no path found.
        """
        # 8-connected neighbors with costs
        SQRT2 = math.sqrt(2)
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
        ]

        def heuristic(r, c):
            return math.sqrt((r - gr) ** 2 + (c - gc) ** 2)

        # Priority queue: (f_cost, counter, row, col)
        counter = 0
        open_set = [(heuristic(sr, sc), counter, sr, sc)]
        g_cost = {(sr, sc): 0.0}
        came_from = {}
        closed = set()

        while open_set:
            f, _, r, c = heapq.heappop(open_set)

            if (r, c) in closed:
                continue
            closed.add((r, c))

            if r == gr and c == gc:
                # Reconstruct path
                path = [(r, c)]
                while (r, c) in came_from:
                    r, c = came_from[(r, c)]
                    path.append((r, c))
                path.reverse()
                return path, g_cost[(gr, gc)]

            for dr, dc, move_cost in neighbors:
                nr, nc = r + dr, c + dc
                if not self._in_bounds(nr, nc):
                    continue
                if (nr, nc) in closed:
                    continue
                if inflated[nr, nc]:
                    continue

                # Penalize unknown cells slightly
                extra = 0.5 if self.grid[nr, nc] == UNKNOWN else 0.0
                new_g = g_cost[(r, c)] + move_cost + extra

                if new_g < g_cost.get((nr, nc), float("inf")):
                    g_cost[(nr, nc)] = new_g
                    f_new = new_g + heuristic(nr, nc)
                    came_from[(nr, nc)] = (r, c)
                    counter += 1
                    heapq.heappush(open_set, (f_new, counter, nr, nc))

        return None, float("inf")

    def get_grid_image(self) -> np.ndarray:
        """Return an RGB visualization of the occupancy grid."""
        img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        img[self.grid == UNKNOWN] = [40, 40, 40]      # dark gray
        img[self.grid == FREE] = [200, 200, 200]       # light gray
        img[self.grid == OCCUPIED] = [0, 0, 0]          # black
        return img
