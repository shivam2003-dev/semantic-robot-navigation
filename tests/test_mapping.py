"""
tests/test_mapping.py — Unit tests for occupancy grid and A* planner.
"""

import math
import numpy as np
import pytest

from mapping import OccupancyMap, FREE, OCCUPIED, UNKNOWN, PathResult


class TestWorldGridConversion:
    """Coordinate conversion between world and grid frames."""

    def test_origin_maps_to_center(self):
        m = OccupancyMap(resolution=0.05, map_size=200)
        r, c = m.world_to_grid(0.0, 0.0)
        assert r == 100
        assert c == 100

    def test_roundtrip(self):
        m = OccupancyMap(resolution=0.05, map_size=200)
        x, z = 1.5, -2.0
        r, c = m.world_to_grid(x, z)
        x2, z2 = m.grid_to_world(r, c)
        assert abs(x2 - x) < m.resolution
        assert abs(z2 - z) < m.resolution

    def test_positive_offset(self):
        m = OccupancyMap(resolution=0.1, map_size=100)
        r, c = m.world_to_grid(1.0, 2.0)
        # 1.0 / 0.1 = 10 cells from origin
        assert c == 50 + 10
        assert r == 50 + 20


class TestAStarPlanning:
    """A* path planner tests on hand-crafted grids."""

    def _make_mapper_with_grid(self, grid: np.ndarray) -> OccupancyMap:
        """Helper: create OccupancyMap with a pre-set grid."""
        size = grid.shape[0]
        m = OccupancyMap(resolution=0.05, map_size=size, agent_radius=0.0)
        m.grid = grid.copy()
        m.origin = size // 2
        return m

    def test_straight_line_path(self):
        """Path on a clear grid should be roughly straight."""
        grid = np.full((20, 20), FREE, dtype=np.uint8)
        m = self._make_mapper_with_grid(grid)

        # Start at (0,0) → goal at (0.4, 0.4) in world coords
        result = m.plan_path((0.0, 0.0), (0.4, 0.4))
        assert result.success
        assert len(result.path) > 0
        assert result.cost < float("inf")

    def test_path_avoids_obstacle(self):
        """Path should go around a wall of obstacles."""
        grid = np.full((20, 20), FREE, dtype=np.uint8)
        # Place a vertical wall at col=12, rows 5-14
        grid[5:15, 12] = OCCUPIED
        m = self._make_mapper_with_grid(grid)

        # Start left of wall, goal right of wall
        start = m.grid_to_world(10, 8)
        goal = m.grid_to_world(10, 16)

        result = m.plan_path(start, goal)
        assert result.success

        # Verify no path cell is on the wall
        for wx, wz in result.path:
            r, c = m.world_to_grid(wx, wz)
            assert grid[r, c] != OCCUPIED

    def test_start_equals_goal(self):
        """Path from a point to itself should succeed with zero cost."""
        grid = np.full((20, 20), FREE, dtype=np.uint8)
        m = self._make_mapper_with_grid(grid)
        result = m.plan_path((0.0, 0.0), (0.0, 0.0))
        assert result.success
        assert result.cost == 0.0

    def test_no_path_exists(self):
        """If start and goal are in disconnected free regions, planning should fail."""
        grid = np.full((40, 40), OCCUPIED, dtype=np.uint8)
        # Two disconnected free islands separated by a wide obstacle band
        grid[3:7, 3:7] = FREE     # island A (start)
        grid[33:37, 33:37] = FREE  # island B (goal)
        m = self._make_mapper_with_grid(grid)

        start = m.grid_to_world(5, 5)
        goal = m.grid_to_world(35, 35)

        result = m.plan_path(start, goal)
        assert not result.success

    def test_path_cost_is_reasonable(self):
        """Path cost should be roughly proportional to distance."""
        grid = np.full((40, 40), FREE, dtype=np.uint8)
        m = self._make_mapper_with_grid(grid)

        start = m.grid_to_world(10, 10)
        goal = m.grid_to_world(30, 30)

        result = m.plan_path(start, goal)
        assert result.success
        # Euclidean distance in cells ≈ sqrt(20² + 20²) ≈ 28.3
        # A* diagonal cost should be close: 20 * sqrt(2) ≈ 28.3
        euclidean = math.sqrt(20**2 + 20**2)
        assert result.cost < euclidean * 1.5  # generous bound


class TestFrontierDetection:
    """Frontier (exploration boundary) detection."""

    def test_finds_frontiers(self):
        m = OccupancyMap(resolution=0.05, map_size=50)
        # Mark a small free region; everything else is unknown → frontiers
        m.grid[20:30, 20:30] = FREE
        frontiers = m.get_frontiers(min_cluster_size=1)
        assert len(frontiers) > 0

    def test_no_frontiers_fully_explored(self):
        m = OccupancyMap(resolution=0.05, map_size=20)
        # Mark everything as free (no unknown neighbors)
        m.grid[:] = FREE
        frontiers = m.get_frontiers()
        # Should be empty since there's nothing unknown adjacent
        assert len(frontiers) == 0


class TestOccupancyUpdate:
    """Occupancy grid update from synthetic depth data."""

    def test_update_marks_cells(self):
        """Feeding a synthetic depth image should mark some cells."""
        m = OccupancyMap(resolution=0.05, map_size=200)
        H, W = 60, 80

        # Synthetic depth: all at 2.0 meters
        depth = np.full((H, W), 2.0, dtype=np.float32)

        pose = {
            "position": {"x": 0.0, "y": 0.9, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "horizon": 0.0,
        }
        intrinsics = {"fx": 40.0, "fy": 40.0, "cx": 40.0, "cy": 30.0}

        m.update(depth, pose, intrinsics)

        # At least some cells should now be FREE or OCCUPIED
        assert np.sum(m.grid == FREE) + np.sum(m.grid == OCCUPIED) > 0
