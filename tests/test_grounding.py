"""
tests/test_grounding.py — Unit tests for VLGrounder.

Tests the grounding pipeline interface. Since loading full models is
expensive, we test the utility functions (NMS, IoU) directly and mock
the model-based tests when models are not available.
"""

import pytest
import numpy as np

from grounding import Detection, _iou, _nms


class TestIoU:
    """Intersection-over-Union computation."""

    def test_identical_boxes(self):
        box = (10, 10, 50, 50)
        assert abs(_iou(box, box) - 1.0) < 1e-6

    def test_no_overlap(self):
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        assert _iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        box1 = (0, 0, 20, 20)
        box2 = (10, 10, 30, 30)
        # Intersection: (10,10)-(20,20) = 10*10 = 100
        # Union: 400 + 400 - 100 = 700
        expected = 100.0 / 700.0
        assert abs(_iou(box1, box2) - expected) < 1e-5

    def test_contained_box(self):
        outer = (0, 0, 100, 100)
        inner = (25, 25, 75, 75)
        # Intersection = 50*50 = 2500, union = 10000
        expected = 2500.0 / 10000.0
        assert abs(_iou(outer, inner) - expected) < 1e-5


class TestNMS:
    """Non-maximum suppression."""

    def test_empty_input(self):
        assert _nms([]) == []

    def test_single_detection(self):
        det = Detection(bbox=(10, 10, 50, 50), score=0.9, label="apple")
        result = _nms([det])
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_suppresses_overlapping(self):
        d1 = Detection(bbox=(10, 10, 50, 50), score=0.9, label="apple")
        d2 = Detection(bbox=(12, 12, 52, 52), score=0.7, label="apple")
        result = _nms([d1, d2], iou_threshold=0.5)
        # d2 should be suppressed (high IoU with d1)
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_keeps_non_overlapping(self):
        d1 = Detection(bbox=(0, 0, 20, 20), score=0.9, label="apple")
        d2 = Detection(bbox=(100, 100, 120, 120), score=0.8, label="mug")
        result = _nms([d1, d2], iou_threshold=0.5)
        assert len(result) == 2

    def test_ordering_by_score(self):
        d1 = Detection(bbox=(0, 0, 50, 50), score=0.5, label="a")
        d2 = Detection(bbox=(200, 200, 250, 250), score=0.9, label="b")
        d3 = Detection(bbox=(100, 100, 150, 150), score=0.7, label="c")
        result = _nms([d1, d2, d3])
        assert result[0].score == 0.9
        assert result[1].score == 0.7
        assert result[2].score == 0.5


class TestDetectionDataclass:
    """Detection dataclass basics."""

    def test_creation(self):
        det = Detection(bbox=(10, 20, 30, 40), score=0.85, label="mug")
        assert det.bbox == (10, 20, 30, 40)
        assert det.score == 0.85
        assert det.label == "mug"


def _can_load_models() -> bool:
    """Check if we can load the grounding models."""
    try:
        import torch
        import transformers
        import clip
        return True
    except ImportError:
        return False


class TestGrounderIntegration:
    """Integration test — only runs if models can be loaded."""

    @pytest.mark.skipif(
        not _can_load_models(),
        reason="VLM models not available (no GPU or models not cached)",
    )
    def test_score_frame_returns_detections(self):
        from grounding import VLGrounder
        grounder = VLGrounder()
        # Create a simple test image (solid color — won't detect much)
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = grounder.score_frame(rgb, "apple")
        assert isinstance(dets, list)
        for d in dets:
            assert isinstance(d, Detection)
