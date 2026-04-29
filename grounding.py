"""
grounding.py — Vision-Language Grounding with OWL-ViT + CLIP

Detects objects in RGB frames using open-vocabulary detection (OWL-ViT)
and re-ranks detections using CLIP similarity to the query phrase.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image


@dataclass
class Detection:
    """A single object detection."""
    bbox: tuple   # (x1, y1, x2, y2) in pixel coordinates
    score: float  # combined confidence score
    label: str    # text label


def _get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _iou(box1, box2) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """Non-maximum suppression on detections sorted by score."""
    if not detections:
        return []
    # Sort by score descending
    dets = sorted(detections, key=lambda d: d.score, reverse=True)
    keep = []
    for det in dets:
        suppressed = False
        for kept in keep:
            if _iou(det.bbox, kept.bbox) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)
    return keep


class VLGrounder:
    """Vision-Language grounding using OWL-ViT detection + CLIP re-ranking."""

    def __init__(
        self,
        owlvit_model: str = "google/owlvit-base-patch32",
        clip_model: str = "ViT-B/32",
        detection_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.device = device or _get_device()
        self.detection_threshold = detection_threshold
        self.nms_threshold = nms_threshold

        print(f"[grounding] Loading models on {self.device}...")

        # Load OWL-ViT
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        self.owlvit_processor = OwlViTProcessor.from_pretrained(owlvit_model)
        self.owlvit_model = OwlViTForObjectDetection.from_pretrained(owlvit_model)
        self.owlvit_model.to(self.device)
        self.owlvit_model.eval()

        # Load CLIP
        import clip as clip_module
        self.clip_module = clip_module
        self.clip_model, self.clip_preprocess = clip_module.load(
            clip_model, device=self.device
        )
        self.clip_model.eval()

        print("[grounding] Models loaded.")

    @torch.no_grad()
    def score_frame(
        self,
        rgb: np.ndarray,
        query: str,
        top_k: int = 10,
    ) -> List[Detection]:
        """
        Detect and rank objects matching the query in an RGB frame.

        Args:
            rgb: (H, W, 3) uint8 numpy array
            query: natural language description (e.g., "red mug")
            top_k: maximum detections to return

        Returns:
            List of Detection objects sorted by score descending.
        """
        pil_image = Image.fromarray(rgb)
        H, W = rgb.shape[:2]

        # --- Stage 1: OWL-ViT open-vocabulary detection ---
        text_queries = [
            [f"a photo of a {query}", f"a {query}", query]
        ]

        inputs = self.owlvit_processor(
            text=text_queries, images=pil_image, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.owlvit_model(**inputs)

        target_sizes = torch.tensor([[H, W]], device=self.device)
        results = self.owlvit_processor.post_process_object_detection(
            outputs, threshold=self.detection_threshold, target_sizes=target_sizes
        )[0]

        boxes = results["boxes"].cpu().numpy()    # (N, 4) x1,y1,x2,y2
        scores = results["scores"].cpu().numpy()  # (N,)

        if len(boxes) == 0:
            return []

        # --- Stage 2: CLIP re-ranking ---
        # Encode the full query phrase
        text_tokens = self.clip_module.tokenize([f"a photo of {query}"]).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            # Clamp to image bounds
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue  # skip tiny boxes

            # Crop and encode with CLIP
            crop = pil_image.crop((x1, y1, x2, y2))
            crop_tensor = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(crop_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            clip_score = (image_features @ text_features.T).item()

            # Combined score: geometric mean of OWL-ViT and CLIP scores
            combined = float(np.sqrt(max(0, scores[i]) * max(0, clip_score)))

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                score=combined,
                label=query,
            ))

        # NMS and sort
        detections = _nms(detections, self.nms_threshold)
        return detections[:top_k]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python grounding.py <image_path> <query>")
        sys.exit(1)

    img = np.array(Image.open(sys.argv[1]).convert("RGB"))
    query = sys.argv[2]

    grounder = VLGrounder()
    dets = grounder.score_frame(img, query)
    for d in dets:
        print(f"  bbox={d.bbox}, score={d.score:.3f}, label={d.label}")
