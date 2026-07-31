"""
tools/detector.py

Perception tool for the PPE Compliance Agent.
Wraps a YOLOv8 model and converts raw Ultralytics output into a clean,
structured data format the reasoning stage can consume — never loose
prints or raw tensor objects passed around.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import os

from ultralytics import YOLO


# Default weight locations, tried in order. The fine-tuned PPE model is
# preferred; if it isn't present (e.g. fresh clone without the large weight
# file downloaded yet), the agent falls back to base YOLOv8n so the pipeline
# still runs end-to-end rather than crashing.
DEFAULT_WEIGHTS_CANDIDATES = [
    "models/trained/best_yolov8n_ppe.pt",
    "models/best_yolov8n_ppe.pt",
    "yolov8n.pt",  # generic fallback (auto-downloads via ultralytics)
]


@dataclass
class Detection:
    """A single structured detection from the perception stage."""
    class_name: str
    confidence: float
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] in pixel coordinates

    def to_dict(self):
        return asdict(self)


class PPEDetector:
    """Loads a YOLOv8 model and exposes a single .detect() perception call."""

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self.weights_path, self.using_fallback = self._resolve_weights(weights_path)
        self.model = YOLO(self.weights_path)

    def _resolve_weights(self, weights_path):
        candidates = [weights_path] if weights_path else DEFAULT_WEIGHTS_CANDIDATES
        for c in candidates:
            if c is None:
                continue
            if c == "yolov8n.pt" or os.path.exists(c):
                is_fallback = c == "yolov8n.pt"
                return c, is_fallback
        # nothing found on disk — fall back to auto-downloaded base weights
        return "yolov8n.pt", True

    def detect(self, image) -> List[Detection]:
        """
        Run perception on a single image (file path or numpy array).
        Returns a list of structured Detection objects — this is the
        contract between the perception stage and the reasoning stage.
        """
        results = self.model.predict(source=image, conf=self.conf_threshold, verbose=False)
        detections = []
        r = results[0]
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                Detection(
                    class_name=names[cls_id],
                    confidence=round(float(box.conf[0]), 4),
                    bbox_xyxy=[round(v, 1) for v in box.xyxy[0].tolist()],
                )
            )
        return detections, r  # also return raw result for annotation/plotting
