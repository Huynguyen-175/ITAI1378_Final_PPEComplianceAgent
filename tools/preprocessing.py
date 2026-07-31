"""
tools/preprocessing.py

Preprocessing stage for the PPE Compliance Agent.
Validates and prepares inputs before they reach the perception stage.
Bad inputs (corrupt files, unreadable images, empty/near-empty images)
are caught and reported here — they must never crash the agent.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BLUR_VARIANCE_THRESHOLD = 15.0  # below this, flag as likely too blurry to be useful


@dataclass
class PreprocessResult:
    valid: bool
    image_array: Optional[np.ndarray]
    reason: str  # "ok", or why it failed/was flagged
    warning: Optional[str] = None  # non-fatal issue, e.g. "low sharpness"


def validate_and_load(image_path: str) -> PreprocessResult:
    """
    Validate a single image path and load it if usable.
    Never raises — always returns a PreprocessResult so the agent can
    continue processing the rest of a batch even if one file is bad.
    """
    path = Path(image_path)

    if not path.exists():
        return PreprocessResult(False, None, f"file not found: {image_path}")

    if path.suffix.lower() not in VALID_EXTENSIONS:
        return PreprocessResult(False, None, f"unsupported file type: {path.suffix}")

    # Verify it's a genuine, non-corrupt image
    try:
        with Image.open(path) as im:
            im.verify()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return PreprocessResult(False, None, f"corrupt or unreadable image: {e}")

    # Reload after verify() (which leaves the file unusable for further ops)
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return PreprocessResult(False, None, "failed to load image data")

    h, w = img_bgr.shape[:2]
    if h < 32 or w < 32:
        return PreprocessResult(False, None, f"image too small ({w}x{h}) to be useful")

    # Non-fatal quality check: blur detection via variance of Laplacian
    warning = None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < BLUR_VARIANCE_THRESHOLD:
        warning = f"low sharpness ({sharpness:.1f}) — image may be too blurry for reliable detection"

    return PreprocessResult(True, img_bgr, "ok", warning=warning)
