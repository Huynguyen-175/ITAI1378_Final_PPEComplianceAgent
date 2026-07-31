# Models

Trained model weights are **not committed to this repository** (binary files, ~6 MB+, and per course guidance large files should not be pushed to GitHub).

## Trained Model

| File | Description | Metrics |
|---|---|---|
| `trained/best_yolov8n_ppe.pt` | YOLOv8n fine-tuned on the PPE mask-detection dataset, 75 epochs | mAP@0.5: 90.0% · Precision: 93.4% · Recall: 82.5% |

### How to get it

1. Run `notebooks/01_exploration.ipynb` in Google Colab (trains the model from scratch, ~76 min on a T4 GPU), **or**
2. Download the already-trained weights from Google Drive: *[add your Drive share link here]*, then place the file at:
   ```
   models/trained/best_yolov8n_ppe.pt
   ```

### Fallback behavior

If `best_yolov8n_ppe.pt` is not present when the agent runs, `tools/detector.py` automatically falls back to base `yolov8n.pt` (generic COCO weights, auto-downloaded by Ultralytics on first use). This means **the pipeline always runs end-to-end**, even without the trained weights — though detections will reflect generic object classes (person, etc.) rather than mask/no-mask classes until the real weights are in place. A warning is printed when this fallback is active.

## Pretrained Base Model

`yolov8n.pt` — Ultralytics' base YOLOv8-nano checkpoint, auto-downloaded on first use. No manual step required.
