# Data

## Sample Data (`data/sample/`)

3 real caregiver photos used during agent development and demo testing, included directly in this repo so the agent can be run immediately after cloning with no external downloads:

| File | Ground Truth | Notes |
|---|---|---|
| `caregiver_car_nomask.jpg` | Non-compliant (no mask) | Vehicle interior, side lighting |
| `caregiver_home_nomask.jpg` | Non-compliant (no mask) | Indoor home setting |
| `caregiver_clinic_masked.jpg` | Compliant (2 people, both masked) | Clinical setting, multi-person frame |

Run the agent on these directly:
```bash
python agents/ppe_compliance_agent.py --input data/sample
```

> **Note:** these 3 images were pulled from the project's own Gradio demo sessions. For a fuller evaluation set (10+ scenarios, required for `notebooks/02_evaluation.ipynb`), pull additional images from the full training dataset's `test/` split (see below) — the label files there give you ground truth for free.

## Full Training Dataset

- **Source:** Roboflow Universe — "Mask-Detection-YOLOv8" by AGH
  https://universe.roboflow.com/agh-ett2f/mask-detection-yolov8
- **Size:** 4,547 labeled images (train/valid/test split)
- **Classes:** `with_mask`, `without_mask`, `incorrectly_worn_mask`
- **License:** CC BY 4.0
- **Download:** via the Roboflow Python package (see `notebooks/01_exploration.ipynb`, Section 3) — not committed to this repo due to size.

```python
import roboflow
roboflow.login()
rf = roboflow.Roboflow()
project = rf.workspace("agh-ett2f").project("mask-detection-yolov8")
dataset = project.version(16).download("yolov8")  # confirm current version on the Roboflow page
```
