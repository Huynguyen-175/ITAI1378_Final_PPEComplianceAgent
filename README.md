# PPE Compliance Agent

*An agent that watches for missing PPE in caregiver photos and reports it — so supervisors don't have to check by hand.*

## Author

Huy Nguyen – ITAI 1378, Summer 2026

## Project Tier

**Tier 1** – Single agent, one CV capability (YOLOv8n object detection), rule-based reasoning loop (`detect → decide → report`). Chosen to guarantee a fully working, well-evaluated, well-documented pipeline within the timeline, rather than risk a shakier Tier 2 build for a small bonus.

## Problem & Solution

### The Problem

Home health and personal care agencies are expected to verify that caregivers wear required PPE (masks) during certain visits, but compliance today is tracked through self-reporting or occasional manual supervisor spot-checks. This doesn't scale across a large caregiver workforce and creates documentation gaps during infection-control audits.

### The Agent

The PPE Compliance Agent perceives a caregiver photo through a fine-tuned YOLOv8n detector, reasons about what it sees using an explicit rule-based decision layer (not a black box), and acts by saving an annotated image plus a written compliance report — with every decision logged in a full inspectable trace.

### Impact

Agency supervisors and compliance officers get a fast, consistent, automated first-pass check instead of manual spot-checks — reducing both the labor cost of manual verification and the documentation gaps that create audit risk.

## Agent Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline diagram and component breakdown.

```
Input (photo/folder) → Preprocessing (validate) → Perception (YOLOv8n)
    → Reasoning (rule-based decision) → Action (annotated image + report)
    → Logging (JSON trace per run)
```

- **Agent framework**: custom Python orchestration loop (no external agent framework needed for Tier 1 rule-based reasoning)
- **CV models/tools**: YOLOv8n (Ultralytics), fine-tuned on a public PPE mask-detection dataset
- **Reasoning**: explicit rule-based priority logic (`agents/reasoning.py`) — every decision records which rule fired and why
- **Communication (multi-agent)**: N/A — single-agent Tier 1 design

## Dataset / Test Inputs

- **Source**: Roboflow Universe — "Mask-Detection-YOLOv8" by AGH ([link](https://universe.roboflow.com/agh-ett2f/mask-detection-yolov8)), CC BY 4.0
- **Size**: 4,547 labeled images (train/valid/test split)
- **Classes**: `with_mask`, `without_mask`, `incorrectly_worn_mask`
- **Preprocessing**: standard Ultralytics augmentation at train time (flip, mosaic, brightness); at inference time the agent's own preprocessing stage validates file integrity, minimum size, and flags low-sharpness images (see `tools/preprocessing.py`)
- **Sample data included in this repo**: `data/sample/` — 3 real caregiver photos for immediate testing after cloning (see `data/README.md`)

## How to Run

### Installation

```bash
git clone <this-repo-url>
cd ITAI1378_Final_PPEComplianceAgent
pip install -r requirements.txt
cp .env.example .env   # no keys required — this agent is rule-based, kept for template compliance
```

### Quick Start

```bash
python agents/ppe_compliance_agent.py --input data/sample
```

This processes all 3 bundled sample images and writes:
- Annotated images → `results/images/`
- Per-image traces → `results/traces/`
- Batch summary → `results/metrics.txt`

**Note on model weights:** the fine-tuned PPE weights (`models/trained/best_yolov8n_ppe.pt`) are not committed to this repo (binary file, too large for git). Without it, the agent automatically falls back to base `yolov8n.pt` (auto-downloaded) so the command above still runs end-to-end — see `models/README.md` for how to get the real trained weights and reproduce the actual reported results below.

## Evaluation & Results

### Component-Level (the CV model itself)

Trained for 75 epochs on a Colab T4 GPU (~76.4 min total). Full log: `results/results.csv`.

| Metric | Value | Target |
|---|---|---|
| mAP@0.5 | **90.0%** (best epoch 64: 90.5%) | ≥ 75–80% ✅ |
| mAP@0.5:0.95 | 60.8% | — |
| Precision | 93.4% | — |
| Recall | 82.5% | — |

### System-Level (the agent as a whole)

- **Task success rate**: demonstrated on 3 bundled sample images (`notebooks/02_evaluation.ipynb`); full 10–20 scenario evaluation requires the real trained weights (see `results/README.md` for current status and what's still needed)
- **Robustness**: verified — corrupt files and undersized images are caught at the preprocessing stage and skipped without crashing the batch; blank-but-valid images are processed normally and correctly return `NO_DETECTION` rather than a guessed answer (evidence in `results/robustness_test_evidence/`)
- **Efficiency**: ~0.7–2.0 sec/image on CPU in the test environment (see `results/metrics.txt`); faster on GPU
- **Failure analysis**: see `notebooks/02_evaluation.ipynb`, Section 5

Full breakdown of what's real trained-model evidence vs. pipeline-mechanics validation: [`results/README.md`](results/README.md).

## Example Agent Run

This trace is reconstructed from an actual Gradio session run against the **real trained
weights** (see `results/demo_compliant_realmodel.png` for the source screenshot — same
detection, same confidence scores, genuinely produced by the trained model):

```json
{
  "image_path": "demo_compliant_realmodel.png (2-person clinical photo)",
  "perception": {
    "num_detections": 2,
    "detections": [
      {"class_name": "with_mask", "confidence": 0.84, "bbox_xyxy": ["..."]},
      {"class_name": "with_mask", "confidence": 0.83, "bbox_xyxy": ["..."]}
    ]
  },
  "reasoning": {
    "status": "COMPLIANT",
    "rule_fired": "R3: with_mask detected, no violations present",
    "explanation": "Detected 2 instance(s) of 'with_mask' and no non-compliant detections in the same frame."
  },
  "action": {"annotated_image_saved_to": "results/demo_compliant_realmodel.png"},
  "status": "COMPLIANT"
}
```

And a non-compliant example (`results/demo_non_compliant_1_realmodel.png`):
```json
{
  "perception": {
    "detections": [{"class_name": "without_mask", "confidence": 0.87, "bbox_xyxy": ["..."]}]
  },
  "reasoning": {
    "status": "NON_COMPLIANT",
    "rule_fired": "R1: without_mask detected",
    "explanation": "Detected 1 instance(s) of 'without_mask' (confidence up to 0.87). Rule R1 takes priority over any other detection in the frame."
  },
  "status": "NON_COMPLIANT"
}
```

For a machine-generated trace showing the full pipeline mechanics (ingestion → preprocessing →
logging), including runs on fallback weights, see the actual JSON files in `results/traces/`.

## Key Learnings

- **What worked well**: rule-based reasoning made every decision fully inspectable — no ambiguity about why the agent flagged something, which matters a lot for a compliance use case
- **Challenges + how solved**: full 75-epoch training took ~76 minutes on free Colab GPU — handled with checkpointing (`save_period=10`) so a disconnect never lost progress; turning raw detections into a trustworthy signal required explicit rule priority (mask-missing always overrides mask-present in the same frame) rather than naive "any mask detected → compliant" logic
- **What I'd do differently**: build the preprocessing/robustness layer earlier in the process rather than after the core detection pipeline — it surfaced edge cases (corrupt files, tiny images) worth designing around from the start

## Mistakes I Made (and How I Got Through Them)

I want to be upfront about this instead of just showing the polished end result, because honestly a lot of this project was figuring things out the hard way.

- **I almost built the wrong project entirely.** My first idea was a face-recognition check-in system, and I got pretty far into planning it before realizing the privacy/data-collection side was a mess — I'd need real people's faces to train on, and I didn't have a clean way to do that. I switched to PPE mask detection instead, which uses public data and sidesteps the whole problem. Lesson: I should've thought through the data situation *before* getting attached to an idea.

- **I built a single-image demo and thought I was done.** My first working version only handled one hardcoded image path. When I actually reread the assignment requirements, it wanted a real pipeline that could take a whole folder of images, not one file I picked ahead of time. I had to go back and rebuild the ingestion step to handle batches properly instead of just patching my one-image script.

- **I didn't think about bad inputs until way too late.** My first version assumed every image handed to it would just... work. It never occurred to me to test what happens with a corrupted file or a tiny 10x10 pixel image until I was doing testing for the "robustness" requirement, and sure enough — the first version I wrote would have just crashed. I had to add a whole separate validation step before the model ever sees the image, so it fails gracefully and logs *why* instead of blowing up.

- **The project requirements changed on me midway through.** I had already built and "finished" a version of this as a straightforward detection + demo project, and then the final assignment spec came out asking for a full agent — perception, reasoning, action, logging, the whole pipeline, plus a totally different repo structure. That was honestly frustrating since I thought I was done. I had to go back and restructure basically everything: split my code into proper modules, add the decision-logic layer, add logging/traces, rewrite the README from scratch. Annoying, but it actually made the project better and more "real" than what I had before.

- **I mixed up "the model working" with "the agent working."** Early on I was just checking my model's mAP score and calling it good. But a good detector isn't the same as a good *agent* — I had to separately test whether my decision logic actually gave the right final answer (compliant/non-compliant), not just whether the detector found the right boxes. Those turned out to be two different things to verify.

- **I didn't have my real trained model file available when testing the final pipeline**, so a lot of my late-stage testing runs came back with weird/empty results at first, which was confusing until I realized it was a weights-loading issue, not a bug in my logic. I ended up building a fallback so the agent still runs even without the "real" weights present — which accidentally became a useful robustness feature instead of just a workaround.

## AI Usage

See [`docs/AI_usage_log.md`](docs/AI_usage_log.md) for the full log and rough attribution breakdown.

## Future Improvements

- Expand to Tier 2 by adding a second CV tool (e.g. glove detection) with the agent choosing which tool(s) to run per image
- Add temporal tracking across video frames instead of single-image inference
- Replace the fixed confidence threshold with a per-class calibrated threshold based on the full evaluation set

## References

1. Ultralytics YOLOv8: https://docs.ultralytics.com/
2. Mask-Detection-YOLOv8 dataset (Roboflow): https://universe.roboflow.com/agh-ett2f/mask-detection-yolov8
3. Gradio: https://gradio.app

## License

Academic use only — ITAI 1378 course project.
