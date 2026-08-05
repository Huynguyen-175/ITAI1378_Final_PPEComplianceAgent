# Architecture

## Agent Type

**Tier 1** — single agent, rule-based reasoning, one CV capability (object detection).
`detect → decide → report`, no LLM required.

## Pipeline Diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  PPEComplianceAgent                      │
                    └─────────────────────────────────────────────────────────┘

 [1] INPUT INGESTION                [2] PREPROCESSING              [3] PERCEPTION (CV)
 ─────────────────────              ──────────────────             ─────────────────────
 Single image, folder,       ──▶    Validate file exists,   ──▶    YOLOv8n inference
 or glob of images                  is a genuine image,             (tools/detector.py)
 (agents/ppe_compliance_             not corrupt, not too           Raw output converted
  agent.py: ingest())                small. Blur-check flags        to structured
                                     low-quality frames.             Detection objects
                                     Bad files are SKIPPED,          (class, confidence,
                                     never crash the batch.          bbox) — never loose
                                     (tools/preprocessing.py)        prints/tensors.


 [6] LOGGING                        [5] ACTION / OUTPUT             [4] REASONING
 ─────────────────────              ──────────────────             ─────────────────────
 Every run (per image)       ◀──    Annotated image saved   ◀──    Rule-based decision
 saved as a JSON trace               to results/images/             (agents/reasoning.py)
 to results/traces/:                 (bounding boxes +              Explicit priority
 input → preprocessing              status banner). Batch           rules (R1-R5), each
 result → detections →              summary appended to             decision records
 decision → action taken.           results/metrics.txt.            WHICH rule fired and
 Batch-level summary also                                           WHY — inspectable,
 written to metrics.txt.                                            not a black box.
```

## Component Breakdown

| Stage | Module | Responsibility |
|---|---|---|
| 1. Input Ingestion | `agents/ppe_compliance_agent.py :: ingest()` | Accepts a file path or folder; builds a list of image paths to process. Handles a batch, not one hard-coded file. |
| 2. Preprocessing | `tools/preprocessing.py :: validate_and_load()` | Confirms the file exists, is a real (non-corrupt) image, is large enough to be useful, and flags (non-fatally) images that are too blurry. Returns a `PreprocessResult` — never raises. |
| 3. Perception | `tools/detector.py :: PPEDetector.detect()` | Wraps a YOLOv8n model (fine-tuned on the PPE dataset). Converts raw Ultralytics output into a list of `Detection` dataclass objects — a defined structure, not raw tensors. |
| 4. Reasoning | `agents/reasoning.py :: decide_compliance()` | Explicit rule-based logic. High-confidence (≥0.65) `without_mask` → NON_COMPLIANT (R1) → high-confidence `incorrectly_worn_mask` → NON_COMPLIANT (R2) → any `with_mask`, no high-confidence violation → COMPLIANT (R3) → only low-confidence violation candidates, nothing else → abstain (R4) → nothing detected → abstain (R5). Every `Decision` records `rule_fired`, a plain-English `explanation`, and `suppressed_detections` (low-confidence violation candidates that did NOT override the result, kept for audit transparency). |
| 5. Action / Output | `agents/ppe_compliance_agent.py :: _save_annotated_image()` | Draws detection boxes + a colored compliance-status banner on the image, saves it to `results/images/`. Appends a run summary to `results/metrics.txt`. |
| 6. Logging | `agents/ppe_compliance_agent.py :: _save_trace()` | Writes one JSON file per processed image to `results/traces/`, capturing every stage's output for that run — the evidence trail that the agent actually did what it claims. |

## Why Rule-Based Reasoning (Not an LLM)

The decision space here is small and well-defined (mask / no mask / incorrectly worn / nothing detected), so a rule-based decision layer is more transparent, faster, cheaper, and easier to verify than routing through an LLM. Per the assignment FAQ, rule-based reasoning is an explicitly valid Tier 1 approach. No API keys or external LLM calls are used anywhere in this pipeline — see `.env.example` for confirmation.

## Design Decision: Confidence-Gated Violation Overrides

Real evaluation surfaced a genuine bug (`notebooks/02_evaluation.ipynb`, Section 6, Failure Case 1): a single low-confidence (0.55) false-positive `without_mask` detection — on an ID badge, not a face — overrode two correct, high-confidence (0.90, 0.87) `with_mask` detections in the same frame, flipping a COMPLIANT photo to NON_COMPLIANT.

**Fix applied:** violation classes (`without_mask`, `incorrectly_worn_mask`) now require confidence ≥ `VIOLATION_CONFIDENCE_FLOOR` (0.65) before they're allowed to override a `with_mask` detection in the same frame — a higher bar than the base detection threshold (0.4). Low-confidence violation candidates are still logged (`suppressed_detections` in every trace) for audit transparency; they just don't unilaterally flip the result. This reflects a deliberate policy choice: a compliance agent should be more skeptical of low-confidence "violation" calls than a general-purpose detector would be, since a false NON_COMPLIANT flag has real consequences for the person being evaluated.

## Fallback Behavior

If the fine-tuned PPE weights (`models/trained/best_yolov8n_ppe.pt`) aren't present on disk (e.g. a fresh clone before downloading the large weight file), `PPEDetector` automatically falls back to base `yolov8n.pt` (auto-downloaded by Ultralytics) so the pipeline still runs end-to-end rather than crashing. A warning is printed to stderr when this happens. This is a deliberate robustness choice, not a bug — see `results/robustness_test/` for a demonstration of graceful failure handling on bad inputs.
