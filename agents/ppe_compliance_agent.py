"""
agents/ppe_compliance_agent.py

PPE Compliance Agent — Tier 1 CV agent for ITAI 1378.

Implements the full required pipeline:
  1. Input Ingestion   - accepts a single image, a folder of images, or a glob
  2. Preprocessing     - validates/loads each image, catches bad inputs gracefully
  3. Perception (CV)   - YOLOv8 detection -> structured Detection objects
  4. Reasoning         - rule-based decision logic (see agents/reasoning.py)
  5. Action / Output   - saves an annotated image + a written report to results/
  6. Logging           - saves a full JSON trace of every run to results/traces/

Run from the command line:
    python agents/ppe_compliance_agent.py --input data/sample

Each image in --input is processed independently; a failure on one image
(corrupt file, no detections, etc.) never stops the batch.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

# allow running as `python agents/ppe_compliance_agent.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.detector import PPEDetector
from tools.preprocessing import validate_and_load
from agents.reasoning import decide_compliance

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PPEComplianceAgent:
    def __init__(self, weights_path: str = None, conf_threshold: float = 0.4,
                 results_dir: str = "results"):
        self.detector = PPEDetector(weights_path=weights_path, conf_threshold=conf_threshold)
        self.results_dir = Path(results_dir)
        self.images_dir = self.results_dir / "images"
        self.traces_dir = self.results_dir / "traces"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        if self.detector.using_fallback:
            print(
                f"[warning] fine-tuned PPE weights not found on disk — "
                f"using fallback '{self.detector.weights_path}'. "
                f"See models/README.md to download the trained weights.",
                file=sys.stderr,
            )

    # ---------- Stage 1: Input Ingestion ----------
    def ingest(self, input_path: str):
        p = Path(input_path)
        if p.is_dir():
            paths = sorted(
                [str(f) for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
            )
        elif p.is_file():
            paths = [str(p)]
        else:
            paths = []
        return paths

    # ---------- Stages 2-6 for a single image ----------
    def process_one(self, image_path: str) -> dict:
        t0 = time.time()
        trace = {
            "image_path": image_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Stage 2: Preprocessing
        pre = validate_and_load(image_path)
        trace["preprocessing"] = {
            "valid": pre.valid,
            "reason": pre.reason,
            "warning": pre.warning,
        }
        if not pre.valid:
            trace["status"] = "SKIPPED_INVALID_INPUT"
            trace["latency_sec"] = round(time.time() - t0, 3)
            self._save_trace(trace)
            return trace

        # Stage 3: Perception
        detections, raw_result = self.detector.detect(pre.image_array)
        trace["perception"] = {
            "num_detections": len(detections),
            "detections": [d.to_dict() for d in detections],
        }

        # Stage 4: Reasoning
        decision = decide_compliance(detections)
        trace["reasoning"] = {
            "status": decision.status,
            "rule_fired": decision.rule_fired,
            "explanation": decision.explanation,
        }

        # Stage 5: Action / Output
        annotated_path = self._save_annotated_image(image_path, raw_result, decision)
        trace["action"] = {
            "annotated_image_saved_to": str(annotated_path),
        }

        trace["status"] = decision.status
        trace["latency_sec"] = round(time.time() - t0, 3)

        # Stage 6: Logging
        self._save_trace(trace)
        return trace

    def run(self, input_path: str) -> list:
        image_paths = self.ingest(input_path)
        if not image_paths:
            print(f"No valid images found at: {input_path}", file=sys.stderr)
            return []

        print(f"Agent processing {len(image_paths)} image(s) from {input_path} ...")
        all_traces = []
        for img_path in image_paths:
            trace = self.process_one(img_path)
            all_traces.append(trace)
            status = trace.get("status", "ERROR")
            print(f"  {Path(img_path).name:40s} -> {status}")

        self._write_batch_summary(all_traces)
        return all_traces

    # ---------- helpers ----------
    def _save_annotated_image(self, image_path, raw_result, decision) -> Path:
        annotated = raw_result.plot()  # BGR numpy array with boxes drawn
        label = decision.status
        color = {
            "COMPLIANT": (60, 160, 60),
            "NON_COMPLIANT": (40, 40, 200),
            "NO_DETECTION": (120, 120, 120),
        }.get(label, (120, 120, 120))
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), color, -1)
        cv2.putText(annotated, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

        out_name = f"{Path(image_path).stem}_annotated.jpg"
        out_path = self.images_dir / out_name
        cv2.imwrite(str(out_path), annotated)
        return out_path

    def _save_trace(self, trace: dict):
        fname = f"{Path(trace['image_path']).stem}_{int(time.time()*1000)}.json"
        with open(self.traces_dir / fname, "w") as f:
            json.dump(trace, f, indent=2)

    def _write_batch_summary(self, traces: list):
        counts = {}
        for t in traces:
            s = t.get("status", "ERROR")
            counts[s] = counts.get(s, 0) + 1
        summary_path = self.results_dir / "metrics.txt"
        with open(summary_path, "a") as f:
            f.write(f"\n--- Run at {datetime.now(timezone.utc).isoformat()} ---\n")
            f.write(f"Total images: {len(traces)}\n")
            for status, count in counts.items():
                f.write(f"  {status}: {count}\n")
            avg_latency = sum(t.get("latency_sec", 0) for t in traces) / max(len(traces), 1)
            f.write(f"Average latency: {avg_latency:.3f} sec/image\n")
        print(f"\nBatch summary appended to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="PPE Compliance Agent")
    parser.add_argument("--input", required=True,
                         help="Path to a single image or a folder of images")
    parser.add_argument("--weights", default=None,
                         help="Path to trained model weights (defaults to models/trained/best_yolov8n_ppe.pt)")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--results-dir", default="results", help="Where to save outputs")
    args = parser.parse_args()

    agent = PPEComplianceAgent(weights_path=args.weights, conf_threshold=args.conf,
                                results_dir=args.results_dir)
    agent.run(args.input)


if __name__ == "__main__":
    main()
