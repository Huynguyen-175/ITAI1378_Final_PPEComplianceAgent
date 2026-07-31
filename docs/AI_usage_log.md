# AI Usage Log

Tool used throughout: **Claude** (Anthropic). This log documents AI usage across both the
original midterm-style proposal phase and the final capstone rebuild into a proper CV agent.

| # | Phase | How It Was Used |
|---|---|---|
| 1 | Scoping | Brainstormed project topic options (including face recognition vs. PPE detection), compared feasibility/privacy tradeoffs, scoped down to a Tier 1, timeline-realistic idea. |
| 2 | Scoping | Selected model (YOLOv8n) and dataset (Roboflow "Mask-Detection-YOLOv8," 4,547 images) with justification. |
| 3 | Midterm build | Generated initial repo scaffold, proposal slide deck, and a Colab training notebook covering dataset download, training, and evaluation. |
| 4 | Midterm build | Explained the role of the `patience` (early stopping) parameter vs. epoch count. |
| 5 | Midterm build | Parsed real `results.csv` training output to identify best/final epoch metrics (90.5% / 90.0% mAP@0.5) and generated a training-curve chart from the actual data. |
| 6 | Midterm build | Added a Gradio interface to the Colab notebook wrapping the trained model + compliance logic into an interactive demo. |
| 7 | **Requirement change** | Read the updated final project guide (CV Agent capstone) and identified the gap between the existing midterm-style deliverable and the new agent-pipeline requirements (perception + reasoning + action + logging, restructured repo, system-level evaluation). |
| 8 | Final rebuild | Designed the 6-stage agent pipeline architecture (ingestion → preprocessing → perception → reasoning → action → logging) and the corresponding repo structure (`agents/`, `tools/`, `docs/architecture.md`). |
| 9 | Final rebuild | Wrote `tools/detector.py` (structured perception output, `Detection` dataclass, automatic fallback to base weights if fine-tuned weights are absent). |
| 10 | Final rebuild | Wrote `tools/preprocessing.py` (input validation: corrupt file detection, size checks, blur detection via Laplacian variance) — designed to never crash on bad input. |
| 11 | Final rebuild | Wrote `agents/reasoning.py` (explicit rule-based decision logic with priority rules R1–R4, each decision recording which rule fired and why). |
| 12 | Final rebuild | Wrote `agents/ppe_compliance_agent.py` (orchestrator tying all 6 stages together, CLI entry point, batch processing, JSON trace logging, annotated image output). |
| 13 | Final rebuild | Actually executed the agent end-to-end in a sandboxed environment (using fallback weights, since the real fine-tuned weights weren't available there) to verify the pipeline runs without errors, including a dedicated robustness test with a corrupt file, an undersized image, and a blank image — confirmed graceful handling rather than crashes. |
| 14 | Final rebuild | Built `notebooks/01_exploration.ipynb` (training), `notebooks/02_evaluation.ipynb` (component + system-level evaluation, robustness tests, failure-analysis scaffold), and `notebooks/03_demo.ipynb` (Gradio walkthrough calling the actual agent modules). |
| 15 | Final rebuild | Rewrote `README.md` to match the required capstone template exactly, using real numbers from actual training output and real screenshots from the trained-model Gradio session (not fabricated examples). |
| 16 | Documentation | Wrote `docs/architecture.md`, `models/README.md`, `data/README.md`, `results/README.md`, and this log. |

## Attribution Summary

- **CV model training & dataset selection**: student decision, executed by student in Colab; Claude assisted with code and interpreting results
- **Agent pipeline architecture (6-stage design)**: designed by Claude based on the assignment's explicit pipeline requirements, reviewed and approved by student
- **Core pipeline code** (`agents/`, `tools/`): ~90% written by Claude, reviewed by student — student understands and can explain every stage
- **Reasoning/rule logic**: designed collaboratively — rule priority order (mask-missing overrides mask-present) reflects a safety-first compliance judgment call made explicit in code comments
- **Evaluation design**: Claude proposed the component-level vs. system-level split and the robustness-test approach per the assignment's evaluation requirements; actual full-scale evaluation against the real trained weights and the complete labeled test set is still to be run by the student (see `results/README.md`, "Still Needed" section)

## Reflection

**What worked well:** using Claude to translate the assignment's abstract pipeline requirements (ingestion, preprocessing, perception, reasoning, action, logging) into an actual concrete, testable code structure — and then actually running the code in a sandboxed environment to catch real bugs before submission, rather than just generating untested code.

**What I was careful about:** the mid-project requirement change meant a lot of new infrastructure was needed quickly. I made sure the delivered code was actually executed and verified (not just generated) before accepting it, and that the README is explicit and honest about which results come from the real trained model versus which are pipeline-mechanics smoke tests using fallback weights — this distinction matters for grading integrity.

**What's still my responsibility:** running the full evaluation notebook against real trained weights and the complete labeled test set (10–20+ scenarios) to get genuine system-level accuracy numbers, and writing the final honest failure-case analysis based on those real results.
