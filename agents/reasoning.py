"""
agents/reasoning.py

Reasoning / decision stage for the PPE Compliance Agent.
Consumes structured perception output (a list of Detections) and applies
explicit, inspectable rules to reach a compliance decision. Every decision
records WHICH rule fired and WHY, so the agent's choice can always be
traced back to a specific reason (required for "inspectable" decisions).
"""

from dataclasses import dataclass, field
from typing import List
from tools.detector import Detection

CONFIDENCE_FLOOR = 0.4  # detections below this were already filtered by the detector,
                         # kept here as a documented constant for the reasoning layer


@dataclass
class Decision:
    status: str              # "COMPLIANT" | "NON_COMPLIANT" | "NO_DETECTION"
    rule_fired: str          # which rule produced this decision
    explanation: str         # human-readable reasoning trail
    triggering_detections: List[dict] = field(default_factory=list)


def decide_compliance(detections: List[Detection]) -> Decision:
    """
    Rule-based decision logic (Tier 1: explicit rules, not an LLM).
    Rule priority, in order:
      1. Any 'without_mask' detection -> NON_COMPLIANT (highest priority: safety-critical)
      2. Any 'incorrectly_worn_mask' detection -> NON_COMPLIANT
      3. Any 'with_mask' detection (and no violation above) -> COMPLIANT
      4. No relevant detections at all -> NO_DETECTION (agent abstains rather than guessing)
    """
    class_names = [d.class_name for d in detections]

    if any(c == "without_mask" for c in class_names):
        hits = [d.to_dict() for d in detections if d.class_name == "without_mask"]
        return Decision(
            status="NON_COMPLIANT",
            rule_fired="R1: without_mask detected",
            explanation=(
                f"Detected {len(hits)} instance(s) of 'without_mask' "
                f"(confidence up to {max(h['confidence'] for h in hits):.2f}). "
                "Rule R1 takes priority over any other detection in the frame."
            ),
            triggering_detections=hits,
        )

    if any(c == "incorrectly_worn_mask" for c in class_names):
        hits = [d.to_dict() for d in detections if d.class_name == "incorrectly_worn_mask"]
        return Decision(
            status="NON_COMPLIANT",
            rule_fired="R2: incorrectly_worn_mask detected",
            explanation=(
                f"Detected {len(hits)} instance(s) of a mask worn incorrectly "
                f"(confidence up to {max(h['confidence'] for h in hits):.2f})."
            ),
            triggering_detections=hits,
        )

    if any(c == "with_mask" for c in class_names):
        hits = [d.to_dict() for d in detections if d.class_name == "with_mask"]
        return Decision(
            status="COMPLIANT",
            rule_fired="R3: with_mask detected, no violations present",
            explanation=(
                f"Detected {len(hits)} instance(s) of 'with_mask' and no "
                "non-compliant detections in the same frame."
            ),
            triggering_detections=hits,
        )

    return Decision(
        status="NO_DETECTION",
        rule_fired="R4: no relevant class detected",
        explanation=(
            "No face/mask-related detections above the confidence threshold. "
            "The agent abstains rather than guessing a compliance status — "
            "this frame should be flagged for manual review."
        ),
        triggering_detections=[],
    )
