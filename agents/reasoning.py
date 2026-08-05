"""
agents/reasoning.py

Reasoning / decision stage for the PPE Compliance Agent.
Consumes structured perception output (a list of Detections) and applies
explicit, inspectable rules to reach a compliance decision. Every decision
records WHICH rule fired and WHY, so the agent's choice can always be
traced back to a specific reason (required for "inspectable" decisions).

v2 fix (see notebooks/02_evaluation.ipynb, Section 6, Failure Case 1):
Originally, ANY 'without_mask' detection overrode all other detections in
the frame regardless of confidence -- a single low-confidence false
positive (e.g. a badge misread as a bare face at 0.55 confidence) could
flip an otherwise-correct COMPLIANT frame to NON_COMPLIANT. Violation
classes now require a higher confidence bar (VIOLATION_CONFIDENCE_FLOOR)
before they're allowed to override a genuine 'with_mask' detection --
a compliance agent should be more skeptical of low-confidence "violation"
calls than a general-purpose detector would be, since a false NON_COMPLIANT
flag has real consequences for the person in the photo.
"""

from dataclasses import dataclass, field
from typing import List
from tools.detector import Detection

DETECTION_CONFIDENCE_FLOOR = 0.4   # already applied by the detector itself
VIOLATION_CONFIDENCE_FLOOR = 0.65  # higher bar: only detections at/above this
                                    # can override a with_mask detection in the same frame


@dataclass
class Decision:
    status: str              # "COMPLIANT" | "NON_COMPLIANT" | "NO_DETECTION"
    rule_fired: str          # which rule produced this decision
    explanation: str         # human-readable reasoning trail
    triggering_detections: List[dict] = field(default_factory=list)
    suppressed_detections: List[dict] = field(default_factory=list)  # low-confidence
                                                                       # violations that
                                                                       # did NOT override,
                                                                       # kept for transparency


def decide_compliance(detections: List[Detection]) -> Decision:
    """
    Rule-based decision logic (Tier 1: explicit rules, not an LLM).

    Rule priority, in order:
      1. A HIGH-CONFIDENCE 'without_mask' detection (>= VIOLATION_CONFIDENCE_FLOOR)
         -> NON_COMPLIANT (safety-critical, highest priority)
      2. A HIGH-CONFIDENCE 'incorrectly_worn_mask' detection -> NON_COMPLIANT
      3. Any 'with_mask' detection, with no high-confidence violation present
         -> COMPLIANT (low-confidence violation candidates, if any, are logged
         but do not override -- see `suppressed_detections`)
      4. Only low-confidence violation candidates, no with_mask at all
         -> NO_DETECTION, abstain and flag for manual review rather than guess
      5. No relevant detections at all -> NO_DETECTION
    """
    violation_classes = {"without_mask", "incorrectly_worn_mask"}

    high_conf_violations = [
        d for d in detections
        if d.class_name in violation_classes and d.confidence >= VIOLATION_CONFIDENCE_FLOOR
    ]
    low_conf_violations = [
        d for d in detections
        if d.class_name in violation_classes and d.confidence < VIOLATION_CONFIDENCE_FLOOR
    ]
    mask_detections = [d for d in detections if d.class_name == "with_mask"]

    # Rule 1/2: high-confidence violation present -> NON_COMPLIANT, always wins
    if high_conf_violations:
        without_mask_hits = [d for d in high_conf_violations if d.class_name == "without_mask"]
        incorrect_hits = [d for d in high_conf_violations if d.class_name == "incorrectly_worn_mask"]

        if without_mask_hits:
            hits = [d.to_dict() for d in without_mask_hits]
            return Decision(
                status="NON_COMPLIANT",
                rule_fired="R1: high-confidence without_mask detected",
                explanation=(
                    f"Detected {len(hits)} instance(s) of 'without_mask' at or above "
                    f"the {VIOLATION_CONFIDENCE_FLOOR:.2f} violation-confidence floor "
                    f"(confidence up to {max(h['confidence'] for h in hits):.2f})."
                ),
                triggering_detections=hits,
                suppressed_detections=[d.to_dict() for d in low_conf_violations],
            )

        hits = [d.to_dict() for d in incorrect_hits]
        return Decision(
            status="NON_COMPLIANT",
            rule_fired="R2: high-confidence incorrectly_worn_mask detected",
            explanation=(
                f"Detected {len(hits)} instance(s) of a mask worn incorrectly at or "
                f"above the {VIOLATION_CONFIDENCE_FLOOR:.2f} violation-confidence floor "
                f"(confidence up to {max(h['confidence'] for h in hits):.2f})."
            ),
            triggering_detections=hits,
            suppressed_detections=[d.to_dict() for d in low_conf_violations],
        )

    # Rule 3: genuine mask detection(s), no high-confidence violation -> COMPLIANT.
    # Any low-confidence violation candidates are logged for transparency but
    # explicitly did not override the result.
    if mask_detections:
        hits = [d.to_dict() for d in mask_detections]
        suppressed = [d.to_dict() for d in low_conf_violations]
        note = ""
        if suppressed:
            note = (
                f" Note: {len(suppressed)} low-confidence violation candidate(s) "
                f"were detected below the {VIOLATION_CONFIDENCE_FLOOR:.2f} override "
                f"threshold and were NOT used to override this result — see "
                f"suppressed_detections in the trace."
            )
        return Decision(
            status="COMPLIANT",
            rule_fired="R3: with_mask detected, no high-confidence violations present",
            explanation=(
                f"Detected {len(hits)} instance(s) of 'with_mask' and no "
                f"high-confidence non-compliant detections in the same frame.{note}"
            ),
            triggering_detections=hits,
            suppressed_detections=suppressed,
        )

    # Rule 4: only low-confidence violation candidates, nothing else -> abstain
    if low_conf_violations:
        hits = [d.to_dict() for d in low_conf_violations]
        return Decision(
            status="NO_DETECTION",
            rule_fired="R4: only low-confidence violation candidate(s) present",
            explanation=(
                f"Detected {len(hits)} possible violation(s) below the "
                f"{VIOLATION_CONFIDENCE_FLOOR:.2f} confidence floor required to act on "
                "them, and no confirmed 'with_mask' detection either. The agent "
                "abstains rather than guessing — this frame should be flagged for "
                "manual review."
            ),
            triggering_detections=[],
            suppressed_detections=hits,
        )

    # Rule 5: nothing relevant detected at all
    return Decision(
        status="NO_DETECTION",
        rule_fired="R5: no relevant class detected",
        explanation=(
            "No face/mask-related detections above the base confidence threshold. "
            "The agent abstains rather than guessing a compliance status — "
            "this frame should be flagged for manual review."
        ),
        triggering_detections=[],
    )
