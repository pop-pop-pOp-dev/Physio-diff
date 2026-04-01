from __future__ import annotations

from typing import Dict


ARTIFACT_LABELS = {
    "clean": "clean wearable signal",
    "motion": "motion contaminated wearable signal",
    "burst": "burst artifact with preserved underlying rhythm",
    "dropout": "dropout affected signal with missing segments",
    "wander": "baseline wander contamination",
    "jitter": "time jitter and phase disturbed signal",
    "mixed": "mixed artifact wearable signal",
}


def build_artifact_text(summary: Dict[str, str]) -> str:
    kind = str(summary.get("artifact_kind", "clean")).lower()
    severity = str(summary.get("artifact_severity", "mild")).lower()
    base = ARTIFACT_LABELS.get(kind, ARTIFACT_LABELS["mixed"])
    return f"{severity} {base}".strip()
