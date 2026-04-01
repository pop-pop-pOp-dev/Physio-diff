from __future__ import annotations

from typing import Dict


CANONICAL_LABEL_TEXT = {
    0: "shared physiology state with lower sympathetic arousal and steadier autonomic balance",
    1: "shared physiology state with elevated sympathetic arousal and stronger stress reactivity",
    2: "shared physiology state with high arousal and unstable autonomic regulation",
}


DATASET_HINTS = {
    "wesad": "laboratory wearable physiology",
    "stress_predict": "daily stress prediction physiology",
    "swell_kw": "office stress physiology",
    "case": "affective physiology benchmark",
    "ubfc_phys": "multimodal physiological stress recording",
    "mahnob_hci": "emotion elicitation physiology",
}


def map_label_to_semantic_text(dataset_name: str, label: int, summary: Dict[str, str] | None = None) -> str:
    dataset_hint = DATASET_HINTS.get(str(dataset_name).lower(), "cross-domain wearable physiology")
    canonical = CANONICAL_LABEL_TEXT.get(int(label), CANONICAL_LABEL_TEXT[1])
    if not summary:
        return f"{dataset_hint}; {canonical}."
    return (
        f"{dataset_hint}; {canonical}; "
        f"eda tonic {summary.get('eda_tonic', 'unknown')}, "
        f"eda phasic density {summary.get('eda_phasic_density', 'unknown')}, "
        f"bvp rhythm {summary.get('bvp_rhythm_stability', 'unknown')}."
    )
