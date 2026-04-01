from __future__ import annotations

from typing import Dict


def build_physio_text(summary: Dict[str, str]) -> str:
    return (
        "physiology profile: "
        f"eda tonic {summary.get('eda_tonic', 'unknown')}, "
        f"eda phasic density {summary.get('eda_phasic_density', 'unknown')}, "
        f"eda peak sharpness {summary.get('eda_peak_sharpness', 'unknown')}, "
        f"eda recovery {summary.get('eda_recovery', 'unknown')}; "
        f"bvp rhythm {summary.get('bvp_rhythm_stability', 'unknown')}, "
        f"bvp amplitude {summary.get('bvp_pulse_amplitude', 'unknown')}, "
        f"bvp morphology {summary.get('bvp_morphology', 'unknown')}."
    )


def build_mechanism_prompt(summary: Dict[str, str]) -> str:
    return (
        "generate wearable signal with "
        f"{summary.get('eda_tonic', 'unknown')} tonic eda, "
        f"{summary.get('eda_phasic_density', 'unknown')} phasic activity, "
        f"{summary.get('bvp_rhythm_stability', 'unknown')} bvp rhythm, "
        f"{summary.get('bvp_morphology', 'unknown')} pulse morphology."
    )
