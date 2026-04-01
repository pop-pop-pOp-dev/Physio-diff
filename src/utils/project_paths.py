from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    root = project_root()
    direct = (root / candidate).resolve()
    if direct.exists():
        return str(direct)
    cwd_candidate = Path.cwd() / candidate
    return str(cwd_candidate.resolve())
