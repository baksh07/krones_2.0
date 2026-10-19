"""Compatibility imports for scripts run directly from this experiment folder."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").is_file()
    ),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Could not locate the repository root from this script.")

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from krones_project.paths import DATA_ROOT, REPO_ROOT  # noqa: E402

__all__ = ["DATA_ROOT", "REPO_ROOT"]
