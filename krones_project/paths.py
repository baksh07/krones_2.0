"""Shared paths for repository scripts and external challenge data."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge")


def get_data_root() -> Path:
    """Return the challenge data directory, honoring the environment override."""
    configured_root = os.environ.get("KRONES_DATA_ROOT")
    if configured_root is None:
        return DEFAULT_DATA_ROOT
    if not configured_root.strip():
        raise ValueError("KRONES_DATA_ROOT must be a non-empty path.")

    data_root = Path(configured_root).expanduser()
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    return data_root.resolve()


DATA_ROOT = get_data_root()
