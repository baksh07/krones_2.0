from pathlib import Path

import pytest

from krones_project.paths import REPO_ROOT, get_data_root


def test_repository_root_is_derived_from_package_location():
    assert REPO_ROOT == Path(__file__).resolve().parents[1]


def test_data_root_defaults_to_challenge_data_location(monkeypatch):
    monkeypatch.delenv("KRONES_DATA_ROOT", raising=False)

    assert get_data_root() == Path(
        "/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge"
    )


def test_data_root_can_be_overridden_with_environment(monkeypatch, tmp_path):
    configured_root = tmp_path / "challenge-data"
    monkeypatch.setenv("KRONES_DATA_ROOT", str(configured_root))

    assert get_data_root() == configured_root.resolve()


def test_empty_data_root_override_is_rejected(monkeypatch):
    monkeypatch.setenv("KRONES_DATA_ROOT", " ")

    with pytest.raises(ValueError, match="KRONES_DATA_ROOT must be a non-empty path"):
        get_data_root()


def test_relative_data_root_is_resolved_from_repository_root(monkeypatch):
    monkeypatch.setenv("KRONES_DATA_ROOT", "local-data")

    assert get_data_root() == (REPO_ROOT / "local-data").resolve()
