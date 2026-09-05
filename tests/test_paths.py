"""Cover the filesystem layout helpers and the artefact write contract."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from rmo import paths


def test_export_path_makes_a_repo_file_relative() -> None:
    inside = paths.repo_root() / "configs" / "perception.yaml"
    assert paths.export_path(inside) == "configs/perception.yaml"


def test_export_path_uses_forward_slashes() -> None:
    assert "\\" not in paths.export_path(paths.fixtures_dir() / "images" / "fixture_000.png")


def test_export_path_passes_a_relative_path_through() -> None:
    assert paths.export_path("data/raw/images/a.jpg") == "data/raw/images/a.jpg"


def test_export_path_normalises_relative_separators() -> None:
    assert paths.export_path("data\\raw\\a.jpg".replace("\\", "/")) == "data/raw/a.jpg"


def test_export_path_of_a_file_outside_the_repo(tmp_path) -> None:
    assert paths.export_path(tmp_path / "elsewhere.jpg") == ""


@pytest.mark.parametrize("value", ["", None])
def test_export_path_of_nothing_is_empty(value) -> None:
    assert paths.export_path(value) == ""


def test_export_path_follows_the_root_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_ROOT", str(tmp_path))
    assert paths.export_path(tmp_path / "models" / "cnn.pt") == "models/cnn.pt"
    assert paths.export_path(paths.repo_root().parent / "outside.pt") == ""


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    payload = b"rmo" * 500_000
    target = tmp_path / "payload.bin"
    target.write_bytes(payload)
    assert paths.file_sha256(target) == hashlib.sha256(payload).hexdigest()


def test_metrics_dir_sits_under_results_not_data_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    assert paths.data_root() == tmp_path.resolve()
    assert paths.results_dir() == paths.repo_root() / "results"
    assert paths.metrics_dir() == paths.repo_root() / "results" / "metrics"


def test_results_and_metrics_dirs_create_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_ROOT", str(tmp_path))
    assert paths.results_dir() == tmp_path.resolve() / "results"
    assert not paths.results_dir().exists()
    assert not paths.metrics_dir().exists()


def test_predictions_dir_follows_the_data_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path))
    assert paths.predictions_dir() == tmp_path.resolve() / "processed" / "predictions"


def test_write_bytes_atomic_leaves_no_temporary_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "artefact.json"
    target.write_bytes(b"original\n")

    def refuse(self: Path, destination: Path) -> None:
        """Stand in for a rename that the filesystem rejects."""
        raise OSError("replace refused")

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError, match="replace refused"):
        paths.write_bytes_atomic(target, b"replacement\n")
    assert target.read_bytes() == b"original\n"
    assert list(tmp_path.iterdir()) == [target]


def test_write_bytes_atomic_writes_through_a_sibling(tmp_path: Path, monkeypatch) -> None:
    written: list[Path] = []
    original = Path.write_bytes

    def capture(self: Path, payload: bytes) -> int:
        """Record the path the writer chose before delegating to the real write."""
        written.append(self)
        return original(self, payload)

    monkeypatch.setattr(Path, "write_bytes", capture)
    target = tmp_path / "nested" / "artefact.json"
    target.parent.mkdir()
    paths.write_bytes_atomic(target, b"payload\n")
    assert [path.parent for path in written] == [target.parent]


def test_git_sha_reports_the_commit_and_falls_back_to_none(monkeypatch) -> None:
    commit = "a" * 40
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, f"{commit}\n", ""),
    )
    assert paths.git_sha() == commit

    def explode(*args: object, **kwargs: object) -> None:
        """Stand in for a machine with no git executable."""
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", explode)
    assert paths.git_sha() is None
