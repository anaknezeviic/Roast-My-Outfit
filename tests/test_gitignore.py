"""Guard the `.gitignore` negation rules."""

from __future__ import annotations

import subprocess

import pytest

from rmo.paths import repo_root

MUST_BE_COMMITTABLE = [
    "data/fixtures/outfit_descriptions.jsonl",
    "data/fixtures/images/fixture_000.png",
    "data/processed/splits/train.txt",
    "data/processed/splits/MANIFEST.json",
    "data/processed/predictions/cnn_val.jsonl",
    "data/processed/gold_harmony.csv",
    "models/README.md",
]

MUST_BE_IGNORED = [
    "data/raw/images/WOMEN-Blouses_Shirts-id_00000001-01_1_front.jpg",
    "data/raw/labels/shape_anno_all.txt",
    "data/raw/captions.json",
    "data/interim/scratch.parquet",
    "data/processed/outfits.parquet",
    "models/perception_cnn_20260901/model.safetensors",
    "logs/rmo.log",
]


def _is_ignored(relpath: str) -> bool:
    """Ask git whether ``relpath`` would be ignored, tracked or not."""
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", relpath],
        cwd=repo_root(),
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        pytest.fail(
            f"`git check-ignore` failed for {relpath!r} "
            f"(exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.returncode == 0


@pytest.mark.parametrize("relpath", MUST_BE_COMMITTABLE)
def test_committed_paths_are_not_ignored(relpath: str) -> None:
    assert not _is_ignored(relpath), (
        f"{relpath} is ignored but must be committable. Re-include one level at "
        "a time; a `data/**` or bare `models/` rule makes every `!` negation "
        "beneath it a no-op."
    )


@pytest.mark.parametrize("relpath", MUST_BE_IGNORED)
def test_bulk_paths_are_ignored(relpath: str) -> None:
    assert _is_ignored(relpath), (
        f"{relpath} is NOT ignored and would be committed. The dataset licence "
        "forbids redistributing the corpus, and model artefacts belong in "
        "models/README.md as links plus checksums."
    )
