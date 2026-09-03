"""Cover group keying and split loading."""

from __future__ import annotations

import hashlib
import itertools
import json
import re

import pandas as pd
import pytest

from rmo import paths
from rmo.splits import SPLIT_NAMES, SPLIT_SEED, group_key_for, load_split, write_splits


@pytest.mark.parametrize(
    ("image_id", "expected"),
    [
        (
            "WOMEN-Blouses_Shirts-id_00000001-01_1_front",
            "WOMEN-Blouses_Shirts-id_00000001",
        ),
        ("MEN-Denim-id_00000080-01_7_additional", "MEN-Denim-id_00000080"),
        ("MEN-Denim-id_00000080", "MEN-Denim-id_00000080"),
    ],
)
def test_group_key_strips_the_shot_token(image_id: str, expected: str) -> None:
    assert group_key_for(image_id) == expected


def test_every_shot_of_one_product_shares_a_group_key() -> None:
    shots = [
        "WOMEN-Blouses_Shirts-id_00000001-01_1_front",
        "WOMEN-Blouses_Shirts-id_00000001-01_2_side",
        "WOMEN-Blouses_Shirts-id_00000001-02_4_full",
    ]
    assert len({group_key_for(shot) for shot in shots}) == 1


@pytest.mark.parametrize("image_id", ["no_id_here", "", "WOMEN-Blouses_Shirts-id_123"])
def test_group_key_rejects_an_id_without_a_product_segment(image_id: str) -> None:
    with pytest.raises(ValueError):
        group_key_for(image_id)


def test_load_split_rejects_an_unknown_name_before_touching_disk(monkeypatch) -> None:
    def fail() -> None:
        raise AssertionError("load_split resolved a path for an unknown split name.")

    monkeypatch.setattr(paths, "splits_dir", fail)
    with pytest.raises(ValueError):
        load_split("bogus")


def test_load_split_reads_stems_and_drops_blank_lines(tmp_path, monkeypatch) -> None:
    (tmp_path / "train.txt").write_text(
        "MEN-Denim-id_00000080-01_7_additional\n\nMEN-Denim-id_00000080-01_1_front\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "splits_dir", lambda: tmp_path)
    assert load_split("train") == {
        "MEN-Denim-id_00000080-01_7_additional",
        "MEN-Denim-id_00000080-01_1_front",
    }


def test_split_names_are_the_three_the_loader_accepts() -> None:
    assert SPLIT_NAMES == ("train", "val", "test")


def test_committed_split_manifest_matches_files() -> None:
    split_dir = paths.splits_dir()
    manifest = json.loads((split_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    split_ids = {name: load_split(name) for name in SPLIT_NAMES}

    assert manifest["counts"] == {
        name: len(split_ids[name]) for name in SPLIT_NAMES
    }
    assert manifest["n_groups"] == len(
        {
            group_key_for(image_id)
            for image_ids in split_ids.values()
            for image_id in image_ids
        }
    )
    assert manifest["split_seed"] == SPLIT_SEED
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["git_sha"])
    assert manifest["sha256"] == {
        f"{name}.txt": hashlib.sha256(
            (split_dir / f"{name}.txt").read_bytes()
        ).hexdigest()
        for name in SPLIT_NAMES
    }


def test_write_splits_rejects_missing_stratification_values(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "image_id": "WOMEN-Dress-id_00000001-01_1_front",
                "gender": None,
                "category_from_filename": "Dress",
                "is_full_body": True,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing gender or category"):
        write_splits(frame, tmp_path)


def test_written_splits_are_deterministic_grouped_and_lf_terminated(tmp_path) -> None:
    records = []
    for number in range(1, 15):
        gender = "MEN" if number % 2 else "WOMEN"
        category = "Denim" if number % 2 else "Blouses_Shirts"
        for shot in ("front", "side"):
            records.append(
                {
                    "image_id": f"{gender}-{category}-id_{number:08d}-01_1_{shot}",
                    "gender": gender.lower(),
                    "category_from_filename": category,
                    "is_full_body": True,
                }
            )
    records.append(
        {
            "image_id": "invalid",
            "gender": "women",
            "category_from_filename": "Dress",
            "is_full_body": True,
        }
    )
    frame = pd.DataFrame(records)

    first = write_splits(frame, tmp_path)
    first_contents = {name: (tmp_path / f"{name}.txt").read_bytes() for name in first}
    second = write_splits(frame, tmp_path)

    assert first == second
    assert "invalid" not in set().union(*first.values())
    groups = {
        name: {group_key_for(image_id) for image_id in image_ids}
        for name, image_ids in first.items()
    }
    for first_name, second_name in itertools.combinations(SPLIT_NAMES, 2):
        assert groups[first_name].isdisjoint(groups[second_name])
    for name, content in first_contents.items():
        assert content == (tmp_path / f"{name}.txt").read_bytes()
        assert content.endswith(b"\n")
        assert b"\r\n" not in content

    manifest = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {name: len(first[name]) for name in SPLIT_NAMES}
    assert manifest["n_groups"] == len(set().union(*groups.values()))
    assert manifest["split_seed"] == 20260101
    assert len(manifest["git_sha"]) == 40
    assert manifest["sha256"] == {
        f"{name}.txt": hashlib.sha256(first_contents[name]).hexdigest()
        for name in SPLIT_NAMES
    }
