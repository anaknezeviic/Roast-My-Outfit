"""Cover group keying and split loading."""

from __future__ import annotations

import pytest

from rmo import paths
from rmo.splits import SPLIT_NAMES, group_key_for, load_split


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
