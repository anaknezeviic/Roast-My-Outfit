"""Label parsing tests covering every decode path and every rejection."""

from __future__ import annotations

from pathlib import Path

import pytest

from rmo.data import parse_annotations as annotations

SHAPE_ROW = "img_0.jpg 3 2 1 1 2 1 0 1 1 0 1 0"
SHAPE_NA_ROW = "img_1.jpg 5 4 3 2 4 2 2 2 4 6 2 2"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def labels(tmp_path):
    root = tmp_path / "labels"
    write(root / "shape_anno_all.txt", f"{SHAPE_ROW}\n{SHAPE_NA_ROW}\n")
    write(root / "fabric_ann.txt", "img_0.jpg 0 1 2\nimg_1.jpg 7 7 7\n")
    write(root / "texture_ann.txt", "img_0.jpg 5 6 3\nimg_1.jpg 7 7 7\n")
    return root


def test_shape_row_decodes_every_field(labels):
    row = annotations.parse_shape(labels / "shape_anno_all.txt").iloc[0]
    assert row["image_id"] == "img_0"
    assert row["sleeve_length"] == "long"
    assert row["lower_length"] == "three_quarter"
    assert row["socks"] == "socks"
    assert row["hat"] == "yes"
    assert row["glasses"] == "sunglasses"
    assert row["neckwear"] == "yes"
    assert row["wrist"] == "no"
    assert row["ring"] == "yes"
    assert row["waist_accessory"] == "belt"
    assert row["neckline"] == "v_shape"
    assert row["navel_covered"] == "no"


def test_cardigan_code_zero_means_yes(labels):
    frame = annotations.parse_shape(labels / "shape_anno_all.txt")
    assert frame.loc[0, "cardigan"] == "no"
    write(labels / "shape_anno_all.txt", "img_9.jpg 0 0 0 0 0 0 0 0 0 0 0 0\n")
    assert annotations.parse_shape(labels / "shape_anno_all.txt").loc[0, "cardigan"] == "yes"


def test_na_codes_decode_to_na_and_are_not_null(labels):
    frame = annotations.parse_shape(labels / "shape_anno_all.txt")
    row = frame.iloc[1]
    for column in annotations.SHAPE_COLUMNS:
        assert row[column] == "na"
    assert frame[list(annotations.SHAPE_COLUMNS)].notna().all().all()


def test_fabric_row_decodes(labels):
    row = annotations.parse_fabric(labels / "fabric_ann.txt").iloc[0]
    assert (row["upper_fabric"], row["lower_fabric"], row["outer_fabric"]) == (
        "denim",
        "cotton",
        "leather",
    )


def test_pattern_five_is_other_and_six_is_color_block(labels):
    row = annotations.parse_pattern(labels / "texture_ann.txt").iloc[0]
    assert row["upper_pattern"] == "other"
    assert row["lower_pattern"] == "color_block"
    assert row["outer_pattern"] == "pure_color"


def test_image_id_is_the_stem(labels):
    frame = annotations.parse_fabric(labels / "fabric_ann.txt")
    assert list(frame["image_id"]) == ["img_0", "img_1"]


def test_columns_are_stable_and_rows_sorted(tmp_path):
    path = write(tmp_path / "fabric_ann.txt", "b.jpg 0 0 0\na.jpg 1 1 1\n")
    frame = annotations.parse_fabric(path)
    assert list(frame.columns) == ["image_id", *annotations.FABRIC_COLUMNS]
    assert list(frame["image_id"]) == ["a", "b"]


def test_categories_cover_the_whole_vocabulary(labels):
    frame = annotations.parse_pattern(labels / "texture_ann.txt")
    assert list(frame["upper_pattern"].cat.categories) == [
        "floral",
        "graphic",
        "striped",
        "pure_color",
        "lattice",
        "other",
        "color_block",
        "na",
    ]


def test_blank_lines_are_ignored(tmp_path):
    path = write(tmp_path / "fabric_ann.txt", "\nimg_0.jpg 0 0 0\n\n\nimg_1.jpg 1 1 1\n\n")
    assert len(annotations.parse_fabric(path)) == 2


def test_wrong_field_count_raises(tmp_path):
    path = write(tmp_path / "fabric_ann.txt", "img_0.jpg 0 1\n")
    with pytest.raises(annotations.AnnotationError, match="expected 4 fields"):
        annotations.parse_fabric(path)


def test_non_integer_code_raises(tmp_path):
    path = write(tmp_path / "fabric_ann.txt", "img_0.jpg 0 denim 2\n")
    with pytest.raises(annotations.AnnotationError, match="not an integer"):
        annotations.parse_fabric(path)


def test_out_of_range_code_raises(tmp_path):
    path = write(tmp_path / "fabric_ann.txt", "img_0.jpg 0 1 8\n")
    with pytest.raises(annotations.AnnotationError, match="outside 0..7"):
        annotations.parse_fabric(path)


def test_negative_code_raises(tmp_path):
    path = write(tmp_path / "fabric_ann.txt", "img_0.jpg 0 1 -1\n")
    with pytest.raises(annotations.AnnotationError, match="outside"):
        annotations.parse_fabric(path)


def test_duplicate_image_id_raises(tmp_path):
    path = write(tmp_path / "fabric_ann.txt", "img_0.jpg 0 1 2\nimg_0.jpg 1 2 3\n")
    with pytest.raises(annotations.AnnotationError, match="already appeared on line 1"):
        annotations.parse_fabric(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(annotations.AnnotationError, match="does not exist"):
        annotations.parse_fabric(tmp_path / "absent.txt")


def test_find_label_files_classifies_by_width_and_name(labels):
    found = annotations.find_label_files(labels)
    assert found["shape"].name == "shape_anno_all.txt"
    assert found["fabric"].name == "fabric_ann.txt"
    assert found["pattern"].name == "texture_ann.txt"


def test_missing_pattern_file_raises(labels):
    (labels / "texture_ann.txt").unlink()
    with pytest.raises(annotations.AnnotationError, match="pattern"):
        annotations.find_label_files(labels)


def test_parse_label_dir_returns_all_three(labels):
    tables = annotations.parse_label_dir(labels)
    assert set(tables) == {"shape", "fabric", "pattern"}
    assert all(len(frame) == 2 for frame in tables.values())


def test_parse_label_dir_rejects_misaligned_files(labels):
    write(labels / "fabric_ann.txt", "img_0.jpg 0 1 2\nimg_7.jpg 1 2 3\n")
    with pytest.raises(annotations.AnnotationError, match="missing"):
        annotations.parse_label_dir(labels)
