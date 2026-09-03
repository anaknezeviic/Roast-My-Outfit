from __future__ import annotations

import json
from pathlib import Path

import pytest

from rmo import paths
from rmo.data import download


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def group(key: str) -> download.AssetGroup:
    return next(item for item in download.GROUPS if item.key == key)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RMO_ROOT", str(tmp_path))
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RMO_SOURCE_DIR", raising=False)
    return tmp_path


@pytest.fixture
def source(env):
    root = env / "drive"
    write(root / "labels" / "shape_anno_all.txt", "img_0.jpg " + " ".join(["0"] * 12) + "\n")
    write(root / "labels" / "fabric_ann.txt", "img_0.jpg 0 1 2\n")
    write(root / "labels" / "texture_ann.txt", "img_0.jpg 3 4 5\n")
    write(root / "textual_descriptions.json", json.dumps({f"img_{i}.jpg": "a shirt" for i in range(3)}))
    for index in range(3):
        write(root / "parsing" / f"img_{index}_segm.png")
    for index in range(6):
        write(root / "images" / f"img_{index}.jpg")
    write(root / "DensePose" / "img_0.png")
    write(root / "keypoints" / "keypoints_loc.txt", "img_0.jpg 1 2\n")
    return root


def test_all_groups_are_staged(source):
    assert download.main(["--all", "--from", str(source)]) == 0
    raw = paths.raw_dir()
    assert len(list((raw / "labels").glob("*.txt"))) == 3
    assert (raw / "textual_descriptions.json").is_file()
    assert len(list((raw / "parsing").glob("*.png"))) == 3
    assert len(list((raw / "images").glob("*.jpg"))) == 6


def test_official_nested_label_layout_is_staged(env):
    source = env / "official"
    write(source / "labels" / "shape" / "shape_anno_all.txt")
    write(source / "labels" / "texture" / "fabric_ann.txt")
    write(source / "labels" / "texture" / "pattern_ann.txt")
    write(source / "captions.json", "{}")

    assert download.main(["--from", str(source)]) == 0
    assert {path.name for path in (paths.raw_dir() / "labels").glob("*.txt")} == {
        "shape_anno_all.txt",
        "fabric_ann.txt",
        "pattern_ann.txt",
    }


def test_densepose_and_keypoints_are_never_staged(source):
    download.main(["--all", "--from", str(source)])
    raw = paths.raw_dir()
    assert not (raw / "DensePose").exists()
    assert not (raw / "keypoints").exists()
    assert download._is_skipped(Path("drive/DensePose/img_0.png"))
    assert download._is_skipped(Path("drive/keypoints/keypoints_loc.txt"))


def test_limit_caps_the_image_group(source):
    assert download.main(["--images", "--from", str(source), "--limit", "2"]) == 0
    assert len(list((paths.raw_dir() / "images").glob("*.jpg"))) == 2


def test_restaging_skips_files_already_present(source):
    images = group("images")
    assert download.stage_group(source, images) == (6, 0)
    assert download.stage_group(source, images) == (0, 6)


def test_interrupted_copy_leaves_no_destination_or_temporary_file(
    tmp_path, monkeypatch
):
    source = write(tmp_path / "source.txt", "complete")
    destination = tmp_path / "raw" / "destination.txt"

    def fail_copy(src, dst):
        Path(dst).write_text("partial", encoding="utf-8")
        raise OSError("copy interrupted")

    monkeypatch.setattr(download.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        download.copy_asset(source, destination)

    assert not destination.exists()
    assert not list(destination.parent.glob("*.tmp"))


def test_dry_run_copies_nothing(source):
    assert download.main(["--all", "--from", str(source), "--dry-run"]) == 0
    assert not (paths.raw_dir() / "images").exists()


def test_missing_source_reports_the_manual_steps(env, capsys):
    assert download.main(["--labels-only"]) == download.EXIT_NO_SOURCE
    assert "--from" in capsys.readouterr().out


def test_empty_source_is_reported_as_incomplete(env):
    empty = env / "empty"
    empty.mkdir()
    assert download.main(["--labels-only", "--from", str(empty)]) == download.EXIT_INCOMPLETE


def test_labels_only_rejects_an_incomplete_annotation_set(env, caplog):
    source = env / "partial"
    write(source / "labels" / "fabric_ann.txt", "img.jpg 0 0 0\n")
    write(source / "captions.json", "{}")

    with caplog.at_level("ERROR", logger="rmo.data.download"):
        result = download.main(["--labels-only", "--from", str(source)])

    assert result == download.EXIT_INCOMPLETE
    assert "missing annotations: shape, pattern" in caplog.text


def test_labels_only_rejects_other_groups(source):
    with pytest.raises(SystemExit):
        download.main(["--labels-only", "--images", "--from", str(source)])


def test_verify_inventories_without_a_source(source):
    download.main(["--all", "--from", str(source)])
    assert download.main(["--verify"]) == 0


def test_inspect_file_counts_lines_and_entries(env):
    root = env / "raw"
    text = write(root / "labels" / "fabric_ann.txt", "a 0 1 2\nb 3 4 5\n")
    captions = write(root / "captions.json", json.dumps({"a": "x", "b": "y", "c": "z"}))
    assert download.inspect_file(text, root).lines == 2
    record = download.inspect_file(captions, root)
    assert record.json_entries == 3
    assert record.json_kind == "dict"


def test_inventory_summarises_bulk_directories(source):
    download.main(["--all", "--from", str(source)])
    records = {record.relpath: record for record in download.inventory(paths.raw_dir())}
    assert records["images/"].sample == "6 files"
    assert records["parsing/"].sample == "3 files"


def test_describe_reports_counts(source):
    download.main(["--all", "--from", str(source)])
    summaries = {
        record.relpath: download.describe(record)
        for record in download.inventory(paths.raw_dir())
    }
    assert summaries["images/"].endswith("6 files")
    assert "1 lines" in summaries["labels/fabric_ann.txt"]


def test_verify_writes_no_documents(source):
    download.main(["--all", "--from", str(source)])
    download.main(["--verify"])
    assert not (paths.repo_root() / "docs").exists()



