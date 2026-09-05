"""Check the prediction export and its sidecar manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rmo import paths
from rmo.eval.metrics import config_hash, file_sha256
from rmo.imaging import IN_MEMORY_ID
from rmo.eval.predictions import (
    MANIFEST_VERSION,
    PredictionError,
    read_predictions,
    write_predictions,
)
from rmo.schemas import SCHEMA_VERSION, Garment, GarmentSlot, OutfitDescription

CONFIG = {"model_id": "smol", "temperature": 0.0}

MANIFEST_KEYS = {
    "config_hash",
    "git_sha",
    "inputs",
    "manifest_version",
    "model",
    "n_records",
    "predictions_file",
    "schema_version",
    "seed",
    "sha256",
    "split",
}


def description(
    image_id: str, category: str = "shirt", caption: str = ""
) -> OutfitDescription:
    """Build one minimal prediction."""
    return OutfitDescription(
        image_id=image_id,
        source_model="smol",
        caption=caption,
        garments=[Garment(slot=GarmentSlot.upper, category=category)],
    )


def export(
    destination: Path,
    records: Sequence[OutfitDescription],
    **overrides: Any,
) -> Path:
    """Write one batch through the shared defaults."""
    arguments: dict[str, Any] = {"split": "test", "model": "smol", "config": CONFIG}
    arguments.update(overrides)
    return write_predictions(records, destination, **arguments)


def manifest_of(destination: Path) -> dict[str, Any]:
    """Return the sidecar beside one prediction file."""
    return json.loads(
        destination.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )


def test_write_predictions_orders_records_by_image_id(tmp_path: Path) -> None:
    destination = export(
        tmp_path / "smol_test.jsonl",
        [description("charlie"), description("alpha"), description("bravo")],
    )
    image_ids = [
        json.loads(line)["image_id"]
        for line in destination.read_text(encoding="utf-8").splitlines()
    ]
    assert image_ids == ["alpha", "bravo", "charlie"]


def test_write_predictions_uses_utf8_lf_and_a_trailing_newline(tmp_path: Path) -> None:
    destination = export(tmp_path / "smol_test.jsonl", [description("alpha")])
    payload = destination.read_bytes()
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in payload
    assert payload.endswith(b"\n")


def test_write_predictions_rejects_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="alpha"):
        export(
            tmp_path / "smol_test.jsonl",
            [description("alpha"), description("alpha", category="tee")],
        )


def test_write_predictions_rejects_an_empty_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no records"):
        export(tmp_path / "smol_test.jsonl", [])


def test_write_predictions_rejects_the_in_memory_sentinel(tmp_path: Path) -> None:
    destination = tmp_path / "smol_test.jsonl"
    with pytest.raises(ValueError, match=IN_MEMORY_ID):
        export(destination, [description("alpha"), description(IN_MEMORY_ID)])
    assert not destination.exists()


def test_write_predictions_rejects_an_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train, val, test"):
        export(tmp_path / "smol_holdout.jsonl", [description("alpha")], split="holdout")


def test_write_predictions_rejects_a_destination_that_is_not_jsonl(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in .jsonl"):
        export(tmp_path / "smol_test.json", [description("alpha")])


def test_write_predictions_rejects_incomplete_coverage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1 missing .bravo., 1 unexpected .charlie."):
        export(
            tmp_path / "smol_test.jsonl",
            [description("alpha"), description("charlie")],
            expected_ids={"alpha", "bravo"},
        )


def test_manifest_has_the_expected_keys(tmp_path: Path) -> None:
    destination = export(tmp_path / "smol_test.jsonl", [description("alpha")], seed=7)
    manifest = manifest_of(destination)
    assert set(manifest) == MANIFEST_KEYS
    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["n_records"] == 1
    assert manifest["model"] == "smol"
    assert manifest["split"] == "test"
    assert manifest["seed"] == 7


def test_manifest_records_the_shared_config_hash(tmp_path: Path) -> None:
    destination = export(tmp_path / "smol_test.jsonl", [description("alpha")])
    assert manifest_of(destination)["config_hash"] == config_hash(CONFIG)


def test_manifest_hash_matches_the_written_bytes(tmp_path: Path) -> None:
    destination = export(
        tmp_path / "smol_test.jsonl", [description("alpha"), description("bravo")]
    )
    digest = manifest_of(destination)["sha256"]
    assert digest == file_sha256(destination)
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_manifest_leaks_no_absolute_path(tmp_path: Path) -> None:
    destination = export(tmp_path / "nested" / "smol_test.jsonl", [description("alpha")])
    manifest_text = destination.with_suffix(".manifest.json").read_text(encoding="utf-8")
    assert manifest_of(destination)["predictions_file"] == "smol_test.jsonl"
    assert str(tmp_path) not in manifest_text


def test_manifest_records_the_split_identity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RMO_DATA_ROOT", str(tmp_path / "data"))
    destination = export(tmp_path / "unstaged.jsonl", [description("alpha")])
    assert manifest_of(destination)["inputs"] == {}

    split_file = paths.splits_dir() / "test.txt"
    split_file.parent.mkdir(parents=True)
    split_file.write_bytes(b"alpha\n")
    staged = export(tmp_path / "staged.jsonl", [description("alpha")])
    assert manifest_of(staged)["inputs"] == {"split": file_sha256(split_file)}


def test_a_failed_manifest_write_leaves_no_orphan_predictions(
    tmp_path: Path, monkeypatch
) -> None:
    def refuse(path: Path, payload: object) -> Path:
        """Stand in for a sidecar write that fails once the JSONL is already on disk."""
        raise OSError("manifest write refused")

    monkeypatch.setattr(paths, "write_json_atomic", refuse)
    with pytest.raises(OSError, match="manifest write refused"):
        export(tmp_path / "smol_test.jsonl", [description("alpha")])
    assert list(tmp_path.iterdir()) == []


def test_a_failed_manifest_write_keeps_an_earlier_export(tmp_path: Path, monkeypatch) -> None:
    destination = export(tmp_path / "smol_test.jsonl", [description("alpha")])
    published = destination.read_bytes()

    def refuse(path: Path, payload: object) -> Path:
        """Stand in for a sidecar write that fails during a re-export."""
        raise OSError("manifest write refused")

    monkeypatch.setattr(paths, "write_json_atomic", refuse)
    with pytest.raises(OSError, match="manifest write refused"):
        export(destination, [description("alpha"), description("bravo")])
    assert destination.is_file()
    assert destination.read_bytes() != published
    assert manifest_of(destination)["n_records"] == 1


def test_write_predictions_is_independent_of_input_order(tmp_path: Path) -> None:
    records = [description("bravo"), description("alpha")]
    first = export(tmp_path / "first" / "smol_test.jsonl", records)
    second = export(tmp_path / "second" / "smol_test.jsonl", list(reversed(records)))
    assert first.read_bytes() == second.read_bytes()
    assert (
        first.with_suffix(".manifest.json").read_bytes()
        == second.with_suffix(".manifest.json").read_bytes()
    )


def test_read_predictions_round_trips_every_record(tmp_path: Path) -> None:
    records = [description("bravo"), description("alpha"), description("charlie")]
    destination = export(tmp_path / "smol_test.jsonl", records)
    restored = read_predictions(destination)
    assert [record.image_id for record in restored] == ["alpha", "bravo", "charlie"]
    assert [record.model_dump() for record in restored] == [
        record.model_dump() for record in sorted(records, key=lambda item: item.image_id)
    ]


def test_read_predictions_round_trips_a_unicode_line_separator(tmp_path: Path) -> None:
    caption = "sharp on top\u2028soft below"
    destination = export(
        tmp_path / "smol_test.jsonl",
        [description("alpha", caption=caption), description("bravo")],
    )
    assert [record.caption for record in read_predictions(destination)] == [caption, ""]


def test_read_predictions_rejects_a_tampered_file(tmp_path: Path) -> None:
    destination = export(tmp_path / "smol_test.jsonl", [description("alpha")])
    destination.write_bytes(destination.read_bytes() + b" ")
    with pytest.raises(PredictionError, match="sha256"):
        read_predictions(destination)


def test_read_predictions_rejects_manifest_disagreement(tmp_path: Path) -> None:
    destination = export(
        tmp_path / "smol_test.jsonl", [description("alpha"), description("bravo")]
    )
    manifest_path = destination.with_suffix(".manifest.json")
    manifest = manifest_of(destination)

    manifest["n_records"] = 5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PredictionError, match="manifest claims 5"):
        read_predictions(destination)

    manifest["n_records"] = 2
    for version in (MANIFEST_VERSION + 1, "1", None, [1], True, 0, -1):
        manifest["manifest_version"] = version
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(PredictionError, match="declares version"):
            read_predictions(destination)

    del manifest["manifest_version"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PredictionError, match="declares version"):
        read_predictions(destination)

    manifest["manifest_version"] = MANIFEST_VERSION
    manifest["predictions_file"] = "elsewhere.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PredictionError, match="belongs to"):
        read_predictions(destination)

    manifest_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PredictionError, match="not valid JSON"):
        read_predictions(destination)

    manifest_path.write_bytes(b'{"manifest_version": 1, "model": "\xff"}')
    with pytest.raises(PredictionError, match="not valid JSON"):
        read_predictions(destination)

    manifest_path.write_text("[]", encoding="utf-8")
    with pytest.raises(PredictionError, match="not a JSON object"):
        read_predictions(destination)

    undecodable = b'{"image_id": "\xff"}\n'
    destination.write_bytes(undecodable)
    manifest["predictions_file"] = destination.name
    manifest["sha256"] = hashlib.sha256(undecodable).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PredictionError, match="not valid UTF-8"):
        read_predictions(destination)

    manifest["predictions_file"] = destination.name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    destination.unlink()
    with pytest.raises(PredictionError, match="does not exist"):
        read_predictions(destination)

    manifest_path.unlink()
    with pytest.raises(PredictionError, match="manifest"):
        read_predictions(destination)


def test_read_predictions_rejects_descending_image_ids(tmp_path: Path) -> None:
    destination = tmp_path / "smol_test.jsonl"
    descending = [description("bravo"), description("alpha")]
    payload = "".join(f"{record.model_dump_json()}\n" for record in descending).encode("utf-8")
    destination.write_bytes(payload)
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": MANIFEST_VERSION,
                "n_records": len(descending),
                "predictions_file": destination.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PredictionError, match="strictly ascending"):
        read_predictions(destination)
