"""Cover the split-wise prediction exporter."""

from __future__ import annotations

import json

import pytest

from rmo import paths
from rmo.eval import cnn_predictions
from rmo.eval.predictions import read_predictions
from rmo.schemas import Garment, GarmentSlot, OutfitDescription, Provenance
from rmo.splits import SPLIT_NAMES


class StubModel:
    name = "stub_perception"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def predict(self, image):
        raise AssertionError("the exporter must batch")

    def predict_batch(self, images):
        self.seen = [image.stem for image in images]
        return [
            OutfitDescription(
                image_id=image.stem,
                image_path=f"data/raw/images/{image.stem}.jpg",
                source_model=self.name,
                provenance=Provenance.predicted,
                garments=[
                    Garment(slot=GarmentSlot.upper, category="top", ref="g1"),
                ],
            )
            for image in images
        ]


@pytest.fixture()
def staged(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    data = root / "data"
    images = data / "raw" / "images"
    splits = data / "processed" / "splits"
    images.mkdir(parents=True)
    splits.mkdir(parents=True)
    (root / "pyproject.toml").write_text("", encoding="utf-8")

    members = {
        "train": ["a_train", "b_train"],
        "val": ["a_val"],
        "test": ["a_test", "b_test"],
    }
    for name, ids in members.items():
        (splits / f"{name}.txt").write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
        for image_id in ids:
            (images / f"{image_id}.jpg").write_bytes(b"not really a jpeg")

    monkeypatch.setenv("RMO_ROOT", str(root))
    monkeypatch.setenv("RMO_DATA_ROOT", str(data))
    monkeypatch.setattr(
        cnn_predictions, "run_config", lambda model: {"model_id": model.name}
    )
    return root, members


def test_the_destination_is_named_for_the_model_and_the_split(tmp_path) -> None:
    destination = cnn_predictions.prediction_path("cnn_multihead_v1", "val", tmp_path)
    assert destination == tmp_path / "cnn_multihead_v1_val.jsonl"


def test_a_model_name_with_a_separator_stays_one_path_segment(tmp_path) -> None:
    destination = cnn_predictions.prediction_path("Vendor/Model", "test", tmp_path)
    assert destination.parent == tmp_path
    assert destination.name == "Vendor_Model_test.jsonl"


def test_the_default_destination_sits_in_the_predictions_directory(staged) -> None:
    destination = cnn_predictions.prediction_path("stub_perception", "train")
    assert destination.parent == paths.predictions_dir()


def test_an_unknown_split_is_refused(staged, tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        cnn_predictions.export_split(StubModel(), "holdout", tmp_path / "out.jsonl")


def test_a_destination_that_is_not_jsonl_is_refused(staged, tmp_path) -> None:
    with pytest.raises(ValueError, match="must end in .jsonl"):
        cnn_predictions.export_split(StubModel(), "val", tmp_path / "out.json")


def test_an_absent_photograph_is_reported_before_the_model_runs(staged, tmp_path) -> None:
    root, _ = staged
    (root / "data" / "raw" / "images" / "a_val.jpg").unlink()
    model = StubModel()
    with pytest.raises(FileNotFoundError, match="No photograph at"):
        cnn_predictions.export_split(model, "val", tmp_path / "out.jsonl")
    assert model.seen == []


def test_an_export_round_trips_through_the_reader(staged, tmp_path) -> None:
    destination = cnn_predictions.export_split(StubModel(), "val", tmp_path / "val.jsonl")
    restored = read_predictions(destination)
    assert [record.image_id for record in restored] == ["a_val"]


def test_the_manifest_names_the_split_it_covers(staged, tmp_path) -> None:
    destination = cnn_predictions.export_split(StubModel(), "train", tmp_path / "train.jsonl")
    manifest = json.loads(
        destination.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["split"] == "train"
    assert manifest["n_records"] == 2
    assert manifest["model"] == "stub_perception"


def test_every_split_is_exported_including_train(staged, tmp_path) -> None:
    written = cnn_predictions.export_splits(StubModel(), directory=tmp_path)
    assert set(written) == set(SPLIT_NAMES)
    assert "train" in written
    assert all(destination.is_file() for destination in written.values())


def test_the_exports_carry_one_record_per_split_member(staged, tmp_path) -> None:
    written = cnn_predictions.export_splits(StubModel(), directory=tmp_path)
    counts = {split: len(read_predictions(path)) for split, path in written.items()}
    assert counts == {"train": 2, "val": 1, "test": 2}


def test_all_splits_share_one_configuration_hash(staged, tmp_path) -> None:
    written = cnn_predictions.export_splits(StubModel(), directory=tmp_path)
    hashes = {
        json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))[
            "config_hash"
        ]
        for path in written.values()
    }
    assert len(hashes) == 1


def test_a_partial_export_does_not_claim_full_coverage(staged, tmp_path) -> None:
    destination = cnn_predictions.export_split(
        StubModel(), "train", tmp_path / "train.jsonl", limit=1
    )
    assert len(read_predictions(destination)) == 1


def test_a_full_export_refuses_incomplete_coverage(staged, tmp_path) -> None:
    class ShortModel(StubModel):
        def predict_batch(self, images):
            return super().predict_batch(images)[:1]

    with pytest.raises(ValueError, match="do not cover"):
        cnn_predictions.export_split(ShortModel(), "train", tmp_path / "train.jsonl")
