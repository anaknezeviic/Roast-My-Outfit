"""Cover feature extraction, head fitting and checkpoint selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
cnn = pytest.importorskip("rmo.perception.cnn")
train_cnn = pytest.importorskip("rmo.perception.train_cnn")

from rmo import paths  # noqa: E402

HEAD_VALUES = {
    "upper_fabric": "denim",
    "lower_fabric": "cotton",
    "outer_fabric": "na",
    "upper_pattern": "striped",
    "lower_pattern": "pure_color",
    "outer_pattern": "na",
    "sleeve_length": "long",
    "lower_length": "long",
    "neckline": "round",
}

CPU = torch.device("cpu")


def tiny_backbone() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.BatchNorm2d(4),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )


def uniform_weights() -> dict[str, list[float]]:
    return {head: [1.0] * size for head, size in cnn.HEAD_CLASS_COUNTS.items()}


def outfit_frame(image_ids, **overrides) -> pd.DataFrame:
    rows = []
    for image_id in image_ids:
        row = {"image_id": image_id, "is_full_body": True, "has_shape": True, **HEAD_VALUES}
        row.update(overrides.get(image_id, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def bank(rows: int, feature_dim: int = 6, *, seed: int = 20260101) -> train_cnn.FeatureBank:
    generator = torch.Generator().manual_seed(seed)
    targets = torch.zeros(rows, len(cnn.HEAD_NAMES), dtype=torch.int64)
    for index, head in enumerate(cnn.HEAD_NAMES):
        targets[:, index] = torch.randint(
            0, cnn.HEAD_CLASS_COUNTS[head], (rows,), generator=generator
        )
    features = torch.zeros(rows, feature_dim)
    for row in range(rows):
        features[row, int(targets[row, 0]) % feature_dim] = 1.0
    return train_cnn.FeatureBank(
        image_ids=tuple(f"image_{index:03d}" for index in range(rows)),
        features=features,
        flipped=features.flip(-1),
        targets=targets,
        supervised=torch.ones(rows, len(cnn.HEAD_NAMES), dtype=torch.bool),
    )


def separable_bank(rows: int, *, seed: int = 20260101) -> train_cnn.FeatureBank:
    """One-hot activations that identify the first head's class exactly."""
    generator = torch.Generator().manual_seed(seed)
    classes = cnn.HEAD_CLASS_COUNTS[cnn.HEAD_NAMES[0]]
    targets = torch.zeros(rows, len(cnn.HEAD_NAMES), dtype=torch.int64)
    targets[:, 0] = torch.randint(0, classes, (rows,), generator=generator)
    features = torch.zeros(rows, classes)
    features[torch.arange(rows), targets[:, 0]] = 1.0
    return train_cnn.FeatureBank(
        image_ids=tuple(f"image_{index:03d}" for index in range(rows)),
        features=features,
        flipped=features.flip(-1),
        targets=targets,
        supervised=torch.ones(rows, len(cnn.HEAD_NAMES), dtype=torch.bool),
    )


def settings(**overrides) -> dict:
    return {
        **train_cnn.DEFAULT_TRAINING,
        "epochs": 2,
        "batch_size": 4,
        "num_workers": 0,
        **overrides,
    }


class StubDataset:
    def __init__(self, rows: int) -> None:
        self.image_ids = tuple(f"image_{index:03d}" for index in range(rows))
        self.targets = np.zeros((rows, len(cnn.HEAD_NAMES)), dtype=np.int64)
        self.supervised = np.ones((rows, len(cnn.HEAD_NAMES)), dtype=bool)
        self._rows = rows

    def __len__(self) -> int:
        return self._rows

    def __getitem__(self, index: int) -> dict:
        pixels = torch.zeros(3, 8, 8)
        pixels[:, :, 0] = float(index + 1)
        return {
            "image_id": self.image_ids[index],
            "pixels": pixels,
            "targets": torch.from_numpy(self.targets[index].copy()),
            "supervised": torch.from_numpy(self.supervised[index].copy()),
        }


def test_extraction_returns_one_row_of_activations_per_photograph() -> None:
    extracted = train_cnn.extract_features(
        tiny_backbone(), StubDataset(5), device=CPU, batch_size=2
    )
    assert extracted.features.shape == (5, 4)
    assert len(extracted) == 5
    assert extracted.image_ids == tuple(f"image_{index:03d}" for index in range(5))


def test_extraction_keeps_no_mirrored_copy_unless_it_is_asked_for() -> None:
    assert (
        train_cnn.extract_features(
            tiny_backbone(), StubDataset(3), device=CPU, batch_size=2
        ).flipped
        is None
    )


def test_the_mirrored_copy_differs_from_the_upright_one() -> None:
    extracted = train_cnn.extract_features(
        tiny_backbone(), StubDataset(3), device=CPU, batch_size=2, with_flip=True
    )
    assert extracted.flipped is not None
    assert extracted.flipped.shape == extracted.features.shape
    assert not torch.allclose(extracted.flipped, extracted.features)


def test_extraction_copies_the_targets_rather_than_sharing_them() -> None:
    dataset = StubDataset(3)
    extracted = train_cnn.extract_features(
        tiny_backbone(), dataset, device=CPU, batch_size=2
    )
    dataset.targets[0, 0] = 5
    assert int(extracted.targets[0, 0]) == 0


def test_a_head_that_sees_one_class_scores_a_perfect_macro_f1() -> None:
    rows = 6
    targets = torch.zeros(rows, len(cnn.HEAD_NAMES), dtype=torch.int64)
    scores = train_cnn.head_scores(
        targets.clone(), targets, torch.ones(rows, len(cnn.HEAD_NAMES), dtype=torch.bool)
    )
    assert scores["upper_fabric"] == {"accuracy": 1.0, "macro_f1": 1.0, "n": rows}


def test_an_unsupervised_head_reports_no_comparisons() -> None:
    rows = 4
    supervised = torch.ones(rows, len(cnn.HEAD_NAMES), dtype=torch.bool)
    supervised[:, cnn.HEAD_NAMES.index("neckline")] = False
    scores = train_cnn.head_scores(
        torch.zeros(rows, len(cnn.HEAD_NAMES), dtype=torch.int64),
        torch.zeros(rows, len(cnn.HEAD_NAMES), dtype=torch.int64),
        supervised,
    )
    assert scores["neckline"] == {"accuracy": 0.0, "macro_f1": 0.0, "n": 0}


def test_macro_f1_ignores_classes_neither_array_uses() -> None:
    actual = np.array([0, 0, 1, 1])
    assert train_cnn._macro_f1(actual, actual, 8) == pytest.approx(1.0)


def test_macro_f1_penalises_collapsing_onto_one_class() -> None:
    actual = np.array([0, 0, 1, 1])
    collapsed = np.zeros(4, dtype=np.int64)
    assert train_cnn._macro_f1(actual, collapsed, 8) == pytest.approx(1 / 3)


def test_fitting_records_one_history_entry_per_epoch() -> None:
    _, _, _, history = train_cnn.fit_heads(
        bank(12), bank(8, seed=7), uniform_weights(), device=CPU, settings=settings(), flip_probability=0.0
    )
    assert [entry["epoch"] for entry in history] == [1, 2]
    assert all(entry["val_mean_macro_f1"] >= 0.0 for entry in history)


def test_fitting_returns_the_best_scoring_epoch() -> None:
    _, best_epoch, best_score, history = train_cnn.fit_heads(
        bank(12), bank(8, seed=7), uniform_weights(), device=CPU, settings=settings(), flip_probability=0.0
    )
    assert best_score == pytest.approx(max(entry["val_mean_macro_f1"] for entry in history))
    assert best_epoch == 1 + max(
        range(len(history)), key=lambda index: history[index]["val_mean_macro_f1"]
    )


def test_fitting_learns_a_separable_head() -> None:
    training = separable_bank(64)
    heads, _, _, _ = train_cnn.fit_heads(
        training,
        training,
        uniform_weights(),
        device=CPU,
        settings=settings(epochs=80, learning_rate=0.1),
        flip_probability=0.0,
    )
    assert isinstance(heads, torch.nn.ModuleDict)
    scores = train_cnn.head_scores(
        train_cnn._predict(heads, training, device=CPU, batch_size=32),
        training.targets,
        training.supervised,
    )
    assert scores[cnn.HEAD_NAMES[0]]["macro_f1"] > 0.9


def test_fitting_is_reproducible_for_one_seed() -> None:
    first, _, _, _ = train_cnn.fit_heads(
        bank(12), bank(8, seed=7), uniform_weights(), device=CPU, settings=settings(), flip_probability=0.5
    )
    second, _, _, _ = train_cnn.fit_heads(
        bank(12), bank(8, seed=7), uniform_weights(), device=CPU, settings=settings(), flip_probability=0.5
    )
    for head in cnn.HEAD_NAMES:
        assert torch.equal(first[head].weight, second[head].weight)


def test_the_mirrored_copy_changes_the_fitted_weights() -> None:
    upright, _, _, _ = train_cnn.fit_heads(
        bank(12), bank(8, seed=7), uniform_weights(), device=CPU, settings=settings(), flip_probability=0.0
    )
    mirrored, _, _, _ = train_cnn.fit_heads(
        bank(12), bank(8, seed=7), uniform_weights(), device=CPU, settings=settings(), flip_probability=1.0
    )
    assert not torch.equal(upright["upper_fabric"].weight, mirrored["upper_fabric"].weight)


def test_a_resumed_run_reaches_the_state_an_uninterrupted_one_does(tmp_path) -> None:
    checkpoint = tmp_path / "cnn.pt"
    whole, _, _, _ = train_cnn.fit_heads(
        bank(12), bank(8, seed=7), uniform_weights(), device=CPU, settings=settings(epochs=4), flip_probability=0.0
    )
    train_cnn.fit_heads(
        bank(12),
        bank(8, seed=7),
        uniform_weights(),
        device=CPU,
        settings=settings(epochs=2),
        flip_probability=0.0,
        checkpoint=checkpoint,
    )
    assert train_cnn.resume_path(checkpoint).is_file()
    resumed, _, _, history = train_cnn.fit_heads(
        bank(12),
        bank(8, seed=7),
        uniform_weights(),
        device=CPU,
        settings=settings(epochs=4),
        flip_probability=0.0,
        checkpoint=checkpoint,
    )
    assert [entry["epoch"] for entry in history] == [1, 2, 3, 4]
    for head in cnn.HEAD_NAMES:
        assert torch.equal(whole[head].weight, resumed[head].weight)


def test_the_resume_sidecar_sits_beside_its_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "models" / "cnn.pt"
    assert train_cnn.resume_path(checkpoint).parent == checkpoint.parent
    assert train_cnn.resume_path(checkpoint).name.startswith(checkpoint.name)


def test_training_writes_a_checkpoint_and_a_metric_record(tmp_path, monkeypatch) -> None:
    image_ids = [f"image_{index:03d}" for index in range(6)]
    frame = outfit_frame(image_ids)
    checkpoint = tmp_path / "cnn.pt"
    metrics_out = tmp_path / "train.json"

    monkeypatch.setattr(train_cnn, "_split_ids", lambda name, limit: image_ids)
    monkeypatch.setattr(
        train_cnn, "split_dataset", lambda name, frame, **kwargs: StubDataset(len(image_ids))
    )
    monkeypatch.setattr(train_cnn, "build_backbone", lambda name, pretrained=False: (tiny_backbone(), 4))
    monkeypatch.setattr(train_cnn.pd, "read_parquet", lambda source: frame)
    monkeypatch.setattr(
        train_cnn,
        "load_perception_config",
        lambda path: {
            "preprocessing": dict(cnn.DEFAULT_PREPROCESSING),
            "checkpoint": {"path": str(checkpoint)},
            "training": settings(),
        },
    )

    run = train_cnn.train(checkpoint_path=checkpoint, metrics_out=metrics_out, device="cpu")

    assert run.checkpoint == checkpoint
    assert checkpoint.is_file()
    assert metrics_out.is_file()
    assert not train_cnn.resume_path(checkpoint).exists()


def test_a_saved_checkpoint_reloads_into_the_same_predictions(tmp_path, monkeypatch) -> None:
    image_ids = [f"image_{index:03d}" for index in range(6)]
    frame = outfit_frame(image_ids)
    checkpoint = tmp_path / "cnn.pt"

    monkeypatch.setattr(train_cnn, "_split_ids", lambda name, limit: image_ids)
    monkeypatch.setattr(
        train_cnn, "split_dataset", lambda name, frame, **kwargs: StubDataset(len(image_ids))
    )
    monkeypatch.setattr(train_cnn, "build_backbone", lambda name, pretrained=False: (tiny_backbone(), 4))
    monkeypatch.setattr(cnn, "build_backbone", lambda name, pretrained=False: (tiny_backbone(), 4))
    monkeypatch.setattr(train_cnn.pd, "read_parquet", lambda source: frame)
    monkeypatch.setattr(
        train_cnn,
        "load_perception_config",
        lambda path: {
            "preprocessing": dict(cnn.DEFAULT_PREPROCESSING),
            "checkpoint": {"path": str(checkpoint)},
            "training": settings(),
        },
    )

    train_cnn.train(
        checkpoint_path=checkpoint, metrics_out=tmp_path / "train.json", device="cpu"
    )
    restored, metadata = cnn.load_checkpoint(checkpoint)
    assert metadata["backbone"] == train_cnn.DEFAULT_TRAINING["backbone"]
    assert set(metadata["class_weights"]) == set(cnn.HEAD_NAMES)
    assert restored.class_counts == cnn.HEAD_CLASS_COUNTS


def test_the_metric_record_names_the_validation_split(tmp_path, monkeypatch) -> None:
    import json

    image_ids = [f"image_{index:03d}" for index in range(6)]
    frame = outfit_frame(image_ids)
    checkpoint = tmp_path / "cnn.pt"
    metrics_out = tmp_path / "train.json"

    monkeypatch.setattr(train_cnn, "_split_ids", lambda name, limit: image_ids)
    monkeypatch.setattr(
        train_cnn, "split_dataset", lambda name, frame, **kwargs: StubDataset(len(image_ids))
    )
    monkeypatch.setattr(train_cnn, "build_backbone", lambda name, pretrained=False: (tiny_backbone(), 4))
    monkeypatch.setattr(train_cnn.pd, "read_parquet", lambda source: frame)
    monkeypatch.setattr(
        train_cnn,
        "load_perception_config",
        lambda path: {
            "preprocessing": dict(cnn.DEFAULT_PREPROCESSING),
            "checkpoint": {"path": str(checkpoint)},
            "training": settings(),
        },
    )

    train_cnn.train(checkpoint_path=checkpoint, metrics_out=metrics_out, device="cpu")
    record = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert record["stage"] == "perception"
    assert record["split"] == "val"
    assert record["seed"] == train_cnn.DEFAULT_TRAINING["seed"]
    assert set(record["metrics"]["per_head"]) == set(cnn.HEAD_NAMES)
    assert record["metrics"]["best_epoch"] >= 1


def test_the_split_digests_name_the_split_they_came_from() -> None:
    digests = train_cnn._split_inputs(("train", "val"))
    staged = {
        name for name in ("train", "val") if (paths.splits_dir() / f"{name}.txt").is_file()
    }
    assert set(digests) == staged
