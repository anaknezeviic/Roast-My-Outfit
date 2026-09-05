"""Cover the nine-head attribute model, its dataset and its masked loss."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from rmo.data.parse_annotations import _PATTERN
from rmo.schemas import Fabric, LowerLength, Neckline, Pattern, SleeveLength

torch = pytest.importorskip("torch")
cnn = pytest.importorskip("rmo.perception.cnn")

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


def tiny_backbone() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.BatchNorm2d(4),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
    )


def uniform_weights() -> dict[str, list[float]]:
    return {head: [1.0] * size for head, size in cnn.HEAD_CLASS_COUNTS.items()}


def random_logits(rows: int, *, grad: bool = False) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260101)
    return {
        head: torch.randn(
            rows, cnn.HEAD_CLASS_COUNTS[head], generator=generator, requires_grad=grad
        )
        for head in cnn.HEAD_NAMES
    }


def outfit_frame(image_ids, **overrides) -> pd.DataFrame:
    rows = []
    for image_id in image_ids:
        row = {"image_id": image_id, "is_full_body": True, "has_shape": True, **HEAD_VALUES}
        row.update(overrides.get(image_id, {}))
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def photos(tmp_path):
    def make(image_ids):
        directory = tmp_path / "images"
        directory.mkdir(exist_ok=True)
        for offset, image_id in enumerate(image_ids):
            Image.new("RGB", (8, 8), (10 * offset, 20, 30)).save(directory / f"{image_id}.jpg")
        return directory

    return make


def test_head_names_follow_the_vocabulary_order() -> None:
    assert cnn.HEAD_NAMES == (
        "upper_fabric",
        "lower_fabric",
        "outer_fabric",
        "upper_pattern",
        "lower_pattern",
        "outer_pattern",
        "sleeve_length",
        "lower_length",
        "neckline",
    )


def test_head_cardinalities_match_the_schema_enums() -> None:
    assert tuple(cnn.HEAD_CLASS_COUNTS[head] for head in cnn.HEAD_NAMES) == (
        8,
        8,
        8,
        8,
        8,
        8,
        6,
        5,
        7,
    )


@pytest.mark.parametrize(
    ("head", "vocabulary"),
    [
        ("upper_fabric", Fabric),
        ("lower_pattern", Pattern),
        ("sleeve_length", SleeveLength),
        ("lower_length", LowerLength),
        ("neckline", Neckline),
    ],
)
def test_class_maps_are_the_schema_enum_values(head, vocabulary) -> None:
    assert cnn.CLASS_MAPS[head] == tuple(member.value for member in vocabulary)


def test_shape_heads_are_the_three_annotated_per_image() -> None:
    assert cnn.SHAPE_HEADS == {"sleeve_length", "lower_length", "neckline"}


def test_class_indices_come_from_the_schema_not_the_annotation_codes() -> None:
    assert cnn.CLASS_MAPS["upper_pattern"].index("color_block") == 5
    assert _PATTERN.index("color_block") == 6


def test_encode_label_returns_the_schema_position() -> None:
    assert cnn.encode_label("upper_fabric", "denim") == 0
    assert cnn.encode_label("neckline", "na") == 6


def test_encode_label_rejects_a_foreign_value() -> None:
    with pytest.raises(ValueError, match="denim"):
        cnn.encode_label("neckline", "denim")


def test_encode_label_rejects_a_foreign_head() -> None:
    with pytest.raises(ValueError, match="socks"):
        cnn.encode_label("socks", "na")


def test_encode_frame_indexes_every_head() -> None:
    targets, supervised = cnn.encode_frame(outfit_frame(["a"]))
    assert targets.tolist() == [[0, 1, 7, 2, 3, 7, 3, 3, 2]]
    assert supervised.all()
    assert targets.dtype == np.int64
    assert supervised.dtype == bool


def test_encode_frame_masks_only_the_shape_heads() -> None:
    frame = outfit_frame(["a", "b"], b={"has_shape": False})
    _, supervised = cnn.encode_frame(frame)
    shape = [index for index, head in enumerate(cnn.HEAD_NAMES) if head in cnn.SHAPE_HEADS]
    assert supervised[0].all()
    assert supervised[1][shape].tolist() == [False, False, False]
    assert supervised[1].sum() == 6


def test_encode_frame_keeps_a_genuine_na_supervised() -> None:
    frame = outfit_frame(["a"], a={"sleeve_length": "na", "neckline": "na"})
    targets, supervised = cnn.encode_frame(frame)
    assert supervised.all()
    assert targets[0][cnn.HEAD_NAMES.index("sleeve_length")] == 5


def test_encode_frame_rejects_an_unknown_value() -> None:
    with pytest.raises(ValueError, match="velour"):
        cnn.encode_frame(outfit_frame(["a"], a={"upper_fabric": "velour"}))


def test_encode_frame_names_the_missing_columns() -> None:
    frame = outfit_frame(["a"]).drop(columns=["has_shape", "neckline"])
    with pytest.raises(ValueError, match="neckline, has_shape"):
        cnn.encode_frame(frame)


def targets_for(column: int, values, rows: int) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((rows, len(cnn.HEAD_NAMES)), dtype=np.int64)
    targets[:, column] = values
    return targets, np.ones((rows, len(cnn.HEAD_NAMES)), dtype=bool)


def test_class_weights_match_the_hand_counted_oracle() -> None:
    targets, supervised = targets_for(0, [0, 0, 1, 1, 1, 1, 1, 1], 8)
    weights = cnn.class_weights(targets, supervised)["upper_fabric"]
    assert weights[0] == pytest.approx(2.0)
    assert weights[1] == pytest.approx(2.0 / 3.0)
    assert weights[2:] == [0.0] * 6


def test_class_weights_keep_the_supervised_row_total() -> None:
    targets, supervised = targets_for(0, [0, 0, 1, 1, 1, 1, 1, 1], 8)
    weights = cnn.class_weights(targets, supervised)["upper_fabric"]
    counts = np.bincount(targets[:, 0], minlength=len(weights))
    assert float(np.dot(counts, weights)) == pytest.approx(8.0)


def test_class_weights_are_a_list_per_class() -> None:
    targets, supervised = targets_for(0, [0] * 4, 4)
    weights = cnn.class_weights(targets, supervised)
    assert set(weights) == set(cnn.HEAD_NAMES)
    assert {head: len(values) for head, values in weights.items()} == cnn.HEAD_CLASS_COUNTS


def test_class_weights_ignore_unsupervised_rows() -> None:
    targets, supervised = targets_for(6, [0, 0, 1, 1], 4)
    supervised[2:, 6] = False
    weights = cnn.class_weights(targets, supervised)["sleeve_length"]
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == 0.0


def test_a_head_with_no_supervised_rows_warns_once(caplog) -> None:
    targets, supervised = targets_for(0, [0] * 4, 4)
    supervised[:, 6] = False
    with caplog.at_level("WARNING", logger="rmo.perception.cnn"):
        weights = cnn.class_weights(targets, supervised)
    assert [record.getMessage() for record in caplog.records] == [
        "head sleeve_length has no supervised rows, so every class weight is zero"
    ]
    assert weights["sleeve_length"] == [0.0] * 6


def test_untrained_heads_lists_the_all_zero_heads() -> None:
    targets, supervised = targets_for(0, [0] * 4, 4)
    supervised[:, 6] = False
    supervised[:, 8] = False
    assert cnn.untrained_heads(cnn.class_weights(targets, supervised)) == (
        "sleeve_length",
        "neckline",
    )


def test_validation_rows_cannot_change_the_train_weights() -> None:
    train_targets, train_supervised = targets_for(0, [0, 0, 1, 1], 4)
    val_targets, val_supervised = targets_for(0, [1, 1, 1, 1], 4)
    train_only = cnn.class_weights(train_targets, train_supervised)
    combined = cnn.class_weights(
        np.vstack([train_targets, val_targets]),
        np.vstack([train_supervised, val_supervised]),
    )
    assert cnn.class_weights(train_targets, train_supervised) == train_only
    assert combined["upper_fabric"] != train_only["upper_fabric"]


def test_the_model_carries_one_head_per_attribute() -> None:
    model = cnn.MultiHeadCNN(tiny_backbone(), 4)
    assert list(model.heads) == list(cnn.HEAD_NAMES)
    assert [model.heads[head].out_features for head in cnn.HEAD_NAMES] == [
        cnn.HEAD_CLASS_COUNTS[head] for head in cnn.HEAD_NAMES
    ]


def test_the_model_returns_finite_logits_for_a_batch() -> None:
    model = cnn.MultiHeadCNN(tiny_backbone(), 4).eval()
    logits = model(torch.rand(2, 3, 224, 224))
    assert list(logits) == list(cnn.HEAD_NAMES)
    assert all(values.shape == (2, cnn.HEAD_CLASS_COUNTS[head]) for head, values in logits.items())
    assert all(bool(torch.isfinite(values).all()) for values in logits.values())


def test_the_model_rejects_a_head_map_that_is_not_the_nine_heads() -> None:
    with pytest.raises(ValueError, match="nine heads"):
        cnn.MultiHeadCNN(tiny_backbone(), 4, {"upper_fabric": 8, "hat": 3})


def test_the_model_rejects_an_empty_head() -> None:
    counts = dict(cnn.HEAD_CLASS_COUNTS)
    counts["neckline"] = 0
    with pytest.raises(ValueError, match="neckline"):
        cnn.MultiHeadCNN(tiny_backbone(), 4, counts)


def test_the_backbone_is_frozen() -> None:
    model = cnn.MultiHeadCNN(tiny_backbone(), 4)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head_parameters())


def test_training_the_model_leaves_the_backbone_in_evaluation_mode() -> None:
    model = cnn.MultiHeadCNN(tiny_backbone(), 4)
    assert model.train() is model
    assert model.heads.training
    assert not model.backbone.training


def test_one_optimizer_step_moves_the_heads_and_nothing_else() -> None:
    model = cnn.MultiHeadCNN(tiny_backbone(), 4).train()
    convolution = model.backbone[0].weight.detach().clone()
    running_mean = model.backbone[1].running_mean.detach().clone()
    head = model.heads["upper_fabric"].weight.detach().clone()

    optimizer = torch.optim.SGD(model.head_parameters(), lr=0.5)
    logits = model(torch.rand(2, 3, 32, 32))
    breakdown = cnn.masked_cross_entropy(
        logits,
        torch.zeros(2, 9, dtype=torch.int64),
        torch.ones(2, 9, dtype=torch.bool),
        uniform_weights(),
    )
    optimizer.zero_grad()
    breakdown.total.backward()
    optimizer.step()

    assert torch.equal(model.backbone[0].weight, convolution)
    assert torch.equal(model.backbone[1].running_mean, running_mean)
    assert not torch.equal(model.heads["upper_fabric"].weight, head)


def test_build_backbone_reports_the_pooled_feature_width() -> None:
    pytest.importorskip("timm")
    model, width = cnn.build_backbone("resnet18", pretrained=False)
    assert width == 512
    assert isinstance(model, torch.nn.Module)


def test_build_transform_normalises_to_the_configured_size() -> None:
    pixels = cnn.build_transform({"preprocessing": cnn.DEFAULT_PREPROCESSING})(
        Image.new("RGB", (8, 12), (255, 0, 0))
    )
    assert pixels.shape == (3, 224, 224)
    assert pixels.dtype == torch.float32
    assert float(pixels[0].mean()) == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-4)


def test_the_loss_of_a_uniformly_weighted_head_is_plain_cross_entropy() -> None:
    logits = random_logits(4)
    targets = torch.zeros(4, 9, dtype=torch.int64)
    breakdown = cnn.masked_cross_entropy(
        logits, targets, torch.ones(4, 9, dtype=torch.bool), uniform_weights()
    )
    expected = torch.nn.functional.cross_entropy(logits["upper_fabric"], targets[:, 0])
    assert float(breakdown.per_head["upper_fabric"]) == pytest.approx(float(expected), abs=1e-6)
    assert breakdown.active == cnn.HEAD_NAMES


def test_the_loss_skips_unsupervised_rows() -> None:
    logits = random_logits(4)
    targets = torch.zeros(4, 9, dtype=torch.int64)
    supervised = torch.ones(4, 9, dtype=torch.bool)
    supervised[0, 0] = False
    breakdown = cnn.masked_cross_entropy(logits, targets, supervised, uniform_weights())
    expected = torch.nn.functional.cross_entropy(
        logits["upper_fabric"][1:], targets[1:, 0]
    )
    assert float(breakdown.per_head["upper_fabric"]) == pytest.approx(float(expected), abs=1e-6)


def test_a_fully_unsupervised_head_is_a_differentiable_zero() -> None:
    logits = random_logits(4, grad=True)
    supervised = torch.ones(4, 9, dtype=torch.bool)
    supervised[:, 6] = False
    breakdown = cnn.masked_cross_entropy(
        logits, torch.zeros(4, 9, dtype=torch.int64), supervised, uniform_weights()
    )
    assert "sleeve_length" not in breakdown.active
    assert float(breakdown.per_head["sleeve_length"].detach()) == 0.0
    assert breakdown.per_head["sleeve_length"].requires_grad


def test_a_texture_only_row_activates_exactly_six_heads() -> None:
    supervised = torch.ones(2, 9, dtype=torch.bool)
    for index, head in enumerate(cnn.HEAD_NAMES):
        if head in cnn.SHAPE_HEADS:
            supervised[:, index] = False
    breakdown = cnn.masked_cross_entropy(
        random_logits(2), torch.zeros(2, 9, dtype=torch.int64), supervised, uniform_weights()
    )
    assert breakdown.active == tuple(
        head for head in cnn.HEAD_NAMES if head not in cnn.SHAPE_HEADS
    )
    assert len(breakdown.active) == 6


def test_a_genuine_na_shape_label_still_contributes() -> None:
    targets = torch.zeros(2, 9, dtype=torch.int64)
    targets[:, 6] = cnn.encode_label("sleeve_length", "na")
    breakdown = cnn.masked_cross_entropy(
        random_logits(2), targets, torch.ones(2, 9, dtype=torch.bool), uniform_weights()
    )
    assert "sleeve_length" in breakdown.active
    assert float(breakdown.per_head["sleeve_length"]) > 0.0


def test_an_all_zero_weight_head_stays_inactive() -> None:
    weights = uniform_weights()
    weights["sleeve_length"] = [0.0] * cnn.HEAD_CLASS_COUNTS["sleeve_length"]
    breakdown = cnn.masked_cross_entropy(
        random_logits(3), torch.zeros(3, 9, dtype=torch.int64), torch.ones(3, 9, dtype=torch.bool), weights
    )
    assert "sleeve_length" not in breakdown.active
    assert bool(torch.isfinite(breakdown.total))


def test_every_head_with_no_supervision_leaves_a_zero_total() -> None:
    logits = random_logits(2, grad=True)
    breakdown = cnn.masked_cross_entropy(
        logits, torch.zeros(2, 9, dtype=torch.int64), torch.zeros(2, 9, dtype=torch.bool), uniform_weights()
    )
    assert breakdown.active == ()
    assert float(breakdown.total.detach()) == 0.0
    breakdown.total.backward()
    assert all(bool(torch.isfinite(values.grad).all()) for values in logits.values())


def test_the_loss_produces_finite_gradients() -> None:
    logits = random_logits(4, grad=True)
    supervised = torch.ones(4, 9, dtype=torch.bool)
    supervised[:, 8] = False
    breakdown = cnn.masked_cross_entropy(
        logits, torch.zeros(4, 9, dtype=torch.int64), supervised, uniform_weights()
    )
    breakdown.total.backward()
    assert breakdown.active == tuple(cnn.HEAD_NAMES[:8])
    assert all(bool(torch.isfinite(logits[head].grad).all()) for head in breakdown.active)
    assert logits["neckline"].grad is None


def test_the_dataset_sorts_the_requested_ids(photos) -> None:
    frame = outfit_frame(["b", "a"], a={"upper_fabric": "cotton"})
    dataset = cnn.OutfitAttributeDataset(frame, ["b", "a"], image_dir=photos(["a", "b"]))
    assert dataset.image_ids == ("a", "b")
    assert len(dataset) == 2
    assert dataset.targets[:, 0].tolist() == [1, 0]


def test_the_dataset_item_carries_pixels_targets_and_masks(photos) -> None:
    frame = outfit_frame(["a"], a={"has_shape": False})
    dataset = cnn.OutfitAttributeDataset(frame, ["a"], image_dir=photos(["a"]))
    item = dataset[0]
    assert item["image_id"] == "a"
    assert item["pixels"].shape == (3, 224, 224)
    assert item["pixels"].dtype == torch.float32
    assert item["targets"].dtype == torch.int64
    assert item["supervised"].dtype == torch.bool
    assert int(item["supervised"].sum()) == 6


def test_the_dataset_rejects_a_repeated_request(photos) -> None:
    frame = outfit_frame(["a"])
    with pytest.raises(ValueError, match="'a'"):
        cnn.OutfitAttributeDataset(frame, ["a", "a"], image_dir=photos(["a"]))


def test_the_dataset_rejects_an_unknown_id(photos) -> None:
    frame = outfit_frame(["a"])
    with pytest.raises(ValueError, match="'b'"):
        cnn.OutfitAttributeDataset(frame, ["b"], image_dir=photos(["a", "b"]))


def test_the_dataset_rejects_an_ambiguous_row(photos) -> None:
    frame = pd.concat([outfit_frame(["a"]), outfit_frame(["a"])], ignore_index=True)
    with pytest.raises(ValueError, match="matches 2 rows"):
        cnn.OutfitAttributeDataset(frame, ["a"], image_dir=photos(["a"]))


def test_the_dataset_rejects_a_partial_body_row(photos) -> None:
    frame = outfit_frame(["a"], a={"is_full_body": False})
    with pytest.raises(ValueError, match="full-body"):
        cnn.OutfitAttributeDataset(frame, ["a"], image_dir=photos(["a"]))


def test_the_dataset_rejects_a_missing_photograph(tmp_path) -> None:
    frame = outfit_frame(["a"])
    with pytest.raises(ValueError, match="a.jpg"):
        cnn.OutfitAttributeDataset(frame, ["a"], image_dir=tmp_path)


def test_split_dataset_takes_every_id_of_the_split(photos, monkeypatch) -> None:
    monkeypatch.setattr(cnn, "load_split", lambda name: {"a", "b"})
    dataset = cnn.split_dataset(
        "train", outfit_frame(["a", "b"]), image_dir=photos(["a", "b"])
    )
    assert dataset.image_ids == ("a", "b")


def test_split_dataset_accepts_a_smoke_subset(photos, monkeypatch) -> None:
    monkeypatch.setattr(cnn, "load_split", lambda name: {"a", "b"})
    dataset = cnn.split_dataset(
        "train", outfit_frame(["a", "b"]), image_ids=["b"], image_dir=photos(["a", "b"])
    )
    assert dataset.image_ids == ("b",)


def test_split_dataset_rejects_an_id_outside_the_split(photos, monkeypatch) -> None:
    monkeypatch.setattr(cnn, "load_split", lambda name: {"a"})
    with pytest.raises(ValueError, match="not part of the train split"):
        cnn.split_dataset(
            "train", outfit_frame(["a", "b"]), image_ids=["b"], image_dir=photos(["a", "b"])
        )


@pytest.fixture()
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(cnn, "build_backbone", lambda name, **options: (tiny_backbone(), 4))
    model = cnn.MultiHeadCNN(tiny_backbone(), 4)
    destination = tmp_path / "weights" / "cnn.pt"
    cnn.save_checkpoint(
        destination,
        model,
        backbone="tiny",
        preprocessing=cnn.DEFAULT_PREPROCESSING,
        class_weights=uniform_weights(),
        untrained=("neckline",),
    )
    return destination, model


def test_a_checkpoint_round_trips(checkpoint) -> None:
    destination, model = checkpoint
    restored, metadata = cnn.load_checkpoint(destination)
    assert torch.equal(restored.heads["upper_fabric"].weight, model.heads["upper_fabric"].weight)
    assert not restored.training
    assert metadata["backbone"] == "tiny"
    assert metadata["untrained_heads"] == ["neckline"]
    assert metadata["feature_dim"] == 4
    assert "state_dict" not in metadata


def test_a_checkpoint_accepts_the_preprocessing_it_was_written_with(checkpoint) -> None:
    destination, _ = checkpoint
    restored, _ = cnn.load_checkpoint(destination, preprocessing=cnn.DEFAULT_PREPROCESSING)
    assert list(restored.heads) == list(cnn.HEAD_NAMES)


def test_a_checkpoint_rejects_different_preprocessing(checkpoint) -> None:
    destination, _ = checkpoint
    other = {**cnn.DEFAULT_PREPROCESSING, "image_size": 256}
    with pytest.raises(cnn.CheckpointError, match="preprocessing"):
        cnn.load_checkpoint(destination, preprocessing=other)


def rewrite(destination, **changes):
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    payload.update(changes)
    torch.save(payload, destination)


def test_a_checkpoint_rejects_a_reordered_class_map(checkpoint) -> None:
    destination, _ = checkpoint
    maps = {head: list(values) for head, values in cnn.CLASS_MAPS.items()}
    maps["upper_fabric"] = list(reversed(maps["upper_fabric"]))
    rewrite(destination, class_maps=maps)
    with pytest.raises(cnn.CheckpointError, match="class vocabulary"):
        cnn.load_checkpoint(destination)


def test_a_checkpoint_rejects_a_foreign_format(checkpoint) -> None:
    destination, _ = checkpoint
    rewrite(destination, format="somebody_elses_cnn")
    with pytest.raises(cnn.CheckpointError, match="somebody_elses_cnn"):
        cnn.load_checkpoint(destination)


def test_a_checkpoint_rejects_a_newer_version(checkpoint) -> None:
    destination, _ = checkpoint
    rewrite(destination, version=cnn.CHECKPOINT_VERSION + 1)
    with pytest.raises(cnn.CheckpointError, match="newer"):
        cnn.load_checkpoint(destination)


def test_a_checkpoint_rejects_a_missing_key(checkpoint) -> None:
    destination, _ = checkpoint
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    del payload["untrained_heads"]
    torch.save(payload, destination)
    with pytest.raises(cnn.CheckpointError, match="untrained_heads"):
        cnn.load_checkpoint(destination)


def test_a_checkpoint_rejects_a_backbone_of_the_wrong_width(checkpoint, monkeypatch) -> None:
    destination, _ = checkpoint
    monkeypatch.setattr(cnn, "build_backbone", lambda name, **options: (tiny_backbone(), 8))
    with pytest.raises(cnn.CheckpointError, match="wide"):
        cnn.load_checkpoint(destination)


def test_an_absent_checkpoint_names_its_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="cnn.pt"):
        cnn.load_checkpoint(tmp_path / "cnn.pt")
