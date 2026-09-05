"""Cover split-local pair construction and its attrition accounting."""

from __future__ import annotations

import numpy as np
import pytest

from rmo.schemas import (
    ColorName,
    Fabric,
    Garment,
    GarmentSlot,
    OutfitDescription,
    Pattern,
    Provenance,
)
from rmo.scoring.features import build_spec
from rmo.scoring.pairs import (
    DEFAULT_SWAPPABLE,
    PairKind,
    attrition_report,
    build_pairs,
    eligible_slots,
    pair_matrix,
)
from rmo.splits import group_key_for

SEED = 20260101


def image_id(group: int, shot: int = 1) -> str:
    return f"WOMEN-Blouses_Shirts-id_{group:08d}-0{shot}_1_front"


def garment(slot: GarmentSlot, color: ColorName, *, pattern: Pattern = Pattern.pure_color) -> Garment:
    return Garment(
        slot=slot,
        category=slot.value,
        color=color,
        pattern=pattern,
        fabric=Fabric.cotton,
        area_fraction=0.3,
    )


def description(
    group: int,
    *,
    shot: int = 1,
    colors: dict[GarmentSlot, ColorName] | None = None,
    slots: tuple[GarmentSlot, ...] = (GarmentSlot.upper, GarmentSlot.lower, GarmentSlot.footwear),
) -> OutfitDescription:
    palette = colors or {}
    return OutfitDescription(
        image_id=image_id(group, shot),
        image_path=f"data/raw/images/{image_id(group, shot)}.jpg",
        source_model="dataset_labels",
        provenance=Provenance.gt,
        garments=[
            garment(slot, palette.get(slot, ColorName(list(ColorName)[group % 19].value)))
            for slot in slots
        ],
    )


def population(count: int = 8) -> list[OutfitDescription]:
    return [description(group) for group in range(1, count + 1)]


def test_eligible_slots_report_only_the_filled_swappable_ones() -> None:
    outfit = description(1, slots=(GarmentSlot.upper, GarmentSlot.bag))
    assert eligible_slots(outfit) == (GarmentSlot.upper,)


def test_eligible_slots_follow_the_configured_order() -> None:
    outfit = description(
        1, slots=(GarmentSlot.footwear, GarmentSlot.lower, GarmentSlot.upper)
    )
    assert eligible_slots(outfit) == (
        GarmentSlot.upper,
        GarmentSlot.lower,
        GarmentSlot.footwear,
    )


def test_building_needs_at_least_one_description() -> None:
    with pytest.raises(ValueError, match="at least one description"):
        build_pairs([], seed=SEED)


def test_every_outfit_contributes_an_observed_record() -> None:
    built = build_pairs(population(), seed=SEED)
    observed = built.of_kind(PairKind.observed)
    assert len(observed) == 8
    assert all(record.label == 1 for record in observed)


def test_a_negative_carries_the_zero_label() -> None:
    built = build_pairs(population(), seed=SEED)
    assert all(record.label == 0 for record in built.of_kind(PairKind.hard))
    assert all(record.label == 0 for record in built.of_kind(PairKind.easy))


def test_a_hard_negative_changes_exactly_one_slot() -> None:
    built = build_pairs(population(), seed=SEED)
    for record in built.of_kind(PairKind.hard):
        assert len(record.donors) == 1


def test_an_easy_negative_changes_every_eligible_slot() -> None:
    built = build_pairs(population(), seed=SEED)
    for record in built.of_kind(PairKind.easy):
        original = next(
            item
            for item in built.of_kind(PairKind.observed)
            if item.recipient_id == record.recipient_id
        )
        assert len(record.donors) == len(eligible_slots(original.description))


def test_an_easy_negative_uses_at_least_two_donor_groups() -> None:
    built = build_pairs(population(), seed=SEED)
    for record in built.of_kind(PairKind.easy):
        groups = {group_key_for(donor_id) for _, donor_id in record.donors}
        assert len(groups) >= 2


def test_no_donor_comes_from_the_recipient_product_group() -> None:
    built = build_pairs(population(), seed=SEED)
    for record in built.records:
        recipient_group = group_key_for(record.recipient_id)
        for _, donor_id in record.donors:
            assert group_key_for(donor_id) != recipient_group


def test_a_swap_actually_changes_the_recipient() -> None:
    built = build_pairs(population(), seed=SEED)
    observed = {
        record.recipient_id: record.description
        for record in built.of_kind(PairKind.observed)
    }
    for record in built.records:
        if record.kind is PairKind.observed:
            continue
        before = observed[record.recipient_id]
        assert [g.color for g in record.description.garments] != [
            g.color for g in before.garments
        ]


def test_a_negative_keeps_the_recipient_slot_layout() -> None:
    built = build_pairs(population(), seed=SEED)
    observed = {
        record.recipient_id: record.description
        for record in built.of_kind(PairKind.observed)
    }
    for record in built.records:
        before = observed[record.recipient_id]
        assert [g.slot for g in record.description.garments] == [
            g.slot for g in before.garments
        ]


def test_a_negative_keeps_the_recipient_refs() -> None:
    built = build_pairs(population(), seed=SEED)
    observed = {
        record.recipient_id: record.description
        for record in built.of_kind(PairKind.observed)
    }
    for record in built.records:
        before = observed[record.recipient_id]
        assert [g.ref for g in record.description.garments] == [
            g.ref for g in before.garments
        ]


def test_a_negative_is_keyed_apart_from_the_outfit_it_came_from() -> None:
    built = build_pairs(population(), seed=SEED)
    ids = [record.description.image_id for record in built.records]
    assert len(set(ids)) == len(ids)
    for record in built.of_kind(PairKind.hard):
        assert record.description.image_id == f"{record.recipient_id}#hard"


def test_building_does_not_mutate_the_observed_description() -> None:
    outfits = population()
    before = [outfit.model_dump_json() for outfit in outfits]
    build_pairs(outfits, seed=SEED)
    assert [outfit.model_dump_json() for outfit in outfits] == before


def test_one_seed_builds_one_corpus() -> None:
    first = build_pairs(population(), seed=SEED)
    second = build_pairs(population(), seed=SEED)
    assert [record.donors for record in first.records] == [
        record.donors for record in second.records
    ]


def test_a_different_seed_draws_different_donors() -> None:
    first = build_pairs(population(), seed=SEED)
    second = build_pairs(population(), seed=SEED + 1)
    assert [record.donors for record in first.records] != [
        record.donors for record in second.records
    ]


def test_records_are_ordered_by_recipient() -> None:
    built = build_pairs(population(), seed=SEED)
    recipients = [record.recipient_id for record in built.records]
    assert recipients == sorted(recipients)


def test_an_outfit_with_no_swappable_slot_is_counted_as_attrition() -> None:
    outfits = [*population(4), description(9, slots=(GarmentSlot.bag,))]
    built = build_pairs(outfits, seed=SEED)
    assert built.attrition["no_eligible_slot"] == 1
    assert all(record.recipient_id != image_id(9) for record in built.records)


def test_a_lone_product_group_yields_no_negative() -> None:
    built = build_pairs([description(1), description(1, shot=2)], seed=SEED)
    assert built.of_kind(PairKind.hard) == ()
    assert built.attrition["hard_no_donor"] == 2


def test_an_identical_population_cannot_produce_a_no_op_swap() -> None:
    outfits = [
        description(group, colors={slot: ColorName.black for slot in DEFAULT_SWAPPABLE})
        for group in range(1, 5)
    ]
    built = build_pairs(outfits, seed=SEED)
    assert built.of_kind(PairKind.hard) == ()
    assert built.attrition["hard_no_donor"] == 4


def test_the_attrition_report_states_every_count() -> None:
    built = build_pairs(population(), seed=SEED)
    report = attrition_report(built)
    assert report["n_observed"] == 8
    assert report["seed"] == SEED
    assert report["swappable"] == [slot.value for slot in DEFAULT_SWAPPABLE]
    assert set(report) == {
        "attrition",
        "n_easy",
        "n_hard",
        "n_observed",
        "seed",
        "swappable",
    }


def test_the_matrix_has_one_row_per_record() -> None:
    built = build_pairs(population(), seed=SEED)
    spec = build_spec()
    features, labels, groups = pair_matrix(built.records, spec)
    assert features.shape[0] == len(built.records)
    assert labels.shape == (len(built.records),)
    assert len(groups) == len(built.records)


def test_the_matrix_groups_a_negative_with_its_recipient() -> None:
    built = build_pairs(population(), seed=SEED)
    _, _, groups = pair_matrix(built.records, build_spec())
    for record, group in zip(built.records, groups, strict=True):
        assert group == group_key_for(record.recipient_id)


def test_the_matrix_labels_match_the_record_kinds() -> None:
    built = build_pairs(population(), seed=SEED)
    _, labels, _ = pair_matrix(built.records, build_spec())
    assert labels.tolist() == [record.label for record in built.records]
    assert set(np.unique(labels)) <= {0, 1}


def test_an_empty_matrix_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        pair_matrix([], build_spec())
