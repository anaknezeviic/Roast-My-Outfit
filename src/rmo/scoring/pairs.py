"""Split-local compatibility pairs built by swapping garments between outfits."""

from __future__ import annotations

import logging
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from rmo.schemas import Garment, GarmentSlot, OutfitDescription
from rmo.scoring.features import FeatureSpec, describe_to_features
from rmo.splits import group_key_for

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SWAPPABLE",
    "PairKind",
    "PairRecord",
    "PairSet",
    "attrition_report",
    "build_pairs",
    "eligible_slots",
    "pair_matrix",
]

DEFAULT_SWAPPABLE: tuple[GarmentSlot, ...] = (
    GarmentSlot.upper,
    GarmentSlot.lower,
    GarmentSlot.outer,
    GarmentSlot.dress,
    GarmentSlot.footwear,
)

_EASY_MIN_DONOR_GROUPS = 2


class PairKind(StrEnum):
    """What one built record represents."""

    observed = "observed"
    hard = "hard"
    easy = "easy"


@dataclass(frozen=True, slots=True)
class PairRecord:
    """One outfit, either as observed or with donor garments substituted in."""

    recipient_id: str
    kind: PairKind
    description: OutfitDescription
    donors: tuple[tuple[str, str], ...]

    @property
    def label(self) -> int:
        """Return 1 for an observed combination and 0 for a synthesised one."""
        return 1 if self.kind is PairKind.observed else 0


@dataclass(frozen=True, slots=True)
class PairSet:
    """Built records with the attrition their construction incurred."""

    records: tuple[PairRecord, ...]
    attrition: dict[str, int]
    swappable: tuple[GarmentSlot, ...]
    seed: int

    def of_kind(self, kind: PairKind) -> tuple[PairRecord, ...]:
        """Return every record of one kind."""
        return tuple(record for record in self.records if record.kind is kind)


def eligible_slots(
    description: OutfitDescription, swappable: Sequence[GarmentSlot] = DEFAULT_SWAPPABLE
) -> tuple[GarmentSlot, ...]:
    """Return the swappable slots this outfit actually fills."""
    present = {garment.slot for garment in description.garments}
    return tuple(slot for slot in swappable if slot in present)


def _by_slot(description: OutfitDescription) -> dict[GarmentSlot, Garment]:
    """Return the first garment occupying each slot."""
    found: dict[GarmentSlot, Garment] = {}
    for garment in description.garments:
        found.setdefault(garment.slot, garment)
    return found


def _substitute(
    recipient: OutfitDescription,
    replacements: Mapping[GarmentSlot, Garment],
    *,
    kind: PairKind,
) -> OutfitDescription:
    """Return a copy of ``recipient`` with the named slots taken from donor garments."""
    built = recipient.model_copy(deep=True)
    garments: list[Garment] = []
    filled: set[GarmentSlot] = set()
    for garment in built.garments:
        donor = replacements.get(garment.slot)
        if donor is None or garment.slot in filled:
            garments.append(garment)
            continue
        swapped = donor.model_copy(deep=True)
        swapped.ref = garment.ref
        garments.append(swapped)
        filled.add(garment.slot)
    built.garments = garments
    built.image_id = f"{recipient.image_id}#{kind.value}"
    return built


def _differs(original: Garment, donor: Garment) -> bool:
    """Return whether substituting ``donor`` would change the outfit at all."""
    fields = ("category", "color", "pattern", "fabric", "sleeve_length", "length", "neckline")
    return any(
        getattr(original, field, None) != getattr(donor, field, None) for field in fields
    )


def _donor_pool(
    descriptions: Sequence[OutfitDescription], swappable: Sequence[GarmentSlot]
) -> dict[GarmentSlot, list[tuple[str, Garment]]]:
    """Return every available donor garment, keyed by slot."""
    pool: dict[GarmentSlot, list[tuple[str, Garment]]] = {slot: [] for slot in swappable}
    for description in descriptions:
        for slot, garment in _by_slot(description).items():
            if slot in pool:
                pool[slot].append((description.image_id, garment))
    return pool


def _choose_donor(
    pool: Sequence[tuple[str, Garment]],
    *,
    recipient_id: str,
    recipient_group: str,
    original: Garment,
    rng: random.Random,
    max_attempts: int,
    forbidden_groups: frozenset[str] = frozenset(),
) -> tuple[str, Garment] | None:
    """Draw a donor from a different product group whose garment changes the outfit."""
    if not pool:
        return None
    for _ in range(max_attempts):
        donor_id, garment = pool[rng.randrange(len(pool))]
        if donor_id == recipient_id:
            continue
        group = group_key_for(donor_id)
        if group == recipient_group or group in forbidden_groups:
            continue
        if not _differs(original, garment):
            continue
        return donor_id, garment
    return None


def build_pairs(
    descriptions: Iterable[OutfitDescription],
    *,
    seed: int,
    swappable: Sequence[GarmentSlot] = DEFAULT_SWAPPABLE,
    max_attempts: int = 32,
) -> PairSet:
    """Return observed outfits with one hard and one easy negative built from donors."""
    ordered = sorted(descriptions, key=lambda record: record.image_id)
    if not ordered:
        raise ValueError("Building pairs needs at least one description.")

    slots = tuple(swappable)
    pool = _donor_pool(ordered, slots)
    rng = random.Random(seed)
    attrition: Counter[str] = Counter()
    records: list[PairRecord] = []

    for description in ordered:
        available = eligible_slots(description, slots)
        if not available:
            attrition["no_eligible_slot"] += 1
            continue

        garments = _by_slot(description)
        group = group_key_for(description.image_id)
        records.append(
            PairRecord(
                recipient_id=description.image_id,
                kind=PairKind.observed,
                description=description,
                donors=(),
            )
        )

        target = available[rng.randrange(len(available))]
        chosen = _choose_donor(
            pool[target],
            recipient_id=description.image_id,
            recipient_group=group,
            original=garments[target],
            rng=rng,
            max_attempts=max_attempts,
        )
        if chosen is None:
            attrition["hard_no_donor"] += 1
        else:
            donor_id, garment = chosen
            records.append(
                PairRecord(
                    recipient_id=description.image_id,
                    kind=PairKind.hard,
                    description=_substitute(
                        description, {target: garment}, kind=PairKind.hard
                    ),
                    donors=((target.value, donor_id),),
                )
            )

        replacements: dict[GarmentSlot, Garment] = {}
        donors: list[tuple[str, str]] = []
        donor_groups: set[str] = set()
        for slot in available:
            picked = _choose_donor(
                pool[slot],
                recipient_id=description.image_id,
                recipient_group=group,
                original=garments[slot],
                rng=rng,
                max_attempts=max_attempts,
            )
            if picked is None:
                continue
            donor_id, garment = picked
            replacements[slot] = garment
            donors.append((slot.value, donor_id))
            donor_groups.add(group_key_for(donor_id))

        if len(replacements) != len(available):
            attrition["easy_incomplete"] += 1
        elif len(donor_groups) < min(_EASY_MIN_DONOR_GROUPS, len(available)):
            attrition["easy_one_donor_group"] += 1
        else:
            records.append(
                PairRecord(
                    recipient_id=description.image_id,
                    kind=PairKind.easy,
                    description=_substitute(description, replacements, kind=PairKind.easy),
                    donors=tuple(donors),
                )
            )

    built = PairSet(
        records=tuple(records),
        attrition=dict(attrition),
        swappable=slots,
        seed=seed,
    )
    log.info(
        "built %d observed, %d hard and %d easy records with attrition %s",
        len(built.of_kind(PairKind.observed)),
        len(built.of_kind(PairKind.hard)),
        len(built.of_kind(PairKind.easy)),
        built.attrition or "none",
    )
    return built


def pair_matrix(
    records: Sequence[PairRecord], spec: FeatureSpec
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return the feature matrix, binary labels and recipient groups of ``records``."""
    if not records:
        raise ValueError("A feature matrix needs at least one record.")
    features = np.vstack(
        [describe_to_features(record.description, spec) for record in records]
    )
    labels = np.array([record.label for record in records], dtype=np.int64)
    groups = [group_key_for(record.recipient_id) for record in records]
    return features, labels, groups


def attrition_report(built: PairSet) -> dict[str, Any]:
    """Return the counts a results table needs to state pair construction honestly."""
    return {
        "attrition": dict(built.attrition),
        "n_easy": len(built.of_kind(PairKind.easy)),
        "n_hard": len(built.of_kind(PairKind.hard)),
        "n_observed": len(built.of_kind(PairKind.observed)),
        "seed": built.seed,
        "swappable": [slot.value for slot in built.swappable],
    }
