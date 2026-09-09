"""Full dataset integrity audit for SmolVLM training.

The audit is intentionally stricter than the training loop.  It verifies frozen
split integrity, staged image/mask decodability and dimensions, canonical target
construction, structured-target parsing, and supervision distributions before a
long-running GPU job is allowed to start.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rmo import paths
from rmo.config import load_perception_config
from rmo.data.descriptions import load_outfit_table
from rmo.data.preflight import photo_path
from rmo.data.smolvlm_examples import build_training_example
from rmo.perception.enrichment import keep_labels, mask_path, mask_skeleton, slot_labels
from rmo.perception.postprocess import parse_description
from rmo.schemas import Fabric, GarmentSlot, LowerLength, Neckline, Pattern, SleeveLength
from rmo.splits import MANIFEST_NAME, SPLIT_NAMES, assert_split_disjoint, load_split

__all__ = [
    "DatasetPreflightError",
    "SplitAudit",
    "PreflightReport",
    "audit_splits",
    "format_preflight_report",
    "verify_split_manifest",
]

_FIELDS: tuple[str, ...] = (
    "color",
    "pattern",
    "fabric",
    "sleeve_length",
    "length",
    "neckline",
)

_REQUIRED_TABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "image_id",
        "is_full_body",
        "upper_fabric",
        "upper_pattern",
        "outer_fabric",
        "outer_pattern",
        "lower_fabric",
        "lower_pattern",
        "sleeve_length",
        "lower_length",
        "neckline",
        "has_shape",
        "caption",
    }
)

_VOCAB_COLUMNS: dict[str, frozenset[str]] = {
    **{name: frozenset(member.value for member in Fabric) for name in (
        "upper_fabric", "outer_fabric", "lower_fabric"
    )},
    **{name: frozenset(member.value for member in Pattern) for name in (
        "upper_pattern", "outer_pattern", "lower_pattern"
    )},
    "sleeve_length": frozenset(member.value for member in SleeveLength),
    "lower_length": frozenset(member.value for member in LowerLength),
    "neckline": frozenset(member.value for member in Neckline),
}


class DatasetPreflightError(RuntimeError):
    """Raised when the frozen split manifest is malformed or inconsistent."""


def _percentile(values: Sequence[int], fraction: float) -> int:
    """Return the nearest-rank percentile for a non-empty integer sequence."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _distribution(values: Sequence[int]) -> dict[str, int]:
    """Return compact deterministic length statistics."""
    if not values:
        return {"min": 0, "median": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "min": min(values),
        "median": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _split_file(name: str) -> Path:
    return paths.splits_dir() / f"{name}.txt"


def _raw_split_ids(name: str) -> list[str]:
    path = _split_file(name)
    if not path.is_file():
        raise DatasetPreflightError(f"Frozen split file does not exist: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_split_manifest() -> dict[str, Any]:
    """Verify frozen split counts, duplicate-free files, hashes and group disjointness."""
    manifest_path = paths.splits_dir() / MANIFEST_NAME
    if not manifest_path.is_file():
        raise DatasetPreflightError(f"Split manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetPreflightError(f"Cannot read split manifest {manifest_path}: {exc}") from exc

    counts = manifest.get("counts")
    hashes = manifest.get("sha256")
    if not isinstance(counts, dict) or not isinstance(hashes, dict):
        raise DatasetPreflightError("Split manifest must contain 'counts' and 'sha256' mappings.")

    verified_counts: dict[str, int] = {}
    verified_hashes: dict[str, str] = {}
    for name in SPLIT_NAMES:
        raw_ids = _raw_split_ids(name)
        duplicates = len(raw_ids) - len(set(raw_ids))
        if duplicates:
            raise DatasetPreflightError(
                f"Split {name!r} contains {duplicates} duplicate image-id line(s)."
            )
        expected_count = counts.get(name)
        if expected_count != len(raw_ids):
            raise DatasetPreflightError(
                f"Split {name!r} has {len(raw_ids)} ids but manifest records {expected_count!r}."
            )
        filename = f"{name}.txt"
        actual_hash = paths.file_sha256(_split_file(name))
        expected_hash = hashes.get(filename)
        if expected_hash != actual_hash:
            raise DatasetPreflightError(
                f"Split {name!r} SHA-256 mismatch: {actual_hash}, expected {expected_hash!r}."
            )
        verified_counts[name] = len(raw_ids)
        verified_hashes[filename] = actual_hash

    try:
        assert_split_disjoint()
    except AssertionError as exc:
        raise DatasetPreflightError(str(exc)) from exc

    return {
        "counts": verified_counts,
        "sha256": verified_hashes,
        "group_disjoint": True,
        "manifest_path": str(manifest_path),
    }


def _decode_dimensions(path: Path) -> tuple[int, int]:
    """Verify that PIL can decode ``path`` and return (width, height)."""
    with Image.open(path) as image:
        size = image.size
        image.verify()
    return size


def _mask_slots(
    path: Path,
    labels: Mapping[GarmentSlot, frozenset[int]],
) -> list[str]:
    """Return garment slots actually claimed by a decoded parsing mask."""
    with Image.open(path) as handle:
        array = np.asarray(handle)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"parsing mask must be 2D after channel selection; got {array.shape}")
    keep = frozenset().union(*labels.values())
    selected = keep_labels(array, keep)
    return [slot.value for slot, _ in mask_skeleton(selected, labels)]


def _row_vocab_error(row: Mapping[str, object]) -> str | None:
    """Return the first canonical annotation vocabulary violation in ``row``."""
    for column, allowed in _VOCAB_COLUMNS.items():
        value = str(row[column])
        if value not in allowed:
            return f"{column}={value!r} is outside the canonical vocabulary"
    return None


def _target_records(answer: str) -> list[tuple[str, str, dict[str, str]]]:
    """Parse the deterministic serializer grammar without fuzzy interpretation."""
    records: list[tuple[str, str, dict[str, str]]] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Target line has no slot delimiter: {line!r}")
        raw_slot, body = line.split(":", 1)
        parts = [part.strip() for part in body.split("|") if part.strip()]
        if not parts:
            raise ValueError(f"Target line has no category: {line!r}")
        category = parts[0]
        fields: dict[str, str] = {}
        for part in parts[1:]:
            if "=" not in part:
                raise ValueError(f"Target field is not keyed: {part!r}")
            name, value = (piece.strip() for piece in part.split("=", 1))
            if name not in _FIELDS:
                raise ValueError(f"Unexpected target field {name!r}")
            if not value:
                raise ValueError(f"Target field {name!r} has an empty value")
            if name in fields:
                raise ValueError(f"Target field {name!r} is repeated")
            fields[name] = value
        records.append((raw_slot.strip(), category, fields))
    if not records:
        raise ValueError("Target contains no garment records")
    return records


def _validate_target_round_trip(answer: str, *, image_id: str) -> None:
    """Ensure every serialized supervised field survives the production parser exactly."""
    expected = _target_records(answer)
    parsed = parse_description(answer, image_id=image_id, source_model="preflight")
    if len(parsed.garments) != len(expected):
        raise ValueError(
            f"parser returned {len(parsed.garments)} garments for {len(expected)} target records"
        )

    for index, ((slot, category, fields), garment) in enumerate(zip(expected, parsed.garments)):
        if garment.slot.value != slot:
            raise ValueError(
                f"record {index}: slot became {garment.slot.value!r}, expected {slot!r}"
            )
        if garment.category != category:
            raise ValueError(
                f"record {index}: category became {garment.category!r}, expected {category!r}"
            )
        for name, expected_value in fields.items():
            actual = getattr(garment, name)
            actual_value = getattr(actual, "value", actual)
            if actual_value != expected_value:
                raise ValueError(
                    f"record {index}: {name} became {actual_value!r}, expected {expected_value!r}"
                )


@dataclass(slots=True)
class SplitAudit:
    """Statistics and failures collected for one audited split."""

    split: str
    requested: int
    built: int = 0
    errors: Counter[str] = field(default_factory=Counter)
    error_examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    slot_counts: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)
    field_presence: Counter[str] = field(default_factory=Counter)
    field_values: dict[str, Counter[str]] = field(
        default_factory=lambda: {name: Counter() for name in _FIELDS}
    )
    garments_per_image: list[int] = field(default_factory=list)
    target_chars: list[int] = field(default_factory=list)
    target_lines: list[int] = field(default_factory=list)
    longest_targets: list[tuple[int, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.built == self.requested and not self.errors

    def add_error(self, kind: str, image_id: str, detail: str, *, examples: int = 5) -> None:
        self.errors[kind] += 1
        bucket = self.error_examples.setdefault(kind, [])
        if len(bucket) < examples:
            bucket.append(f"{image_id}: {detail}")

    def add_target(self, image_id: str, answer: str) -> None:
        records = _target_records(answer)
        self.built += 1
        self.garments_per_image.append(len(records))
        self.target_chars.append(len(answer))
        self.target_lines.append(len(records))
        self.longest_targets.append((len(answer), image_id))
        self.longest_targets = sorted(self.longest_targets, reverse=True)[:10]
        for slot, category, fields in records:
            self.slot_counts[slot] += 1
            self.category_counts[category] += 1
            for name, value in fields.items():
                self.field_presence[f"{slot}.{name}"] += 1
                self.field_values[name][value] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "requested": self.requested,
            "built": self.built,
            "ok": self.ok,
            "errors": _counter_dict(self.errors),
            "error_examples": {key: value for key, value in sorted(self.error_examples.items())},
            "slot_counts": _counter_dict(self.slot_counts),
            "category_counts": _counter_dict(self.category_counts),
            "field_presence": _counter_dict(self.field_presence),
            "field_values": {
                name: _counter_dict(counter) for name, counter in self.field_values.items()
            },
            "garments_per_image": _distribution(self.garments_per_image),
            "target_characters": _distribution(self.target_chars),
            "target_lines": _distribution(self.target_lines),
            "longest_targets": [
                {"image_id": image_id, "characters": length}
                for length, image_id in self.longest_targets
            ],
        }


@dataclass(slots=True)
class PreflightReport:
    """Complete train/validation preflight report."""

    split_manifest: dict[str, Any]
    artifacts: dict[str, Any]
    splits: dict[str, SplitAudit]
    coverage_errors: list[str]
    coverage_warnings: list[str]

    @property
    def ok(self) -> bool:
        return all(report.ok for report in self.splits.values()) and not self.coverage_errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "split_manifest": self.split_manifest,
            "artifacts": self.artifacts,
            "coverage_errors": self.coverage_errors,
            "coverage_warnings": self.coverage_warnings,
            "splits": {name: report.as_dict() for name, report in self.splits.items()},
        }


def _coverage_findings(audits: Mapping[str, SplitAudit]) -> tuple[list[str], list[str]]:
    """Return fatal coverage gaps and non-fatal derived-colour warnings.

    Slots and source-annotated categorical attributes must be represented in
    training if they occur in validation.  ``color`` is different: it is a
    deterministic quantisation of continuous LAB measurements, so a very rare
    hue can legitimately land only in validation after a group-aware split.
    Such colours are reported with counts but do not invalidate the dataset.
    """
    train = audits.get("train")
    val = audits.get("val")
    if train is None or val is None:
        return [], []

    errors: list[str] = []
    warnings: list[str] = []
    unseen_slots = sorted(set(val.slot_counts) - set(train.slot_counts))
    if unseen_slots:
        errors.append(f"val has slot(s) absent from train: {', '.join(unseen_slots)}")

    for field_name in _FIELDS:
        train_values = set(train.field_values[field_name])
        val_counter = val.field_values[field_name]
        unseen = sorted(set(val_counter) - train_values)
        if not unseen:
            continue
        if field_name == "color":
            rendered = ", ".join(f"{value}={val_counter[value]}" for value in unseen)
            warnings.append(
                "val has derived color value(s) absent from train: " + rendered
            )
        else:
            errors.append(
                f"val has {field_name} value(s) absent from train: {', '.join(unseen)}"
            )
    return errors, warnings


def audit_splits(
    splits: Iterable[str] = ("train", "val"),
    *,
    progress_every: int = 500,
) -> PreflightReport:
    """Audit the requested frozen splits without silently skipping any example."""
    requested_splits = tuple(dict.fromkeys(splits))
    invalid = [name for name in requested_splits if name not in SPLIT_NAMES]
    if invalid:
        raise ValueError(f"Unknown split(s): {', '.join(invalid)}")

    manifest = verify_split_manifest()
    table_path = paths.processed_dir() / "outfits.parquet"
    table = load_outfit_table(table_path)
    missing_columns = _REQUIRED_TABLE_COLUMNS - set(table.columns)
    if missing_columns:
        raise DatasetPreflightError(
            f"outfits.parquet is missing required columns: {', '.join(sorted(missing_columns))}"
        )
    config_path = paths.configs_dir() / "perception.yaml"
    labels = slot_labels(load_perception_config(config_path))
    artifacts = {
        "git_sha": paths.git_sha(),
        "outfits_parquet": str(table_path),
        "outfits_parquet_sha256": paths.file_sha256(table_path),
        "outfits_rows": len(table),
        "perception_config": str(config_path),
        "perception_config_sha256": paths.file_sha256(config_path),
    }
    audits: dict[str, SplitAudit] = {}

    for split in requested_splits:
        image_ids = sorted(load_split(split))
        audit = SplitAudit(split=split, requested=len(image_ids))
        audits[split] = audit
        for position, image_id in enumerate(image_ids, start=1):
            if progress_every > 0 and position % progress_every == 0:
                print(f"[{split}] audited {position}/{len(image_ids)}")

            if image_id not in table.index:
                audit.add_error("missing_table_row", image_id, "absent from outfits.parquet")
                continue
            row = table.loc[image_id]
            if not bool(row["is_full_body"]):
                audit.add_error("not_full_body", image_id, "split row is not marked full-body")
                continue
            vocab_error = _row_vocab_error(row)
            if vocab_error is not None:
                audit.add_error("annotation_vocabulary", image_id, vocab_error)
                continue
            image_file = photo_path(image_id)
            if not image_file.is_file():
                audit.add_error("missing_image", image_id, str(image_file))
                continue
            parsing_file = mask_path(image_id)
            if parsing_file is None:
                audit.add_error("missing_mask", image_id, "no staged parsing PNG")
                continue

            try:
                image_size = _decode_dimensions(image_file)
            except Exception as exc:  # PIL exposes several decoder-specific exceptions
                audit.add_error("image_decode", image_id, str(exc))
                continue
            try:
                mask_size = _decode_dimensions(parsing_file)
            except Exception as exc:
                audit.add_error("mask_decode", image_id, str(exc))
                continue
            if image_size != mask_size:
                audit.add_error(
                    "dimension_mismatch",
                    image_id,
                    f"image={image_size}, mask={mask_size}",
                )
                continue
            try:
                expected_slots = _mask_slots(parsing_file, labels)
            except Exception as exc:
                audit.add_error("mask_labels", image_id, str(exc))
                continue
            if not expected_slots:
                audit.add_error(
                    "mask_no_garments", image_id, "mask contains no configured garment labels"
                )
                continue

            try:
                example = build_training_example(image_id, row, config_path=config_path)
            except Exception as exc:
                audit.add_error("target_build", image_id, str(exc))
                continue
            try:
                target_slots = [slot for slot, _, _ in _target_records(example.answer)]
                if target_slots != expected_slots:
                    raise ValueError(
                        f"target slots {target_slots!r} do not match mask slots {expected_slots!r}"
                    )
                _validate_target_round_trip(example.answer, image_id=image_id)
            except Exception as exc:
                audit.add_error("target_round_trip", image_id, str(exc))
                continue
            audit.add_target(image_id, example.answer)

    coverage_errors, coverage_warnings = _coverage_findings(audits)
    return PreflightReport(
        split_manifest=manifest,
        artifacts=artifacts,
        splits=audits,
        coverage_errors=coverage_errors,
        coverage_warnings=coverage_warnings,
    )


def format_preflight_report(report: PreflightReport) -> str:
    """Return a human-readable summary while keeping full details for JSON output."""
    lines = [
        f"SmolVLM dataset preflight: {'PASS' if report.ok else 'FAIL'}",
        "Frozen split manifest: PASS (counts, SHA-256, duplicate lines, group disjointness)",
        f"outfits.parquet SHA-256: {report.artifacts['outfits_parquet_sha256']}",
        f"perception.yaml SHA-256: {report.artifacts['perception_config_sha256']}",
    ]
    for name, audit in report.splits.items():
        lines.append("")
        lines.append(f"{name}: built {audit.built}/{audit.requested} examples")
        if audit.errors:
            lines.append("  errors:")
            for kind, count in sorted(audit.errors.items()):
                lines.append(f"    {kind}: {count}")
                for example in audit.error_examples.get(kind, []):
                    lines.append(f"      {example}")
        lines.append(f"  garments: {sum(audit.slot_counts.values())}")
        lines.append(f"  slot counts: {_counter_dict(audit.slot_counts)}")
        lines.append(f"  garments/image: {_distribution(audit.garments_per_image)}")
        lines.append(f"  target chars: {_distribution(audit.target_chars)}")
        lines.append(f"  target lines: {_distribution(audit.target_lines)}")
        lines.append(f"  supervised field presence: {_counter_dict(audit.field_presence)}")
        lines.append("  field values:")
        for field_name in _FIELDS:
            lines.append(f"    {field_name}: {_counter_dict(audit.field_values[field_name])}")
        if audit.longest_targets:
            rendered = ", ".join(
                f"{image_id} ({length})" for length, image_id in audit.longest_targets[:5]
            )
            lines.append(f"  longest targets: {rendered}")

    lines.append("")
    if report.coverage_errors:
        lines.append("Train/val coverage: FAIL")
        lines.extend(f"  {message}" for message in report.coverage_errors)
    else:
        lines.append("Train/val coverage: PASS")
    if report.coverage_warnings:
        lines.append("Train/val coverage warnings:")
        lines.extend(f"  {message}" for message in report.coverage_warnings)
    return "\n".join(lines)
