#!/usr/bin/env python
"""Measure ground-truth ambiguities that must be resolved before SmolVLM training."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rmo.data.ground_truth_audit import GroundTruthAuditError, audit_ground_truth
from rmo.splits import SPLIT_NAMES, load_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_smolvlm_ground_truth",
        description=(
            "Audit parsing-mask garment presence and dress/romper upper/lower label "
            "consistency before choosing SmolVLM ground-truth projection rules."
        ),
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLIT_NAMES,
        dest="splits",
        help="Audit one frozen split. Repeat for multiple splits; default: train only.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_path",
        help="Also write the complete audit report as JSON.",
    )
    return parser


def _format_pair(name: str, pair) -> list[str]:
    return [
        f"  {name}: total={pair.total}",
        f"    both_na={pair.both_na}",
        f"    upper_only={pair.upper_only}",
        f"    lower_only={pair.lower_only}",
        f"    same={pair.same}",
        f"    conflict={pair.conflict}",
    ]


def _print_report(report) -> None:
    print(f"Images audited: {report.images}")
    print("Slot presence:")
    for slot, count in report.slot_presence.items():
        if count:
            print(f"  {slot}: {count}")
    print("Regional garments present in mask but fabric=na and pattern=na:")
    for slot, count in report.regional_texture_both_na.items():
        print(f"  {slot}: {count}")
    for slot_name in ("dress", "romper"):
        item = getattr(report, slot_name)
        print(f"{slot_name}: images={item.images}, shape_annotated={item.shape_annotated}")
        for line in _format_pair("fabric upper/lower relation", item.fabric):
            print(line)
        for line in _format_pair("pattern upper/lower relation", item.pattern):
            print(line)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    splits = tuple(args.splits or ("train",))
    image_ids: list[str] = []
    for split in splits:
        image_ids.extend(sorted(load_split(split)))

    try:
        report = audit_ground_truth(image_ids)
    except (GroundTruthAuditError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _print_report(report)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
