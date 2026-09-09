"""Run the strict train/validation data gate before SmolVLM training."""

from __future__ import annotations

import argparse
from pathlib import Path

from rmo import paths
from rmo.data.smolvlm_preflight import audit_splits, format_preflight_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=("train", "val", "test"),
        help="splits to audit; defaults to train and val so test remains untouched",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional JSON report path",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="print progress after this many examples; 0 disables progress",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_splits(args.splits, progress_every=args.progress_every)
    print(format_preflight_report(report))
    if args.json is not None:
        paths.write_json_atomic(args.json, report.as_dict())
        print(f"JSON report: {args.json}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
