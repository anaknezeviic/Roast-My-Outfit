"""Audit semantic quality of colour labels used for SmolVLM ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

from rmo import paths
from rmo.data.color_ground_truth_audit import audit_color_descriptions, format_color_audit
from rmo.data.descriptions import describe_image, load_outfit_table
from rmo.splits import load_split


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "val"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None, help="Audit the first N sorted IDs (debug only).")
    selection.add_argument("--sample", type=int, default=None, help="Audit a deterministic random sample of N split IDs.")
    parser.add_argument("--seed", type=int, default=20260909, help="Random seed used by --sample.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress after every N audited images; 0 disables progress.")
    parser.add_argument("--high-chroma", type=float, default=25.0)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--json", type=Path, default=None)
    return parser


def _selected_ids(split: str, *, limit: int | None, sample: int | None, seed: int) -> list[str]:
    image_ids = sorted(load_split(split))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        return image_ids[:limit]
    if sample is not None:
        if sample < 1:
            raise ValueError("--sample must be positive.")
        if sample > len(image_ids):
            raise ValueError(f"--sample={sample} exceeds split size {len(image_ids)}.")
        chosen = random.Random(seed).sample(image_ids, sample)
        return sorted(chosen)
    return image_ids


def _selection_sha256(image_ids: Sequence[str]) -> str:
    payload = "".join(f"{image_id}\n" for image_id in image_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _descriptions_with_progress(
    image_ids: Sequence[str],
    frame: pd.DataFrame,
    *,
    progress_every: int,
) -> Iterator:
    total = len(image_ids)
    for index, image_id in enumerate(image_ids, start=1):
        description = describe_image(image_id, frame.loc[image_id])
        if progress_every and (index % progress_every == 0 or index == total):
            print(f"[color-audit] audited {index}/{total}", flush=True)
        yield description


def main() -> int:
    args = _parser().parse_args()
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative.")
    image_ids = _selected_ids(args.split, limit=args.limit, sample=args.sample, seed=args.seed)
    frame = load_outfit_table()
    missing = [image_id for image_id in image_ids if image_id not in frame.index]
    if missing:
        raise SystemExit(f"{len(missing)} selected IDs are absent from outfits.parquet; first: {missing[0]}")

    mode = "full split"
    if args.sample is not None:
        mode = f"random sample (seed={args.seed})"
    elif args.limit is not None:
        mode = "sorted prefix"
    print(f"Color audit: {args.split} {mode}, {len(image_ids)} images", flush=True)

    descriptions = _descriptions_with_progress(
        image_ids, frame, progress_every=args.progress_every
    )
    report = audit_color_descriptions(
        descriptions,
        high_chroma_threshold=args.high_chroma,
        max_examples=args.examples,
    )
    print(format_color_audit(report))
    if args.json is not None:
        payload = {
            "split": args.split,
            "limit": args.limit,
            "sample": args.sample,
            "seed": args.seed,
            "selected_images": len(image_ids),
            "selection_sha256": _selection_sha256(image_ids),
            "high_chroma_threshold": args.high_chroma,
            "outfits_parquet_sha256": paths.file_sha256(
                paths.processed_dir() / "outfits.parquet"
            ),
            "report": report.to_dict(),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"JSON report: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
