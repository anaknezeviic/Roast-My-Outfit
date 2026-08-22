"""Stage DeepFashion-MultiModal into ``data/raw`` and report what is there.

Fetching is manual: the dataset sits behind a Google Drive consent screen. Point
``--from`` or ``$RMO_SOURCE_DIR`` at the unpacked download. DensePose and
``keypoints/`` are never staged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rmo import paths

log = logging.getLogger(__name__)

__all__ = ["AssetGroup", "GROUPS", "build_parser", "main"]

EXIT_NO_SOURCE = 2
EXIT_INCOMPLETE = 3

SKIPPED_ASSETS = ("DensePose", "densepose", "keypoints")

_SAMPLE_CHARS = 120


@dataclass(frozen=True)
class AssetGroup:
    """One group of dataset files. ``dest`` is relative to ``data/raw``."""

    key: str
    patterns: tuple[str, ...]
    dest: str
    description: str


GROUPS: tuple[AssetGroup, ...] = (
    AssetGroup(
        key="labels",
        patterns=(
            "labels/*.txt",
            "*_anno_*.txt",
            "*_ann.txt",
            "shape_anno*.txt",
        ),
        dest="labels",
        description="shape / fabric / pattern annotations (~575 KB, 3 files expected)",
    ),
    AssetGroup(
        key="captions",
        patterns=(
            "captions.json",
            "textual_descriptions.json",
            "*caption*.json",
            "*description*.json",
        ),
        dest="",
        description="free-text descriptions (~11 MB)",
    ),
    AssetGroup(
        key="parsing",
        patterns=("parsing/*.png", "segm/*.png"),
        dest="parsing",
        description="24-class segmentation masks (~90 MB, 12,701 files)",
    ),
    AssetGroup(
        key="images",
        patterns=("images/*.jpg", "images/**/*.jpg", "*.jpg"),
        dest="images",
        description="source photographs (~5.4 GB, 44,096 files)",
    ),
)

_GROUPS_BY_KEY = {group.key: group for group in GROUPS}

_MANUAL_INSTRUCTIONS = """\
No staging directory given.

Download DeepFashion-MultiModal from
https://github.com/yumingj/DeepFashion-MultiModal (the labels archive and the
captions JSON, ~11.5 MB, are enough to start), unpack it, then re-run with
--from <dir> or set $RMO_SOURCE_DIR. Use --verify to inventory data/raw without
copying anything.
"""


@dataclass(frozen=True)
class FileInfo:
    """Inventory record for one staged file."""

    relpath: str
    size_bytes: int
    lines: int | None = None
    json_entries: int | None = None
    json_kind: str | None = None
    sample: str | None = None

    @property
    def size_human(self) -> str:
        """Size in the largest unit that keeps the number readable."""
        size = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"


def resolve_source(explicit: str | None) -> Path | None:
    """Resolve the staging directory from ``explicit`` or ``$RMO_SOURCE_DIR``, else ``None``."""
    raw = explicit or os.environ.get("RMO_SOURCE_DIR")
    if not raw:
        return None
    source = Path(raw).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Staging source {source} does not exist.")
    return source


def find_assets(source: Path, group: AssetGroup) -> list[Path]:
    """Return the files in ``source`` matching ``group``, de-duplicated, order stable."""
    found: dict[Path, None] = {}
    for pattern in group.patterns:
        for candidate in sorted(source.glob(pattern)):
            if candidate.is_file() and not _is_skipped(candidate):
                found.setdefault(candidate.resolve(), None)
    return list(found)


def _is_skipped(path: Path) -> bool:
    """True if ``path`` sits under an asset that is never staged."""
    return any(part in SKIPPED_ASSETS for part in path.parts)


def copy_asset(src: Path, dst: Path, *, dry_run: bool = False) -> bool:
    """Copy ``src`` to ``dst`` unless a same-sized file is already there; True if copied."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    if dry_run:
        log.info("would copy %s -> %s", src.name, dst)
        return True
    paths.ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def stage_group(
    source: Path,
    group: AssetGroup,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Stage one asset group into ``data/raw``; returns ``(copied, skipped)``."""
    matches = find_assets(source, group)
    if not matches:
        log.warning(
            "no files matched group %r in %s (tried: %s)",
            group.key,
            source,
            ", ".join(group.patterns),
        )
        return (0, 0)

    if limit is not None:
        matches = matches[:limit]

    destination = paths.raw_dir() / group.dest if group.dest else paths.raw_dir()
    copied = skipped = 0
    for match in matches:
        if copy_asset(match, destination / match.name, dry_run=dry_run):
            copied += 1
        else:
            skipped += 1

    log.info(
        "%s: %d copied, %d already present -> %s",
        group.key,
        copied,
        skipped,
        destination,
    )
    return (copied, skipped)


def inspect_file(path: Path, root: Path) -> FileInfo:
    """Inventory one file: line count and first line for text, entry count for JSON."""
    relpath = path.relative_to(root).as_posix()
    size = path.stat().st_size

    if path.suffix.lower() == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.error("could not parse %s as JSON: %s", relpath, exc)
            return FileInfo(relpath, size, sample=f"UNPARSEABLE: {exc}")
        kind = type(payload).__name__
        entries = len(payload) if isinstance(payload, (dict, list)) else None
        sample = None
        if isinstance(payload, dict) and payload:
            first_key = next(iter(payload))
            sample = f"{first_key!r}: {str(payload[first_key])[:_SAMPLE_CHARS]}"
        return FileInfo(relpath, size, json_entries=entries, json_kind=kind, sample=sample)

    if path.suffix.lower() in {".txt", ".csv", ".jsonl"}:
        lines = 0
        first_line = ""
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index == 0:
                    first_line = line.rstrip("\n")[:_SAMPLE_CHARS]
                lines += 1
        return FileInfo(relpath, size, lines=lines, sample=first_line)

    return FileInfo(relpath, size)


def inventory(root: Path) -> list[FileInfo]:
    """Inspect every annotation file under ``root``; summarise bulk directories as counts."""
    if not root.is_dir():
        return []

    records: list[FileInfo] = []
    bulk_dirs = {"images", "parsing"}

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative_parts = path.relative_to(root).parts
        if relative_parts and relative_parts[0] in bulk_dirs:
            continue
        records.append(inspect_file(path, root))

    for bulk in sorted(bulk_dirs):
        directory = root / bulk
        if not directory.is_dir():
            continue
        files = [p for p in directory.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        records.append(
            FileInfo(
                relpath=f"{bulk}/",
                size_bytes=total,
                lines=None,
                sample=f"{len(files)} files",
            )
        )

    return records


def describe(record: FileInfo) -> str:
    """Return a one-line summary of an inventory record."""
    if record.json_entries is not None:
        detail = f"{record.json_entries:,} {record.json_kind}"
    elif record.lines is not None:
        detail = f"{record.lines:,} lines"
    else:
        detail = record.sample or ""
    return f"{record.relpath}  {record.size_human}{'  ' + detail if detail else ''}"


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI."""
    parser = argparse.ArgumentParser(
        prog="download_data",
        description=(
            "Stage DeepFashion-MultiModal into data/raw and report what is staged."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Asset groups:\n"
        + "\n".join(f"  {g.key:<10} {g.description}" for g in GROUPS),
    )
    parser.add_argument(
        "--from",
        dest="source",
        metavar="DIR",
        help="Directory holding the unpacked Google Drive download. "
        "Defaults to $RMO_SOURCE_DIR.",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Stage annotations and captions only (~11.5 MB). This is the default.",
    )
    parser.add_argument(
        "--parsing",
        action="store_true",
        help="Also stage the 24-class segmentation masks (~90 MB).",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="Also stage source photographs (~5.4 GB unless --limit is given).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Cap how many image files are staged.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Stage every group. Implies --parsing --images.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Skip staging entirely; just report what is already in data/raw.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be staged without copying anything.",
    )
    return parser


def _selected_groups(args: argparse.Namespace) -> list[AssetGroup]:
    """Map CLI flags onto asset groups."""
    if args.all:
        return list(GROUPS)

    keys = ["labels", "captions"]
    if args.parsing:
        keys.append("parsing")
    if args.images:
        keys.append("images")
    return [_GROUPS_BY_KEY[key] for key in keys]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.labels_only and (args.parsing or args.images or args.all):
        parser.error("--labels-only cannot be combined with --parsing, --images or --all")

    if args.limit is not None and not (args.images or args.all):
        log.warning("--limit only affects the images group; ignoring it.")

    missing: list[str] = []

    if not args.verify:
        try:
            source = resolve_source(args.source)
        except NotADirectoryError as exc:
            log.error("%s", exc)
            return EXIT_NO_SOURCE

        if source is None:
            sys.stdout.write(_MANUAL_INSTRUCTIONS)
            return EXIT_NO_SOURCE

        log.info("staging from %s", source)
        for group in _selected_groups(args):
            limit = args.limit if group.key == "images" else None
            copied, skipped = stage_group(
                source, group, limit=limit, dry_run=args.dry_run
            )
            if copied == 0 and skipped == 0:
                missing.append(group.key)

    if args.dry_run:
        log.info("dry run: nothing copied")
        return 0

    root = paths.raw_dir()
    records = inventory(root)
    for record in records:
        log.info("%s", describe(record))
    log.info("staged %d entries under %s", len(records), root)

    if missing:
        log.error("no files matched: %s", ", ".join(missing))
        return EXIT_INCOMPLETE

    return 0
