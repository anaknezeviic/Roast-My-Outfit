"""Fail fast when the staged inputs a real run needs are missing."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rmo import paths
from rmo.perception.enrichment import mask_path

log = logging.getLogger(__name__)

__all__ = [
    "InputReport",
    "PreflightError",
    "photo_path",
    "report_for_ids",
    "require",
]


class PreflightError(RuntimeError):
    """Raised when a run is missing inputs it cannot proceed without."""


@dataclass(frozen=True, slots=True)
class InputReport:
    """Which staged photographs and parsing masks are missing for a set of image ids."""

    label: str
    total: int
    missing_photos: tuple[str, ...]
    missing_masks: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether every id has both a photograph and a parsing mask."""
        return not self.missing_photos and not self.missing_masks

    def summary(self) -> str:
        """Return a one-line description of the shortfall."""
        return (
            f"{self.label}: {self.total} ids, "
            f"{len(self.missing_photos)} photographs and "
            f"{len(self.missing_masks)} parsing masks missing"
        )


def photo_path(image_id: str) -> Path:
    """Return where the photograph for ``image_id`` is staged."""
    return paths.raw_dir() / "images" / f"{image_id}.jpg"


def report_for_ids(image_ids: Iterable[str], *, label: str = "ids") -> InputReport:
    """Inventory the photographs and parsing masks staged for ``image_ids``."""
    ids = list(image_ids)
    return InputReport(
        label=label,
        total=len(ids),
        missing_photos=tuple(name for name in ids if not photo_path(name).is_file()),
        missing_masks=tuple(name for name in ids if mask_path(name) is None),
    )


def require(report: InputReport, *, examples: int = 3) -> None:
    """Raise unless every id covered by ``report`` has both inputs staged."""
    if report.ok:
        return
    shortfall = report.missing_photos or report.missing_masks
    raise PreflightError(f"{report.summary()}; first missing: {', '.join(shortfall[:examples])}")
