"""Stage registry and the end-to-end outfit pipeline.

Stage packages import this module, so nothing here may import them at module scope.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from rmo.imaging import ImageInput
    from rmo.perception.base import PerceptionModel
    from rmo.roast.base import RoastGenerator
    from rmo.schemas import OutfitDescription, OutfitScore, RoastOutput
    from rmo.scoring.base import ScoringModel

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PERCEPTION",
    "DEFAULT_SCORER",
    "DEFAULT_ROASTER",
    "register",
    "create",
    "registered_names",
    "OutfitRoaster",
    "build_parser",
    "main",
]

DEFAULT_PERCEPTION = "dummy_perception"
DEFAULT_SCORER = "dummy_scorer"
DEFAULT_ROASTER = "dummy_roaster"

_REGISTRY: dict[str, Callable[[], Any]] = {}

_STAGE_PACKAGES = ("rmo.perception", "rmo.scoring", "rmo.roast")

_Stage = TypeVar("_Stage")


def _import_stage_packages() -> None:
    """Import every stage package so the registry is complete."""
    for package in _STAGE_PACKAGES:
        importlib.import_module(package)


def register(name: str, factory: Callable[[], Any]) -> None:
    """Record ``factory`` under ``name``, raising if the name is taken."""
    if name in _REGISTRY:
        raise ValueError(f"Model name {name!r} is already registered; pick another one.")
    _REGISTRY[name] = factory


def create(name: str) -> Any:
    """Return a new instance of the model registered under ``name``."""
    _import_stage_packages()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}; expected one of {', '.join(sorted(_REGISTRY))}."
        )
    return _REGISTRY[name]()


def registered_names() -> list[str]:
    """Return every model name, sorted."""
    _import_stage_packages()
    return sorted(_REGISTRY)


def _resolve_stage(stage: _Stage | None, name: str) -> _Stage:
    """Return ``stage``, or the model registered under ``name`` when it is missing."""
    if stage is not None:
        return stage
    log.info("no stage given; falling back to %s", name)
    return create(name)


class OutfitRoaster:
    """Run one image through perception, scoring and roasting."""

    def __init__(
        self,
        perception: PerceptionModel | None = None,
        scorer: ScoringModel | None = None,
        roaster: RoastGenerator | None = None,
    ) -> None:
        self.perception = _resolve_stage(perception, DEFAULT_PERCEPTION)
        self.scorer = _resolve_stage(scorer, DEFAULT_SCORER)
        self.roaster = _resolve_stage(roaster, DEFAULT_ROASTER)

    def run(self, image: ImageInput) -> tuple[OutfitDescription, OutfitScore, RoastOutput]:
        """Return the description, the score and the roast for one image."""
        description = self.perception.predict(image)
        score = self.scorer.score(description)
        roast = self.roaster.generate(description, score)
        return description, score, roast


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI."""
    parser = argparse.ArgumentParser(
        prog="rmo.pipeline",
        description="Run one photograph through the perception, scoring and roast stages.",
    )
    parser.add_argument(
        "--image",
        metavar="PATH",
        required=True,
        help="Photograph to run through the stages.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the description and the score alongside the roast.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one photograph through every stage and write the result to stdout."""
    args = build_parser().parse_args(argv)
    description, score, roast = OutfitRoaster().run(args.image)

    if args.json:
        document = json.dumps(
            {
                "description": description.model_dump(mode="json"),
                "score": score.model_dump(mode="json"),
                "roast": roast.model_dump(mode="json"),
            }
        )
    else:
        document = roast.model_dump_json()

    sys.stdout.write(f"{document}\n")
    return 0


if __name__ == "__main__":
    # Under ``-m`` this file is __main__; the stage packages register into rmo.pipeline.
    from rmo.pipeline import main as entry_point

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    sys.exit(entry_point())
