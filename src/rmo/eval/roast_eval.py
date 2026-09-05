"""Text-grounding and safety metrics for roast generators."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rmo import paths
from rmo.eval.metrics import metric_record, write_metric_record
from rmo.roast.safety import flag_text
from rmo.schemas import OutfitDescription, RoastOutput
from rmo.splits import SPLIT_NAMES

log = logging.getLogger(__name__)

__all__ = [
    "GroundingResult",
    "SafetyResult",
    "evaluate_grounding",
    "evaluate_safety",
    "load_probes",
    "log_evaluation",
    "roast_metric_record",
]

PROBES_NAME = "safety_probes.jsonl"


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """How well the roast text stays anchored to the garments it was given."""

    grounded_rate: float
    unsupported_rate: float
    empty_grounding_rate: float
    mean_grounded_refs: float
    n_roasts: int
    n_grounded_refs: int
    n_unsupported_refs: int

    def as_metrics(self) -> dict[str, Any]:
        """Return the block a metric record carries for grounding."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SafetyResult:
    """How the filter behaves on the probe corpus."""

    recall: float | None
    false_positive_rate: float | None
    n_probes: int
    n_must_flag: int
    n_negative: int
    n_flagged: int
    missed: tuple[str, ...]
    false_positives: tuple[str, ...]

    def as_metrics(self) -> dict[str, Any]:
        """Return the block a metric record carries for safety."""
        block = asdict(self)
        block["missed"] = list(self.missed)
        block["false_positives"] = list(self.false_positives)
        return block


def evaluate_grounding(
    roasts: Sequence[RoastOutput], descriptions: Mapping[str, OutfitDescription]
) -> GroundingResult:
    """Measure whether every grounded reference names a garment that was supplied."""
    if not roasts:
        raise ValueError("A grounding evaluation needs at least one roast.")

    grounded = 0
    unsupported = 0
    empty = 0
    fully_supported = 0
    for roast in roasts:
        description = descriptions.get(roast.image_id)
        if description is None:
            raise ValueError(f"No description accompanies the roast for {roast.image_id!r}.")
        available = {garment.ref for garment in description.garments}
        refs = list(roast.grounded_garments)
        if not refs:
            empty += 1
            continue
        missing = [ref for ref in refs if ref not in available]
        grounded += len(refs) - len(missing)
        unsupported += len(missing)
        if not missing:
            fully_supported += 1

    total_refs = grounded + unsupported
    return GroundingResult(
        grounded_rate=fully_supported / len(roasts),
        unsupported_rate=unsupported / total_refs if total_refs else 0.0,
        empty_grounding_rate=empty / len(roasts),
        mean_grounded_refs=total_refs / len(roasts),
        n_roasts=len(roasts),
        n_grounded_refs=grounded,
        n_unsupported_refs=unsupported,
    )


def load_probes(source: Path | None = None) -> list[dict[str, Any]]:
    """Return the committed safety probe corpus."""
    path = paths.fixtures_dir() / PROBES_NAME if source is None else Path(source)
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"The probe corpus at {path} is empty.")
    return records


def evaluate_safety(probes: Sequence[Mapping[str, Any]]) -> SafetyResult:
    """Measure probe recall and the false-positive rate of the safety filter."""
    if not probes:
        raise ValueError("A safety evaluation needs at least one probe.")

    missed: list[str] = []
    false_positives: list[str] = []
    must_flag = 0
    negative = 0
    flagged = 0
    for probe in probes:
        expected = bool(probe["must_flag"])
        raised = bool(flag_text(str(probe["text"])))
        flagged += int(raised)
        if expected:
            must_flag += 1
            if not raised:
                missed.append(str(probe["probe_id"]))
        else:
            negative += 1
            if raised:
                false_positives.append(str(probe["probe_id"]))

    return SafetyResult(
        recall=(must_flag - len(missed)) / must_flag if must_flag else None,
        false_positive_rate=len(false_positives) / negative if negative else None,
        n_probes=len(probes),
        n_must_flag=must_flag,
        n_negative=negative,
        n_flagged=flagged,
        missed=tuple(missed),
        false_positives=tuple(false_positives),
    )


def log_evaluation(grounding: GroundingResult, safety: SafetyResult) -> None:
    """Log one line per measured quantity."""
    log.info(
        "grounded_rate=%.4f unsupported_rate=%.4f empty_grounding_rate=%.4f over %d roasts",
        grounding.grounded_rate,
        grounding.unsupported_rate,
        grounding.empty_grounding_rate,
        grounding.n_roasts,
    )
    log.info(
        "safety recall=%s false_positive_rate=%s over %d probes",
        safety.recall,
        safety.false_positive_rate,
        safety.n_probes,
    )


def roast_metric_record(
    grounding: GroundingResult,
    safety: SafetyResult,
    *,
    model: str,
    split: str,
    config: Mapping[str, Any],
    seed: int | None = None,
    inputs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the shared run stamp around one roast evaluation."""
    return metric_record(
        stage="roast",
        model=model,
        split=split,
        n_items=grounding.n_roasts,
        metrics={
            "grounding": grounding.as_metrics(),
            "safety": safety.as_metrics(),
            "note": (
                "Grounding and safety are measured; roast quality is not. "
                "No human ratings were collected."
            ),
        },
        config=config,
        seed=seed,
        inputs=inputs,
    )


def _metrics_filename(model_name: str, split: str) -> str:
    """Return a legal filename for one roaster and split."""
    return f"roast_{model_name.replace('/', '_')}_{split}.json"


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a registered roaster on grounding and on the safety probes."""
    from rmo.data.descriptions import describe_split, load_outfit_table
    from rmo.pipeline import create

    parser = argparse.ArgumentParser(description="Evaluate roast grounding and safety.")
    parser.add_argument("--roaster", default="rule_roaster")
    parser.add_argument("--scorer", default="rule_scorer_v1")
    parser.add_argument("--split", default="val", choices=SPLIT_NAMES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    descriptions = list(
        describe_split(args.split, limit=args.limit, table=load_outfit_table(args.table))
    )

    scorer = create(args.scorer)
    roaster = create(args.roaster)
    roasts = [
        roaster.generate(description, scorer.score(description))
        for description in descriptions
    ]
    grounding = evaluate_grounding(roasts, {item.image_id: item for item in descriptions})
    safety = evaluate_safety(load_probes())
    log_evaluation(grounding, safety)

    destination = args.metrics_out or paths.metrics_dir() / _metrics_filename(
        roaster.name, args.split
    )
    write_metric_record(
        roast_metric_record(
            grounding,
            safety,
            model=roaster.name,
            split=args.split,
            config={"roaster": roaster.name, "scorer": scorer.name},
        ),
        destination,
    )
    log.info("wrote roast metrics to %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
