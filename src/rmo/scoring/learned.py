"""Learned compatibility scorers fitted on split-local pairs."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rmo import paths
from rmo.schemas import OutfitDescription, OutfitScore
from rmo.scoring.base import ScoringModel
from rmo.scoring.features import (
    FeatureSpec,
    build_spec,
    describe_to_features,
    feature_contract,
    verify_contract,
)
from rmo.scoring.missingness import positive_column
from rmo.scoring.rules import RuleScorer

log = logging.getLogger(__name__)

__all__ = [
    "BUNDLE_VERSION",
    "KINDS",
    "BundleError",
    "LearnedScorer",
    "ScorerBundle",
    "fit_scorer",
    "load_bundle",
    "save_bundle",
]

BUNDLE_VERSION = 1

KINDS: tuple[str, ...] = ("logreg",)

_BUNDLE_KEYS: tuple[str, ...] = (
    "bundle_version",
    "contract",
    "feature_names",
    "intercept",
    "kind",
    "mean",
    "n_fit",
    "scale",
    "seed",
    "weights",
)


class BundleError(ValueError):
    """Raised when a saved scorer bundle cannot be trusted."""


@dataclass(frozen=True, slots=True)
class ScorerBundle:
    """A fitted linear scorer with the feature order it was fitted under."""

    kind: str
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    intercept: float
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    classes: tuple[int, ...]
    contract: Mapping[str, Any]
    seed: int
    n_fit: int

    def probability(self, features: np.ndarray) -> np.ndarray:
        """Return the probability each row is an observed combination."""
        matrix = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if matrix.shape[1] != len(self.feature_names):
            raise BundleError(
                f"This bundle expects {len(self.feature_names)} features, "
                f"got {matrix.shape[1]}."
            )
        if not np.all(np.isfinite(matrix)):
            raise BundleError("A feature row carries a non-finite value.")
        standardised = (matrix - np.asarray(self.mean)) / np.asarray(self.scale)
        logits = standardised @ np.asarray(self.weights) + self.intercept
        observed = 1.0 / (1.0 + np.exp(-logits))
        if positive_column(self.classes) == 0:
            observed = 1.0 - observed
        if not np.all(np.isfinite(observed)):
            raise BundleError("The fitted scorer produced a non-finite probability.")
        return observed


def fit_scorer(
    features: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    spec: FeatureSpec,
    *,
    kind: str = "logreg",
    seed: int,
) -> ScorerBundle:
    """Fit a scorer on training rows only and return its portable bundle."""
    if kind not in KINDS:
        raise ValueError(f"Unknown scorer {kind!r}; expected one of {', '.join(KINDS)}.")
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError(f"A feature matrix must be two dimensional, got {matrix.ndim}.")
    if matrix.shape[0] != target.shape[0]:
        raise ValueError(
            f"{matrix.shape[0]} feature rows do not match {target.shape[0]} labels."
        )
    if matrix.shape[1] != len(spec.names):
        raise ValueError(
            f"The matrix is {matrix.shape[1]} features wide, "
            f"but this specification builds {len(spec.names)}."
        )
    if len(np.unique(target)) < 2:
        raise ValueError("Fitting a scorer needs both an observed and a synthesised class.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(matrix)
    estimator = LogisticRegression(max_iter=2000, random_state=seed)
    estimator.fit(scaler.transform(matrix), target)
    classes = tuple(int(value) for value in estimator.classes_)
    positive_column(classes)

    return ScorerBundle(
        kind=kind,
        feature_names=tuple(spec.names),
        weights=tuple(float(value) for value in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
        mean=tuple(float(value) for value in scaler.mean_),
        scale=tuple(float(value) for value in scaler.scale_),
        classes=classes,
        contract=feature_contract(spec),
        seed=seed,
        n_fit=int(matrix.shape[0]),
    )


def save_bundle(bundle: ScorerBundle, destination: Path) -> Path:
    """Write a fitted bundle as sorted JSON and return its path."""
    payload = {
        "bundle_version": BUNDLE_VERSION,
        "classes": list(bundle.classes),
        "contract": dict(bundle.contract),
        "feature_names": list(bundle.feature_names),
        "intercept": bundle.intercept,
        "kind": bundle.kind,
        "mean": list(bundle.mean),
        "n_fit": bundle.n_fit,
        "scale": list(bundle.scale),
        "seed": bundle.seed,
        "weights": list(bundle.weights),
    }
    return paths.write_json_atomic(destination, payload)


def load_bundle(source: Path, spec: FeatureSpec | None = None) -> ScorerBundle:
    """Read a bundle and refuse it unless its feature contract still holds."""
    try:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"Could not read a scorer bundle from {source}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise BundleError(f"The scorer bundle at {source} is not a JSON object.")

    missing = [key for key in _BUNDLE_KEYS if key not in payload]
    if missing:
        raise BundleError(f"The scorer bundle at {source} is missing {', '.join(missing)}.")
    version = payload["bundle_version"]
    if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version <= BUNDLE_VERSION:
        raise BundleError(
            f"The scorer bundle at {source} declares version {version!r}, "
            f"and this reader understands {BUNDLE_VERSION}."
        )

    resolved = build_spec() if spec is None else spec
    verify_contract(payload["contract"], resolved)
    if list(payload["feature_names"]) != list(resolved.names):
        raise BundleError(f"The scorer bundle at {source} was fitted on another feature order.")

    widths = {
        len(payload["weights"]),
        len(payload["mean"]),
        len(payload["scale"]),
        len(resolved.names),
    }
    if len(widths) != 1:
        raise BundleError(f"The scorer bundle at {source} has inconsistent vector widths.")
    if not all(float(value) > 0.0 for value in payload["scale"]):
        raise BundleError(
            f"The scorer bundle at {source} carries a non-positive scale, so its fitted "
            "preprocessing is missing or corrupt."
        )

    return ScorerBundle(
        kind=str(payload["kind"]),
        feature_names=tuple(payload["feature_names"]),
        weights=tuple(float(value) for value in payload["weights"]),
        intercept=float(payload["intercept"]),
        mean=tuple(float(value) for value in payload["mean"]),
        scale=tuple(float(value) for value in payload["scale"]),
        classes=tuple(int(value) for value in payload["classes"]),
        contract=payload["contract"],
        seed=int(payload["seed"]),
        n_fit=int(payload["n_fit"]),
    )


class LearnedScorer(ScoringModel):
    """Rule sub-scores and issues with a learned overall from pair distribution fit."""

    name = "learned_scorer_v1"

    def __init__(
        self,
        bundle: ScorerBundle | None = None,
        *,
        bundle_path: Path | None = None,
        spec: FeatureSpec | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Hold the fitted bundle beside the rule scorer it borrows structure from."""
        self._spec = build_spec() if spec is None else spec
        if bundle is None:
            source = (
                paths.repo_root() / "models" / "scoring" / "logreg.json"
                if bundle_path is None
                else Path(bundle_path)
            )
            bundle = load_bundle(source, self._spec)
        self._bundle = bundle
        self._rules = RuleScorer(config_path)

    @property
    def bundle(self) -> ScorerBundle:
        """Return the fitted bundle this scorer carries."""
        return self._bundle

    def score(self, description: OutfitDescription) -> OutfitScore:
        """Return the rule score with its overall replaced by the learned probability."""
        base = self._rules.score(description)
        features = describe_to_features(description, self._spec)
        probability = float(self._bundle.probability(features)[0])
        scored = base.model_copy(deep=True)
        scored.overall = round(100.0 * probability, 4)
        scored.source_model = self.name
        return scored
