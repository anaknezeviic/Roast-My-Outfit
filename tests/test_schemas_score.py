"""Cover the outfit score contract and the scoring interface."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from rmo.schemas import (
    _SEVERITY_RANK,
    Issue,
    IssueCode,
    IssueSeverity,
    OutfitDescription,
    OutfitScore,
    Provenance,
    SubScores,
)
from rmo.scoring.base import ScoringModel


def make_subscores(**overrides: Any) -> SubScores:
    payload: dict[str, Any] = {
        "color_harmony": 70.0,
        "formality_consistency": 60.0,
        "seasonality": 80.0,
        "proportion": 50.0,
    }
    payload.update(overrides)
    return SubScores.model_validate(payload)


def make_issue(**overrides: Any) -> Issue:
    payload: dict[str, Any] = {
        "code": IssueCode.hue_clash,
        "message": "The upper and lower hues sit close to opposite each other.",
    }
    payload.update(overrides)
    return Issue.model_validate(payload)


def make_score(**overrides: Any) -> OutfitScore:
    payload: dict[str, Any] = {
        "image_id": "fixture_000",
        "overall": 62.5,
        "subscores": make_subscores(),
        "provenance": Provenance.fixture,
        "source_model": "fixture",
    }
    payload.update(overrides)
    return OutfitScore.model_validate(payload)


def test_overall_is_bounded_to_a_hundred_points() -> None:
    assert make_score(overall=0).overall == 0
    assert make_score(overall=100).overall == 100
    with pytest.raises(ValidationError):
        make_score(overall=101)
    with pytest.raises(ValidationError):
        make_score(overall=-1)


def test_subscores_are_bounded_to_a_hundred_points() -> None:
    with pytest.raises(ValidationError):
        make_subscores(color_harmony=100.1)
    with pytest.raises(ValidationError):
        make_subscores(proportion=-0.1)


def test_scores_reject_booleans() -> None:
    with pytest.raises(ValidationError):
        make_score(overall=True)
    with pytest.raises(ValidationError):
        make_subscores(seasonality=False)


def test_scores_reject_numeric_strings() -> None:
    with pytest.raises(ValidationError):
        make_score(overall="62.5")


def test_whole_number_scores_are_accepted() -> None:
    assert make_score(overall=62).overall == 62.0
    assert OutfitScore.model_validate_json(make_score().model_dump_json()).overall == 62.5


def test_subscores_are_all_required() -> None:
    with pytest.raises(ValidationError):
        SubScores(color_harmony=70.0, formality_consistency=60.0, seasonality=80.0)


def test_provenance_must_be_stated() -> None:
    with pytest.raises(ValidationError):
        OutfitScore(
            image_id="fixture_000",
            overall=62.5,
            subscores=make_subscores(),
            source_model="fixture",
        )


def test_unknown_field_on_issue_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_issue(weight=3)


def test_unknown_field_on_score_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_score(verdict="tragic")


def test_issue_message_length_limits() -> None:
    assert len(make_issue(message="a" * 280).message) == 280
    with pytest.raises(ValidationError):
        make_issue(message="a" * 281)
    with pytest.raises(ValidationError):
        make_issue(message="")


def test_issue_defaults() -> None:
    issue = make_issue()
    assert issue.severity is IssueSeverity.minor
    assert issue.garment_refs == []


def test_blank_garment_ref_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_issue(garment_refs=[""])
    with pytest.raises(ValidationError):
        make_issue(garment_refs=["upper_0", "   "])


def test_worst_issues_puts_major_first() -> None:
    score = make_score(
        issues=[
            make_issue(severity=IssueSeverity.info),
            make_issue(severity=IssueSeverity.minor),
            make_issue(severity=IssueSeverity.major),
        ]
    )
    result = score.worst_issues()
    assert result[0].severity is IssueSeverity.major
    assert result[1].severity is IssueSeverity.minor
    assert result[2].severity is IssueSeverity.info


def test_equal_severity_orders_by_code_value() -> None:
    score = make_score(
        issues=[
            make_issue(code=IssueCode.hue_clash),
            make_issue(code=IssueCode.formality_mismatch),
        ]
    )
    assert [issue.code for issue in score.worst_issues()] == [
        IssueCode.formality_mismatch,
        IssueCode.hue_clash,
    ]


def test_equal_severity_and_code_order_by_garment_refs() -> None:
    score = make_score(
        issues=[
            make_issue(garment_refs=["upper_0"]),
            make_issue(garment_refs=["lower_0"]),
            make_issue(garment_refs=[]),
        ]
    )
    assert [issue.garment_refs for issue in score.worst_issues()] == [
        [],
        ["lower_0"],
        ["upper_0"],
    ]


def test_worst_issues_honours_the_limit() -> None:
    score = make_score(issues=[make_issue(code=code) for code in IssueCode])
    assert len(score.worst_issues(limit=2)) == 2
    assert len(score.worst_issues()) == 3


def test_worst_issues_on_no_issues_is_empty() -> None:
    assert make_score().worst_issues() == []


def test_worst_issues_is_repeatable() -> None:
    score = make_score(
        issues=[
            make_issue(code=IssueCode.pattern_clash, severity=IssueSeverity.major),
            make_issue(code=IssueCode.low_contrast, severity=IssueSeverity.major),
            make_issue(code=IssueCode.low_contrast, severity=IssueSeverity.minor),
        ]
    )
    assert score.worst_issues() == score.worst_issues()


def test_worst_issues_does_not_reorder_the_stored_issues() -> None:
    score = make_score(
        issues=[
            make_issue(severity=IssueSeverity.info),
            make_issue(severity=IssueSeverity.major),
        ]
    )
    score.worst_issues()
    assert [issue.severity for issue in score.issues] == [
        IssueSeverity.info,
        IssueSeverity.major,
    ]


def test_worst_issues_hands_out_copies() -> None:
    score = make_score(issues=[make_issue(garment_refs=["upper_0"])])
    taken = score.worst_issues()[0]
    taken.message = "rewritten downstream"
    taken.garment_refs.append("lower_0")
    assert score.issues[0].message != "rewritten downstream"
    assert score.issues[0].garment_refs == ["upper_0"]


def test_severity_rank_covers_every_member() -> None:
    assert set(_SEVERITY_RANK) == set(IssueSeverity)
    assert sorted(_SEVERITY_RANK.values()) == list(range(len(IssueSeverity)))


def test_alphabetical_sort_differs_from_severity_rank() -> None:
    severities = [IssueSeverity.info, IssueSeverity.minor, IssueSeverity.major]
    alphabetical = sorted(severities)
    by_rank = sorted(severities, key=lambda severity: _SEVERITY_RANK[severity])
    assert alphabetical != by_rank
    assert alphabetical[0] is IssueSeverity.info
    assert by_rank[0] is IssueSeverity.major


def test_issue_code_has_an_escape_hatch() -> None:
    assert IssueCode.other.value == "other"


def test_unlisted_issue_code_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_issue(code="bad_vibes")


class ConstantScoring(ScoringModel):
    name = "constant"

    def score(self, description: OutfitDescription) -> OutfitScore:
        return make_score(image_id=description.image_id)


def test_scoring_model_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ScoringModel()


def test_subclass_without_score_cannot_be_instantiated() -> None:
    class Incomplete(ScoringModel):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_scoring_model_scores_the_description_it_is_given() -> None:
    description = OutfitDescription.model_validate(
        {
            "image_id": "fixture_001",
            "garments": [{"slot": "upper", "category": "graphic tee"}],
            "source_model": "fixture",
        }
    )
    assert ConstantScoring().score(description).image_id == "fixture_001"
