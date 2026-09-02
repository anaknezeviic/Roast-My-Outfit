"""Deterministic roast generator grounded in scored garment issues."""

from __future__ import annotations

from rmo.roast.base import RoastGenerator
from rmo.roast.safety import flag_text
from rmo.schemas import (
    Garment,
    Issue,
    IssueCode,
    IssueSeverity,
    OutfitDescription,
    OutfitScore,
    RoastOutput,
    Tone,
)

__all__ = ["RuleBasedRoaster"]


class RuleBasedRoaster(RoastGenerator):
    """Generate short, grounded roast text from score issues and garment metadata.

    This implementation is intentionally deterministic and dependency-free so Part C can be
    integrated before any external LLM provider is introduced.
    """

    name = "rule_roaster"

    def generate(self, description: OutfitDescription, score: OutfitScore) -> RoastOutput:
        if description.image_id != score.image_id:
            raise ValueError(
                f"Mismatched records: description.image_id={description.image_id!r} "
                f"but score.image_id={score.image_id!r}."
            )

        tone = self._pick_tone(score)
        issues = score.worst_issues(limit=2)
        grounded = self._grounded_refs(issues)

        if self._is_unscorable(description, score):
            roast = "There is not enough reliable detail here to roast the clothes honestly."
            suggestions = [
                "Re-run perception; this description does not carry enough usable garment detail."
            ]
            grounded = []
            tone = Tone.gentle
        elif not score.issues:
            roast = self._compliment(description, score)
            suggestions = ["Keep it exactly as it is."]
            grounded = description.refs()[:5]
            tone = Tone.compliment
        else:
            roast = self._compose_roast(description, tone, issues)
            suggestions = self._compose_suggestions(description, issues)

        safety_flags = flag_text(" ".join([roast, *suggestions]))

        return RoastOutput(
            image_id=description.image_id,
            roast=roast,
            suggestions=suggestions,
            tone=tone,
            grounded_garments=grounded,
            safety_flags=safety_flags,
            provenance=description.provenance,
            source_model=self.name,
        )

    def _is_unscorable(self, description: OutfitDescription, score: OutfitScore) -> bool:
        score_fields = type(score.subscores).model_fields
        if score.overall == 0.0 and all(
            getattr(score.subscores, field) == 0.0 for field in score_fields
        ):
            return True
        if any(issue.code is IssueCode.other for issue in score.issues):
            return True
        return all(garment.confidence == 0.0 for garment in description.garments)

    def _pick_tone(self, score: OutfitScore) -> Tone:
        if not score.issues and score.overall >= 90.0:
            return Tone.compliment

        major_count = sum(issue.severity is IssueSeverity.major for issue in score.issues)
        minor_count = sum(issue.severity is IssueSeverity.minor for issue in score.issues)

        if score.overall < 30.0 or major_count >= 2:
            return Tone.savage
        if major_count >= 1 or minor_count >= 2 or score.overall < 60.0:
            return Tone.playful
        return Tone.gentle

    def _grounded_refs(self, issues: list[Issue]) -> list[str]:
        grounded: list[str] = []
        for issue in issues:
            for ref in issue.garment_refs:
                if ref not in grounded:
                    grounded.append(ref)
        return grounded

    def _compose_roast(
        self,
        description: OutfitDescription,
        tone: Tone,
        issues: list[Issue],
    ) -> str:
        lead = issues[0]
        primary = self._primary_issue_line(description, lead, tone)
        if len(issues) == 1:
            return primary

        followup = self._secondary_issue_line(description, issues[1])
        return f"{primary} {followup}"

    def _primary_issue_line(self, description: OutfitDescription, issue: Issue, tone: Tone) -> str:
        subject = self._garment_subject(description, issue.garment_refs)
        templates: dict[IssueCode, dict[Tone, str]] = {
            IssueCode.hue_clash: {
                Tone.gentle: f"The colour conversation between {subject} never really settles.",
                Tone.playful: f"{subject.capitalize()} are arguing across the colour wheel.",
                Tone.savage: f"{subject.capitalize()} look like they were assigned by random colour picker.",
                Tone.compliment: f"{subject.capitalize()} sit together cleanly.",
            },
            IssueCode.too_many_colors: {
                Tone.gentle: "There are more colour ideas here than the outfit can comfortably carry.",
                Tone.playful: "This palette brought every opinion to the meeting.",
                Tone.savage: "You did not get dressed; you got sorted into every colour idea at once.",
                Tone.compliment: "The colour count stays disciplined.",
            },
            IssueCode.low_contrast: {
                Tone.gentle: f"{subject.capitalize()} sit so close in value that the outfit blurs into itself.",
                Tone.playful: f"{subject.capitalize()} are dressed on the same brightness setting.",
                Tone.savage: f"{subject.capitalize()} merge into one long beige paragraph.",
                Tone.compliment: f"{subject.capitalize()} separate cleanly.",
            },
            IssueCode.monochrome_flat: {
                Tone.gentle: "The monochrome idea is sound, but it needs one sharper focal point.",
                Tone.playful: "Monochrome can be chic; this one forgot to invite contrast.",
                Tone.savage: "This outfit found one note and played it for the entire concert.",
                Tone.compliment: "The monochrome palette stays lively.",
            },
            IssueCode.pattern_clash: {
                Tone.gentle: "The patterns are competing instead of taking turns.",
                Tone.playful: "These patterns are all speaking at once.",
                Tone.savage: "Three patterns walked in and none of them agreed to share the stage.",
                Tone.compliment: "The pattern mix stays controlled.",
            },
            IssueCode.formality_mismatch: {
                Tone.gentle: f"{subject.capitalize()} land in different dress codes.",
                Tone.playful: "Half the outfit booked dinner, the other half showed up for errands.",
                Tone.savage: "This outfit is formal in one tab and casual in another.",
                Tone.compliment: "Everything stays in the same register.",
            },
            IssueCode.season_mismatch: {
                Tone.gentle: f"{subject.capitalize()} feel built for different weather.",
                Tone.playful: "One piece says winter and another says the sun is out.",
                Tone.savage: "The forecast clearly was not invited into this decision.",
                Tone.compliment: "The seasonal read is consistent.",
            },
            IssueCode.fabric_mismatch: {
                Tone.gentle: f"{subject.capitalize()} pull in different texture directions.",
                Tone.playful: "One fabric wants a dinner reservation, the other wants utility pockets.",
                Tone.savage: "These fabrics met five minutes ago and it shows.",
                Tone.compliment: "The texture mix is coherent.",
            },
            IssueCode.proportion_imbalance: {
                Tone.gentle: "The silhouette could use a clearer line.",
                Tone.playful: "The proportions are having a hard time finding the waist and hem.",
                Tone.savage: "The silhouette starts a sentence and never finds the full stop.",
                Tone.compliment: "The proportions stay balanced.",
            },
            IssueCode.missing_footwear: {
                Tone.gentle: "The outfit reaches the ankle and then stops negotiating.",
                Tone.playful: "Strong look, unfinished ending.",
                Tone.savage: "The outfit forgot the final sentence: shoes.",
                Tone.compliment: "The finishing pieces are all present.",
            },
            IssueCode.accessory_overload: {
                Tone.gentle: "The accessories are doing a little too much at the same time.",
                Tone.playful: "Every accessory wanted a speaking role.",
                Tone.savage: "This is less styling and more a traffic jam of add-ons.",
                Tone.compliment: "The accessories stay disciplined.",
            },
            IssueCode.other: {
                Tone.gentle: "There is not enough reliable detail here to judge the outfit fairly.",
                Tone.playful: "The outfit brief arrived, but the evidence did not.",
                Tone.savage: "There is not enough here to be savage about with a straight face.",
                Tone.compliment: "The record is complete.",
            },
        }
        return templates.get(issue.code, templates[IssueCode.other])[tone]

    def _secondary_issue_line(self, description: OutfitDescription, issue: Issue) -> str:
        subject = self._garment_subject(description, issue.garment_refs)
        followups: dict[IssueCode, str] = {
            IssueCode.hue_clash: f"{subject.capitalize()} need one colour to lead and the rest to support.",
            IssueCode.too_many_colors: "Editing the palette down would make the whole look feel intentional.",
            IssueCode.low_contrast: "One lighter or darker anchor would give the eye somewhere to land.",
            IssueCode.monochrome_flat: "A stronger light-dark break would wake it up.",
            IssueCode.pattern_clash: "Keeping one pattern as the feature would calm the frame down.",
            IssueCode.formality_mismatch: "Matching the shoes and outer layer to the same register would fix a lot.",
            IssueCode.season_mismatch: "Choose one weather story and let every piece support it.",
            IssueCode.fabric_mismatch: "Bringing the textures closer together would make the look feel deliberate.",
            IssueCode.proportion_imbalance: "A cleaner hem relationship would help the outfit breathe.",
            IssueCode.missing_footwear: "Adding the right pair of shoes would complete the sentence.",
            IssueCode.accessory_overload: "Removing just one or two pieces would restore the hierarchy.",
            IssueCode.other: "Re-running the upstream stages would help more than guessing.",
        }
        return followups.get(issue.code, followups[IssueCode.other])

    def _compose_suggestions(self, description: OutfitDescription, issues: list[Issue]) -> list[str]:
        suggestions: list[str] = []
        for issue in issues:
            for suggestion in self._issue_suggestions(description, issue):
                if suggestion not in suggestions:
                    suggestions.append(suggestion)
                if len(suggestions) == 5:
                    return suggestions
        return suggestions or ["Simplify the outfit by letting one idea lead and the rest support it."]

    def _issue_suggestions(self, description: OutfitDescription, issue: Issue) -> list[str]:
        labels = [self._garment_label(description, ref) for ref in issue.garment_refs]
        first = labels[0] if labels else "main piece"
        second = labels[1] if len(labels) > 1 else "supporting piece"
        mapping: dict[IssueCode, list[str]] = {
            IssueCode.hue_clash: [
                f"Let the {first} stay loud and move the {second} to a neutral.",
                "Repeat one accent colour once and let the rest quiet down.",
            ],
            IssueCode.too_many_colors: [
                "Pick one accent and take the rest of the palette neutral.",
                "Make one accessory repeat a clothing colour so the palette looks intentional.",
            ],
            IssueCode.low_contrast: [
                f"Create more separation with a lighter or darker {second}.",
                "Use a shoe or belt with clearer contrast to break the block.",
            ],
            IssueCode.monochrome_flat: [
                "Add one lighter or darker piece to create a focal point.",
                "Use texture or a contrasting shoe to stop the outfit reading as one block.",
            ],
            IssueCode.pattern_clash: [
                f"Keep the {first} as the feature and make the other patterned pieces plain.",
                "Limit the outfit to one dominant pattern.",
            ],
            IssueCode.formality_mismatch: [
                "Either dress the footwear up or relax the tailoring so both speak the same language.",
                "Keep every major piece in the same formality band.",
            ],
            IssueCode.season_mismatch: [
                "Choose one season and let every piece support that weight and coverage.",
                f"Swap the most out-of-season piece, starting with the {first}.",
            ],
            IssueCode.fabric_mismatch: [
                "Pair delicate fabrics with cleaner tailored pieces or group the utilitarian fabrics together.",
                f"Starting with the {second}, move one texture closer to the rest of the outfit.",
            ],
            IssueCode.proportion_imbalance: [
                "Balance the silhouette by clarifying the hem lengths and waist line.",
                "Use either a shorter outer layer or a longer bottom to restore the line.",
            ],
            IssueCode.missing_footwear: [
                "Add footwear that matches the outfit's overall formality.",
            ],
            IssueCode.accessory_overload: [
                "Remove one or two accessories so the clothes have room to lead.",
                "Keep one hero accessory and let the rest support it.",
            ],
            IssueCode.other: [
                "Re-run perception; this description does not carry enough usable garment detail.",
            ],
        }
        return mapping.get(issue.code, mapping[IssueCode.other])

    def _compliment(self, description: OutfitDescription, score: OutfitScore) -> str:
        garments = [self._describe_garment(garment) for garment in description.garments[:3]]
        if garments:
            joined = ", ".join(garments[:-1]) + (
                " and " + garments[-1] if len(garments) > 1 else garments[0]
            )
            return (
                f"{joined.capitalize()} work together cleanly. "
                "The outfit looks considered without trying too hard."
            )
        return (
            f"This outfit is coherent, balanced and confidently put together. "
            f"Overall score: {score.overall:.0f}."
        )

    def _garment_subject(self, description: OutfitDescription, refs: list[str]) -> str:
        labels = [self._garment_label(description, ref) for ref in refs]
        labels = [label for label in labels if label]
        if not labels:
            return "these pieces"
        if len(labels) == 1:
            return f"the {labels[0]}"
        if len(labels) == 2:
            return f"the {labels[0]} and the {labels[1]}"
        return ", ".join(f"the {label}" for label in labels[:-1]) + f", and the {labels[-1]}"

    def _garment_label(self, description: OutfitDescription, ref: str) -> str:
        garment = self._garment_by_ref(description, ref)
        if garment is None:
            return ref
        return garment.category

    def _garment_by_ref(self, description: OutfitDescription, ref: str) -> Garment | None:
        for garment in description.garments:
            if garment.ref == ref:
                return garment
        return None

    def _describe_garment(self, garment: Garment) -> str:
        color = garment.color.value.replace("_", " ")
        if color == "unknown":
            return garment.category
        return f"{color} {garment.category}"