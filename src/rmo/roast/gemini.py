"""Gemini-backed roast generator with deterministic fallback and local safety checks."""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any

from pydantic import BaseModel, Field

from rmo.paths import repo_root
from rmo.roast.base import RoastGenerator
from rmo.roast.rules import RuleBasedRoaster
from rmo.roast.safety import flag_text
from rmo.schemas import OutfitDescription, OutfitScore, RoastOutput

__all__ = ["GeminiRoaster"]

log = logging.getLogger(__name__)


class _GeminiRoastResponse(BaseModel):
    """Only the copy fields that Gemini is allowed to generate."""

    roast: str = Field(min_length=1, max_length=1000)
    suggestions: list[Annotated[str, Field(min_length=1, max_length=280)]] = Field(
        min_length=1, max_length=5
    )


class GeminiRoaster(RoastGenerator):
    """Generate roast copy with Gemini and fall back safely on deterministic rules."""

    name = "gemini_roaster"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        self._fallback = RuleBasedRoaster()

        if client is None:
            self._load_dotenv()

        self.model = model or os.environ.get("RMO_GEMINI_MODEL", "gemini-3.6-flash")

        if client is not None:
            self._client = client
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in <project-root>/.env as "
                "GEMINI_API_KEY=your_key, or define it as an environment variable."
            )

        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - installation dependent
            raise RuntimeError(
                "Gemini support requires google-genai. Install project dependencies first."
            ) from exc

        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _load_dotenv() -> None:
        """Load <repo-root>/.env without overriding real environment variables."""
        try:
            from dotenv import load_dotenv
        except ImportError as exc:  # pragma: no cover - installation dependent
            raise RuntimeError(
                "Loading .env files requires python-dotenv. Install project dependencies first."
            ) from exc

        load_dotenv(dotenv_path=repo_root() / ".env", override=False)

    def generate(self, description: OutfitDescription, score: OutfitScore) -> RoastOutput:
        """Generate a Gemini roast, or return the deterministic fallback on failure."""
        fallback = self._fallback.generate(description, score)

        # Do not spend an API request when upstream data is explicitly unscorable.
        if not fallback.grounded_garments and score.overall == 0.0:
            return fallback

        prompt = self._build_prompt(description, score, fallback)

        try:
            interaction = self._client.interactions.create(
                model=self.model,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": _GeminiRoastResponse.model_json_schema(),
                },
            )
            generated = _GeminiRoastResponse.model_validate_json(interaction.output_text)
        except Exception as exc:
            log.warning(
                "Gemini roast generation failed (%s); using rule-based fallback.",
                type(exc).__name__,
            )
            return fallback

        safety_flags = flag_text(" ".join([generated.roast, *generated.suggestions]))
        if safety_flags:
            log.warning(
                "Gemini roast failed local safety checks (%s); using rule-based fallback.",
                ", ".join(safety_flags),
            )
            return fallback

        return RoastOutput(
            image_id=description.image_id,
            roast=generated.roast,
            suggestions=generated.suggestions,
            tone=fallback.tone,
            grounded_garments=fallback.grounded_garments,
            safety_flags=[],
            provenance=description.provenance,
            source_model=self.name,
        )

    def _build_prompt(
    self,
    description: OutfitDescription,
    score: OutfitScore,
    fallback: RoastOutput,
    ) -> str:
        garments = "\n".join(
            f"- {garment.ref}: {garment.color.value} {garment.category}, "
            f"pattern={garment.pattern.value}, fabric={garment.fabric.value}"
            for garment in description.garments
        )

        issues = "\n".join(
            f"- {issue.severity.value} / {issue.code.value}: {issue.message} "
            f"(garments: {', '.join(issue.garment_refs) or 'none'})"
            for issue in score.worst_issues(limit=3)
        ) or "- No issues were reported."

        return f"""
    You are writing the final user-facing response for an app called Roast My Outfit.

    Your job is to roast the OUTFIT, not describe it and not analyse it like a report.

    The response has two clearly different parts:

    1. roast
    2. suggestions

    ROAST STYLE

    Write a short, natural roast that sounds like something a witty friend would actually say.

    The roast should:
    - Be direct and conversational.
    - Usually be one sentence, or at most two short sentences.
    - Focus on the strongest or funniest styling problem.
    - Turn the main problem into one clear joke or observation.
    - Feel spontaneous rather than technical or analytical.
    - Mention clothing pieces only when they make the joke better.
    - Prefer mentioning zero, one, or two garments rather than listing the whole outfit.
    - Never enumerate everything the person is wearing.
    - Never mention internal garment references such as upper_0, lower_0, footwear_0.
    - Never mention issue codes, severity labels, subscores, or the numerical outfit score.
    - Do not simply repeat the scoring explanation.
    - Avoid formulaic phrases such as "the X and the Y are competing" unless they genuinely sound natural.
    - Avoid explaining the joke after making it.

    The roast should NOT sound like:
    "The blouse, skirt, sandals, scarf and bag all use different colours and patterns."

    It should sound more like:
    "This outfit has five different ideas and somehow all of them got approved."

    Or:
    "The patterns are fighting for custody of the outfit."

    Or:
    "The dress code appears to have been decided by committee."

    These examples illustrate the style only. Do not copy them mechanically.

    Requested tone: {fallback.tone.value}

    SUGGESTIONS STYLE

    Suggestions are separate from the roast.

    They should:
    - Be practical and useful rather than funny.
    - Clearly explain how to improve the outfit.
    - Be concise.
    - Usually provide 2 or 3 suggestions.
    - Focus first on the issues that most affect the outfit.
    - Mention specific garments, colours, patterns, fabrics, or accessories when useful.
    - It is completely fine to mention several or all relevant clothing items here.
    - Each suggestion should describe a concrete change the wearer could make.
    - Do not repeat the roast.
    - Do not include jokes or insults in the suggestions.

    Example suggestion style:
    "Keep the floral blouse, but pair it with a solid neutral skirt."
    "Repeat the sandal colour in one accessory to make the palette feel intentional."

    SAFETY

    Roast the clothes and styling only, never the person wearing them.

    Do not comment on:
    - body shape
    - weight
    - age
    - race or ethnicity
    - skin tone
    - gender
    - attractiveness
    - disability
    - health
    - posture
    - any other personal characteristic

    Do not use profanity, slurs, threats, sexual content, or humiliating language aimed at the wearer.

    GROUNDING

    Use only the outfit information below.
    Do not invent garments, colours, fabrics, patterns, accessories, or styling problems.

    Outfit description:
    {description.caption or "(no caption)"}

    Garments:
    {garments}

    Styling issues:
    {issues}

    If no meaningful styling issues are present, make the roast a short playful compliment and make the suggestions positive and minimal.
    """.strip()
