"""Gemini-backed roast generator with deterministic fallback and local safety checks."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field

from rmo.config import load_roast_config
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
        config_path: Path | None = None,
    ) -> None:
        self._fallback = RuleBasedRoaster()
        self._config = load_roast_config(config_path)
        self._client: Any | None

        if client is None:
            self._load_dotenv()

        self.model = model or os.environ.get("RMO_GEMINI_MODEL", "gemini-3.6-flash")

        if client is not None:
            self._client = client
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            log.warning(
                "GEMINI_API_KEY is not set; roasting with the rule-based generator instead."
            )
            self._client = None
            return

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

        if self._client is None:
            return fallback

        if not fallback.grounded_garments and score.overall == 0.0:
            return fallback

        prompt = self._build_prompt(description, score, fallback)

        log.debug("gemini request: model=%s input=%s", self.model, prompt)

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
            output_text = interaction.output_text
            log.debug("gemini response: model=%s output=%s", self.model, output_text)
            log.info(
                "gemini roast for %s: model=%s prompt_chars=%d response_chars=%d",
                description.image_id,
                self.model,
                len(prompt),
                len(output_text or ""),
            )
            generated = _GeminiRoastResponse.model_validate_json(output_text)
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
            rejected = fallback.model_copy(deep=True)
            rejected.safety_flags = safety_flags
            return rejected

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

        body = self._config["prompt_template"].format(
            tone=fallback.tone.value,
            caption=description.caption or "(no caption)",
            garments=garments,
            issues=issues,
        )
        return f"{self._config['persona']}\n\n{body}".strip()
