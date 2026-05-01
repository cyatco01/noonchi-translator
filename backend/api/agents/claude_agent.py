"""
Claude AI Agent for English-to-Korean translation with formality conditioning.

Acts as the prototype translation backend (comparison baseline) before the
fine-tuned mBART model is available. Mirrors the same interface the mBART
inference endpoint will expose: receives a conditioned input string
("<token> english text") and returns Korean output.
"""

import json
import logging
from typing import Optional

from anthropic import Anthropic

from config import get_settings
from models.schemas import (
    RelationshipType,
    SettingType,
    FormalityToken,
    SocialContext,
    TranslationResponse
)

logger = logging.getLogger(__name__)


def parse_situation(client: Anthropic, situation: str) -> tuple[SocialContext, str]:
    """
    Parse a free-text situation description into a structured SocialContext.

    Uses claude-haiku for speed and cost efficiency — this is a pure
    classification task that doesn't need a large model.

    Returns:
        (SocialContext, explanation) where explanation is a one-sentence
        human-readable summary of what was detected.
    """
    prompt = f"""You are analyzing social context for a Korean translation app.

Given a situation description, extract the social context needed to determine
the appropriate Korean speech level.

Valid relationship values: boss, elder, professor, colleague, peer, subordinate,
friend, acquaintance, stranger

Valid setting values: workplace, academic, social, public, intimate

Age differential: integer from -50 to 50.
  Negative = speaker is younger than the other person.
  Positive = speaker is older than the other person.
  0 = similar age or unknown.

Situation: "{situation}"

Respond with JSON only:
{{
    "relationship": "...",
    "age_differential": 0,
    "setting": "...",
    "explanation": "one short sentence describing what you detected"
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()

    data = json.loads(raw)
    context = SocialContext(
        relationship=RelationshipType(data["relationship"]),
        age_differential=int(data["age_differential"]),
        setting=SettingType(data["setting"]),
    )
    return context, data["explanation"]

logger = logging.getLogger(__name__)


class FormalityResolver:
    """
    Rule-based pragmatic inference engine.

    Maps structured social context (relationship, age_differential, setting)
    to one of three operative formality tokens: <formal>, <polite>, <casual>.

    Rules encode Korean sociolinguistic norms:
    - Workplace superiors, professors, elders, public settings → <formal>
    - Close friends, similar age, intimate/social settings → <casual>
    - Default (acquaintances, colleagues, neutral settings) → <polite>
    """

    def resolve(self, context: SocialContext) -> FormalityToken:
        """
        Resolve social context to a formality token.

        Args:
            context: Structured social context from the user

        Returns:
            FormalityToken: formal, polite, or casual
        """
        if context.formality_override is not None:
            return context.formality_override

        rel = context.relationship
        age_diff = context.age_differential
        setting = context.setting

        # --- Formal triggers ---
        # Superior relationships always warrant formal register
        if rel in (RelationshipType.BOSS, RelationshipType.PROFESSOR, RelationshipType.ELDER):
            return FormalityToken.FORMAL

        # Stranger in any setting, or any relationship in public → formal
        if rel == RelationshipType.STRANGER or setting == SettingType.PUBLIC:
            return FormalityToken.FORMAL

        # Speaker is notably younger (age_diff < -5) in formal/professional setting
        if age_diff < -5 and setting in (SettingType.WORKPLACE, SettingType.ACADEMIC):
            return FormalityToken.FORMAL

        # --- Casual triggers ---
        # Friend + similar age + informal setting → casual
        if rel == RelationshipType.FRIEND and age_diff >= -3 and setting in (
            SettingType.INTIMATE, SettingType.SOCIAL
        ):
            return FormalityToken.CASUAL

        # Subordinate in intimate/social setting, speaker notably older → casual
        if rel == RelationshipType.SUBORDINATE and age_diff > 5 and setting in (
            SettingType.INTIMATE, SettingType.SOCIAL
        ):
            return FormalityToken.CASUAL

        # --- Default ---
        return FormalityToken.POLITE


class ClaudeTranslationAgent:
    """
    Prototype translation backend using Claude API.

    Receives a conditioned input string ("<token> english text") and returns
    Korean output at the specified formality level. This mirrors the interface
    that the fine-tuned mBART model will expose, making it easy to swap in
    the real model once trained.
    """

    # Maps formality token to Korean speech level description for the prompt
    FORMALITY_GUIDE = {
        FormalityToken.FORMAL: {
            "level": "합쇼체 / 하십시오체 (formal polite)",
            "endings": "-습니다, -ㅂ니다, -습니까",
            "notes": "Use honorific vocabulary where applicable (드시다 instead of 먹다, etc.)"
        },
        FormalityToken.POLITE: {
            "level": "해요체 (informal polite)",
            "endings": "-아요, -어요, -여요",
            "notes": "Neutral, everyday polite register. Most common in modern Korean."
        },
        FormalityToken.CASUAL: {
            "level": "해라체 / 해체 (casual / banmal)",
            "endings": "-아, -어, -냐, -지, -구나",
            "notes": "Drop honorifics. Use plain verb stems with casual endings."
        }
    }

    def __init__(self):
        self.settings = get_settings()
        self.client = Anthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        self.model = self.settings.CLAUDE_MODEL
        self.resolver = FormalityResolver()
        logger.info("Claude translation agent initialized")

    def resolve_formality(self, context: SocialContext) -> FormalityToken:
        """Delegate to FormalityResolver."""
        return self.resolver.resolve(context)

    def _build_prompt(self, conditioned_input: str, formality_token: FormalityToken) -> str:
        guide = self.FORMALITY_GUIDE[formality_token]
        token_str = formality_token.as_token()

        return f"""You are a Korean translator. Translate the English text to Korean using the specified speech level.

The input follows the conditioning format used in formality-conditioned mBART fine-tuning:
  {token_str} <english text>

Input: {conditioned_input}

Target speech level: {guide['level']}
Required sentence-final endings: {guide['endings']}
Notes: {guide['notes']}

Respond with valid JSON only:
{{
    "translated_text": "Korean translation here",
    "romanization": "romanized pronunciation (optional)",
    "explanation": "brief note on formality choices made"
}}"""

    async def translate(self, context: SocialContext, text: str) -> TranslationResponse:
        """
        Translate English text conditioned on the resolved formality token.

        Args:
            context: Full social context (relationship, age_differential, setting)
            text: English source text

        Returns:
            TranslationResponse with Korean output and metadata
        """
        formality_token = self.resolver.resolve(context)
        conditioned_input = f"{formality_token.as_token()} {text}"

        logger.info(
            f"Translating: '{conditioned_input}' "
            f"[relationship={context.relationship.value}, "
            f"age_diff={context.age_differential}, "
            f"setting={context.setting.value}]"
        )

        prompt = self._build_prompt(conditioned_input, formality_token)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.settings.MAX_TOKENS,
            temperature=self.settings.TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                result = json.loads(response_text[start:end].strip())
            else:
                raise ValueError(f"Invalid JSON from Claude: {response_text}")

        logger.info(f"Translation: '{result['translated_text']}'")

        return TranslationResponse(
            original_text=text,
            conditioned_input=conditioned_input,
            translated_text=result["translated_text"],
            formality_token=formality_token,
            relationship=context.relationship,
            explanation=result.get("explanation"),
            romanization=result.get("romanization")
        )
