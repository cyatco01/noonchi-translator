"""
Claude AI Agent for English-to-Korean translation with formality conditioning.

Acts as the prototype translation backend (comparison baseline) before the
fine-tuned mBART model is available. Mirrors the same interface the mBART
inference endpoint will expose: receives a conditioned input string
("<token> english text") and returns Korean output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from anthropic import Anthropic

# app.py runs from backend/api/ — insert project root so backend.formality is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from backend.formality.resolver import FormalityResolver  # noqa: E402

from config import get_settings
from models.schemas import (
    RelationshipType,
    SettingType,
    FormalityToken,
    SocialContext,
    TranslationResponse
)
from rag.retriever import SociolinguisticRetriever

logger = logging.getLogger(__name__)


def parse_situation(client: Anthropic, situation: str) -> tuple[SocialContext, str, float]:
    """
    Parse a free-text situation description into a structured SocialContext.

    Uses tool_use with a forced tool call so the response is schema-validated
    by the API — no JSON parsing needed.

    Returns:
        (SocialContext, reasoning, confidence)
    """
    response = client.messages.create(
        model=get_settings().CONTEXT_PARSE_MODEL,
        max_tokens=200,
        tools=[{
            "name": "extract_social_context",
            "description": "Extract social context from a situation description for Korean formality inference.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "relationship": {
                        "type": "string",
                        "enum": ["boss", "elder", "professor", "colleague", "peer",
                                 "subordinate", "friend", "acquaintance", "stranger"]
                    },
                    "age_differential": {
                        "type": "integer",
                        "minimum": -50,
                        "maximum": 50,
                        "description": "Negative = speaker is younger. Positive = speaker is older. 0 = similar age or unknown."
                    },
                    "setting": {
                        "type": "string",
                        "enum": ["workplace", "academic", "social", "public", "intimate"]
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence explaining what social context was detected."
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence in the extraction (0.0–1.0)."
                    }
                },
                "required": ["relationship", "age_differential", "setting", "reasoning", "confidence"]
            }
        }],
        tool_choice={"type": "tool", "name": "extract_social_context"},
        messages=[{"role": "user", "content": f"Extract social context: {situation}"}]
    )

    data = next(b for b in response.content if b.type == "tool_use").input
    context = SocialContext(
        relationship=RelationshipType(data["relationship"]),
        age_differential=int(data["age_differential"]),
        setting=SettingType(data["setting"]),
    )
    return context, data["reasoning"], float(data["confidence"])


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
        self.retriever = SociolinguisticRetriever()
        logger.info("Claude translation agent initialized")

    def resolve_formality(self, context: SocialContext) -> FormalityToken:
        """Delegate to FormalityResolver."""
        return self.resolver.resolve(context)

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

        guide = self.FORMALITY_GUIDE[formality_token]

        retrieved = self.retriever.retrieve(text, formality_token.value)
        if retrieved:
            rag_block = "\n\nRelevant sociolinguistic notes:\n" + "\n".join(
                f"- {note}" for note in retrieved
            )
        else:
            rag_block = ""

        prompt = (
            f"You are a Korean translator. Translate the English text to Korean "
            f"using the specified speech level.\n\n"
            f"Input: {conditioned_input}\n\n"
            f"Target speech level: {guide['level']}\n"
            f"Required sentence-final endings: {guide['endings']}\n"
            f"Notes: {guide['notes']}"
            f"{rag_block}"
        )

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.settings.MAX_TOKENS,
            temperature=self.settings.TEMPERATURE,
            tools=[{
                "name": "provide_translation",
                "description": "Provide the Korean translation with optional romanization and explanation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "translated_text": {
                            "type": "string",
                            "description": "Korean translation at the specified formality level."
                        },
                        "romanization": {
                            "type": "string",
                            "description": "Romanized pronunciation (optional)."
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief note on formality choices made (optional)."
                        }
                    },
                    "required": ["translated_text"]
                }
            }],
            tool_choice={"type": "tool", "name": "provide_translation"},
            messages=[{"role": "user", "content": prompt}]
        )

        result = next(b for b in message.content if b.type == "tool_use").input
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
