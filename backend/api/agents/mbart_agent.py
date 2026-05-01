"""
mBART inference agent for formality-conditioned EN→KR translation.

Mirrors ClaudeTranslationAgent exactly so app.py can swap backends
without changing any endpoint logic. Returns Korean text only — no
explanation or romanization (the contrast with Claude's richer output
is intentional and is the portfolio demonstration).
"""

import logging
import sys
from pathlib import Path

# app.py runs from backend/api/ — add project root so backend.model is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backend.model.inference import MBartInference
from agents.claude_agent import FormalityResolver
from models.schemas import FormalityToken, SocialContext, TranslationResponse

logger = logging.getLogger(__name__)


class MBartTranslationAgent:
    def __init__(self, model_dir: str):
        self.inference = MBartInference(model_dir)
        self.resolver = FormalityResolver()
        logger.info("mBART translation agent initialized")

    def resolve_formality(self, context: SocialContext) -> FormalityToken:
        return self.resolver.resolve(context)

    async def translate(self, context: SocialContext, text: str) -> TranslationResponse:
        formality_token = self.resolver.resolve(context)
        conditioned_input = f"{formality_token.as_token()} {text}"

        logger.info(
            f"Translating: '{conditioned_input}' "
            f"[relationship={context.relationship.value}, "
            f"age_diff={context.age_differential}, "
            f"setting={context.setting.value}]"
        )

        korean = self.inference.translate(text, formality_token.value)
        logger.info(f"Translation: '{korean}'")

        return TranslationResponse(
            original_text=text,
            conditioned_input=conditioned_input,
            translated_text=korean,
            formality_token=formality_token,
            relationship=context.relationship,
            explanation=None,
            romanization=None,
        )
