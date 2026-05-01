"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class RelationshipType(str, Enum):
    """Speaker-addressee relationship types."""
    BOSS = "boss"
    ELDER = "elder"
    PROFESSOR = "professor"
    COLLEAGUE = "colleague"
    PEER = "peer"
    SUBORDINATE = "subordinate"
    FRIEND = "friend"
    ACQUAINTANCE = "acquaintance"
    STRANGER = "stranger"


class SettingType(str, Enum):
    """Situational setting of the interaction."""
    WORKPLACE = "workplace"
    ACADEMIC = "academic"
    SOCIAL = "social"
    PUBLIC = "public"
    INTIMATE = "intimate"


class FormalityToken(str, Enum):
    """
    Three operative formality tokens used for conditioning.
    These are prepended to the source sentence before passing to mBART.

    <formal>  → 합쇼체 / 하십시오체  (-습니다 / -ㅂ니다)
    <polite>  → 해요체               (-아요 / -어요)
    <casual>  → 해라체 / 해체        (-아 / -어 / -냐)
    """
    FORMAL = "formal"
    POLITE = "polite"
    CASUAL = "casual"

    def as_token(self) -> str:
        """Return the conditioning token string used in model input."""
        return f"<{self.value}>"


class FormalityLevel(str, Enum):
    """
    All 7 Korean speech levels (경어법) — reference enum.
    The system consolidates these into three FormalityToken tiers for training/inference.
    """
    HASIPSIOCHE = "hasipsioche"   # 하십시오체 — formal polite (→ <formal>)
    HAOCHE = "haoche"             # 하오체 — formal neutral, archaic
    HAGECHE = "hageche"           # 하게체 — semi-formal, archaic
    HAERACHE = "haerache"         # 해라체 — plain/casual (→ <casual>)
    HAEYOCHE = "haeyoche"         # 해요체 — informal polite (→ <polite>)
    HAECHE = "haeche"             # 해체 / banmal — casual (→ <casual>)
    HONORIFIC = "honorific"       # 존댓말 — honorific vocabulary layer


class SocialContext(BaseModel):
    """
    Structured social context used by FormalityResolver to infer formality.
    Mirrors the doc spec exactly.
    """
    relationship: RelationshipType = Field(
        ...,
        description="Speaker-addressee relationship",
        examples=["boss"]
    )
    age_differential: int = Field(
        ...,
        ge=-50,
        le=50,
        description="Age difference: negative = speaker is younger, positive = speaker is older",
        examples=[-10, 0, 10]
    )
    setting: SettingType = Field(
        ...,
        description="Situational setting of the interaction",
        examples=["workplace"]
    )
    formality_override: Optional[FormalityToken] = Field(
        None,
        description="Optional: manually override the inferred formality level"
    )


class ContextRequest(BaseModel):
    """Request body for Step 1: set conversation context via free-text description."""
    situation: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Describe who you're talking to and the context",
        examples=["Emailing my professor about a missed deadline"]
    )
    formality_override: Optional[FormalityToken] = Field(
        None,
        description="Optional: manually override the inferred formality level"
    )


class ContextResponse(BaseModel):
    """Response after setting context (Step 1)."""
    situation: str
    session_id: str
    relationship: RelationshipType
    age_differential: int
    setting: SettingType
    formality_token: FormalityToken
    conditioning_input_prefix: str = Field(
        ...,
        description="The token that will be prepended to source text, e.g. '<formal>'"
    )
    message: str


class TranslationRequest(BaseModel):
    """Request body for Step 2: translate text."""
    session_id: str = Field(
        ...,
        description="Session ID from /api/set-context",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="English text to translate",
        examples=["Do you want to eat?"]
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class TranslationResponse(BaseModel):
    """Response body for translation endpoint."""
    original_text: str
    conditioned_input: str = Field(
        ...,
        description="The actual model input: '<token> english text'",
        examples=["<formal> Do you want to eat?"]
    )
    translated_text: str
    formality_token: FormalityToken
    relationship: RelationshipType
    explanation: Optional[str] = None
    romanization: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
