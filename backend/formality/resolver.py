"""
FormalityResolver — rule-based pragmatic inference engine.

Maps structured social context (relationship, age_differential, setting)
to one of three formality tokens used for mBART conditioning.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RelationshipType(str, Enum):
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
    WORKPLACE = "workplace"
    ACADEMIC = "academic"
    SOCIAL = "social"
    PUBLIC = "public"
    INTIMATE = "intimate"


class FormalityToken(str, Enum):
    """
    Three operative formality tokens for mBART conditioning.

    <formal>  → 합쇼체 / 하십시오체  (-습니다 / -ㅂ니다)
    <polite>  → 해요체               (-아요 / -어요)
    <casual>  → 해라체 / 해체        (-아 / -어 / -냐)
    """
    FORMAL = "formal"
    POLITE = "polite"
    CASUAL = "casual"

    def as_token(self) -> str:
        return f"<{self.value}>"


@dataclass
class SocialContext:
    relationship: RelationshipType
    age_differential: int           # negative = speaker is younger, positive = speaker is older
    setting: SettingType
    formality_override: Optional[FormalityToken] = None


class FormalityResolver:
    """
    Core sociolinguistic inference component.

    Encodes Korean pragmatic norms as rules over (relationship, age_differential, setting):
      - Workplace superiors, professors, elders, public settings → <formal>
      - Close friends, similar age, intimate/social settings    → <casual>
      - Default (acquaintances, colleagues, neutral)            → <polite>
    """

    def resolve(self, context: SocialContext) -> FormalityToken:
        if context.formality_override is not None:
            return context.formality_override

        rel = context.relationship
        age_diff = context.age_differential
        setting = context.setting

        # --- Formal triggers ---
        if rel in (RelationshipType.BOSS, RelationshipType.PROFESSOR, RelationshipType.ELDER):
            return FormalityToken.FORMAL

        if rel == RelationshipType.STRANGER or setting == SettingType.PUBLIC:
            return FormalityToken.FORMAL

        if age_diff < -5 and setting in (SettingType.WORKPLACE, SettingType.ACADEMIC):
            return FormalityToken.FORMAL

        # --- Casual triggers ---
        if rel == RelationshipType.FRIEND and age_diff >= -3 and setting in (
            SettingType.INTIMATE, SettingType.SOCIAL
        ):
            return FormalityToken.CASUAL

        if rel == RelationshipType.SUBORDINATE and age_diff > 5 and setting in (
            SettingType.INTIMATE, SettingType.SOCIAL
        ):
            return FormalityToken.CASUAL

        return FormalityToken.POLITE
