"""
Session management for storing conversation context between API calls.

Allows users to set social context once (Step 1), then translate
multiple times with that context (Step 2).
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass
import logging

from models.schemas import RelationshipType, SettingType, FormalityToken

logger = logging.getLogger(__name__)


@dataclass
class ConversationSession:
    """Stores the full social context for a user session."""
    situation: str
    session_id: str
    relationship: RelationshipType
    age_differential: int
    setting: SettingType
    formality_token: FormalityToken
    formality_override: Optional[FormalityToken]
    created_at: datetime
    last_used: datetime


class SessionManager:
    """
    In-memory session store. Suitable for prototype/development.
    Production would use Redis or a database.
    """

    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, ConversationSession] = {}
        self.timeout = timedelta(minutes=session_timeout_minutes)
        logger.info(f"Session manager initialized (timeout: {session_timeout_minutes}m)")

    def create_session(
        self,
        situation: str,
        relationship: RelationshipType,
        age_differential: int,
        setting: SettingType,
        formality_token: FormalityToken,
        formality_override: Optional[FormalityToken] = None
    ) -> ConversationSession:
        session_id = str(uuid.uuid4())
        now = datetime.now()

        session = ConversationSession(
            situation=situation,
            session_id=session_id,
            relationship=relationship,
            age_differential=age_differential,
            setting=setting,
            formality_token=formality_token,
            formality_override=formality_override,
            created_at=now,
            last_used=now
        )

        self.sessions[session_id] = session
        logger.info(
            f"Created session {session_id}: "
            f"{relationship.value} / age_diff={age_differential} / "
            f"{setting.value} → {formality_token.as_token()}"
        )
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        session = self.sessions.get(session_id)

        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        if datetime.now() - session.last_used > self.timeout:
            logger.info(f"Session expired: {session_id}")
            del self.sessions[session_id]
            return None

        session.last_used = datetime.now()
        return session

    def update_session_usage(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session:
            session.last_used = datetime.now()

    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        now = datetime.now()
        expired = [
            sid for sid, s in self.sessions.items()
            if now - s.last_used > self.timeout
        ]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)

    def get_session_count(self) -> int:
        return len(self.sessions)


_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
