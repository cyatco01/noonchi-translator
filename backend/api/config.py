"""
Configuration management for Noonchi Translator backend.

Loads environment variables and provides centralized config access.
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file at project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Settings:
    """Application settings loaded from environment variables."""

    # Backend selection
    BACKEND: str = os.getenv('BACKEND', 'claude')               # 'claude' | 'mbart'
    MBART_MODEL_DIR: str = os.getenv('MBART_MODEL_DIR', 'models/noonchi-mbart')

    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')

    # Claude Model Configuration
    CLAUDE_MODEL: str = os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-6')
    MAX_TOKENS: int = int(os.getenv('MAX_TOKENS', '4096'))
    TEMPERATURE: float = float(os.getenv('TEMPERATURE', '0.3'))  # Lower temp for consistent translations

    # API Configuration
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))

    # CORS Configuration
    CORS_ORIGINS: list = os.getenv('CORS_ORIGINS', 'http://localhost:5173').split(',')

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / 'data'

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate required settings.

        Returns:
            tuple: (is_valid, error_message)
        """
        if not self.ANTHROPIC_API_KEY:
            return False, "ANTHROPIC_API_KEY not set. Required for context parsing."

        if self.BACKEND == 'mbart' and not Path(self.MBART_MODEL_DIR).exists():
            return False, f"MBART model directory not found: {self.MBART_MODEL_DIR}"

        if not self.DATA_DIR.exists():
            return False, f"Data directory not found: {self.DATA_DIR}"

        return True, None


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
