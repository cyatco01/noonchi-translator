"""
FastAPI application for Noonchi Translator.

Two-step flow:
  POST /api/set-context  — provide social context, get back resolved formality token
  POST /api/translate    — provide text, get Korean output conditioned on that token

The Claude API backend acts as a prototype / comparison baseline.
The same API contract will be satisfied by the fine-tuned mBART model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import get_settings
from models.schemas import (
    ContextRequest,
    ContextResponse,
    TranslationRequest,
    TranslationResponse,
    SocialContext,
    HealthResponse
)
from agents.claude_agent import ClaudeTranslationAgent
from session_manager import get_session_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

translation_agent: ClaudeTranslationAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global translation_agent

    settings = get_settings()
    is_valid, error_msg = settings.validate()
    if not is_valid:
        logger.error(f"Configuration error: {error_msg}")
        raise RuntimeError(error_msg)

    try:
        translation_agent = ClaudeTranslationAgent()
        logger.info("Translation agent initialized")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Noonchi Translator API",
    description=(
        "English-to-Korean translation with formality conditioning. "
        "Resolves social context to a formality token (<formal>/<polite>/<casual>) "
        "and produces Korean output at the appropriate speech level."
    ),
    version="1.0.0",
    lifespan=lifespan
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", message="API is operational")


@app.post("/api/set-context", response_model=ContextResponse)
async def set_context(request: ContextRequest):
    """
    Step 1: Provide social context. Returns the resolved formality token.

    The FormalityResolver applies rule-based pragmatic inference over
    relationship, age_differential, and setting to select one of:
    <formal>, <polite>, <casual>
    """
    if not translation_agent:
        raise HTTPException(status_code=503, detail="Translation service not available")

    try:
        context = SocialContext(
            relationship=request.relationship,
            age_differential=request.age_differential,
            setting=request.setting,
            formality_override=request.formality_override
        )

        formality_token = translation_agent.resolve_formality(context)

        session_manager = get_session_manager()
        session = session_manager.create_session(
            relationship=request.relationship,
            age_differential=request.age_differential,
            setting=request.setting,
            formality_token=formality_token,
            formality_override=request.formality_override
        )

        logger.info(
            f"Context set — relationship={request.relationship.value}, "
            f"age_diff={request.age_differential}, setting={request.setting.value} "
            f"→ {formality_token.as_token()} [session={session.session_id}]"
        )

        return ContextResponse(
            session_id=session.session_id,
            relationship=request.relationship,
            age_differential=request.age_differential,
            setting=request.setting,
            formality_token=formality_token,
            conditioning_input_prefix=formality_token.as_token(),
            message=(
                f"Formality resolved to {formality_token.as_token()}. "
                "You can now translate text using this session."
            )
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Context setup error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set context: {str(e)}")


@app.post("/api/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Step 2: Translate English text using the session's resolved formality token.

    Constructs the conditioned input ("<token> english text") and returns
    Korean output at the appropriate speech level.
    """
    if not translation_agent:
        raise HTTPException(status_code=503, detail="Translation service not available")

    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(request.session_id)

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired. Call /api/set-context first."
            )

        context = SocialContext(
            relationship=session.relationship,
            age_differential=session.age_differential,
            setting=session.setting,
            formality_override=session.formality_override
        )

        result = await translation_agent.translate(context=context, text=request.text)
        session_manager.update_session_usage(request.session_id)

        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
