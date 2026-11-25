from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, load_settings
from app.db import Database
from app.llm import AllProvidersUnavailableError, LLMGateway
from app.schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionMessagesResponse,
    StoredMessage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None,
    db: Optional[Database] = None,
    gateway: Optional[LLMGateway] = None,
) -> FastAPI:
    app_settings = settings or load_settings()
    app_db = db or Database(app_settings.database_url)
    app_gateway = gateway or LLMGateway(app_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        app_db.init_schema()
        yield

    app = FastAPI(
        title="Lyftr AI LLM Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.settings = app_settings
    app.state.db = app_db
    app.state.gateway = app_gateway

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            db_ok = app_db.check_health()
        except Exception as exc:
            logger.error("Database health check failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="database unavailable") from exc

        if not db_ok:
            raise HTTPException(status_code=503, detail="database unavailable")

        keys_count = len(app_settings.keys)
        if keys_count == 0:
            raise HTTPException(status_code=503, detail="no API keys configured")

        return HealthResponse(
            status="ok",
            keys_configured=keys_count,
            cache_enabled=app_gateway.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ) -> ChatResponse:
        stored_prompt, history = app_db.get_session_context(payload.session_id)
        effective_system_prompt = (
            payload.system_prompt
            if payload.system_prompt is not None
            else (stored_prompt or app_settings.default_system_prompt)
        )

        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": effective_system_prompt},
            *history,
            {"role": "user", "content": payload.message},
        ]

        try:
            result = app_gateway.generate_reply(
                messages=llm_messages,
                simulate_failure_header=x_simulate_failure,
            )
        except AllProvidersUnavailableError as exc:
            raise HTTPException(status_code=503, detail="all providers unavailable") from exc

        metadata = ChatMetadata(
            resolution_source=result.resolution_source,
            model=result.model,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )

        app_db.save_turn(
            session_id=payload.session_id,
            user_content=payload.message,
            assistant_content=result.reply,
            metadata=metadata.model_dump(),
            system_prompt=payload.system_prompt,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            metadata=metadata,
        )

    @app.get(
        "/sessions/{session_id}/messages",
        response_model=SessionMessagesResponse,
        response_model_exclude_none=True,
    )
    def get_session_messages(session_id: str) -> SessionMessagesResponse:
        raw_messages = app_db.get_session_messages(session_id)
        if raw_messages is None:
            raise HTTPException(status_code=404, detail="session not found")

        messages = [StoredMessage(**item) for item in raw_messages]
        return SessionMessagesResponse(session_id=session_id, messages=messages)

    return app


app = create_app()
