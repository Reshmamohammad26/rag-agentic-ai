"""FastAPI application exposing the RAG chatbot."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import ConfigError
from src.graph import run_rag

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

app = FastAPI(title="Agentic AI RAG Chatbot", version="1.0.0")


class ChatRequest(BaseModel):
    query: str = Field(..., max_length=2000, examples=["What is Agentic AI?"])

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # Plain `def` -> FastAPI runs it in a threadpool, so blocking SDK calls don't block the event loop.
    try:
        result = run_rag(request.query)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG pipeline failed")
        raise HTTPException(status_code=502, detail="Upstream service error (OpenAI/Pinecone).") from exc

    return ChatResponse(
        query=request.query,
        final_answer=result["answer"],
        retrieved_context_chunks=[c["text"] for c in result.get("context", [])],
        confidence_score=result.get("confidence_score", 0.0),
    )
