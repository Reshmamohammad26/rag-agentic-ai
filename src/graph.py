r"""LangGraph RAG workflow.

    START -> retrieve --(relevant chunks?)--> generate -> grounding_check -> END
                      \--(none relevant)----> refuse ------------------------> END
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone
from pydantic import BaseModel, Field

from src.config import get_settings
from src.prompts import (
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_TEMPLATE,
    GROUNDING_SYSTEM_PROMPT,
    GROUNDING_USER_TEMPLATE,
    REFUSAL_MESSAGE,
)

logger = logging.getLogger(__name__)

# ---- confidence weights (heuristic, documented in README) -------------------
WEIGHT_RETRIEVAL = 0.4
WEIGHT_SUPPORT = 0.6
SCORE_CEILING = 0.75  # cosine score treated as "excellent match"
SUPPORT_VALUES = {"fully_supported": 1.0, "partially_supported": 0.5, "not_supported": 0.0}


class AgentState(TypedDict, total=False):
    question: str
    context: list[dict]  # relevant chunks: text, page, source, score
    answer: str
    confidence_score: float
    grounded: bool


class GroundingVerdict(BaseModel):
    """Structured output of the LLM fact-checker."""

    support: Literal["fully_supported", "partially_supported", "not_supported"]
    unsupported_claims: list[str] = Field(description="Claims not backed by the context")


# ---- lazily created clients (so importing this module needs no API keys) ----
@lru_cache(maxsize=1)
def _embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key)


@lru_cache(maxsize=1)
def _index():
    s = get_settings()
    return Pinecone(api_key=s.pinecone_api_key).Index(s.pinecone_index_name)


@lru_cache(maxsize=1)
def _llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(model=s.chat_model, temperature=0, api_key=s.openai_api_key)


# ---- helpers ------------------------------------------------------------------
def format_context(context: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in context
    )


def is_refusal(answer: str) -> bool:
    return REFUSAL_MESSAGE.lower().rstrip(".") in answer.lower()


def compute_confidence(scores: list[float], support: float, threshold: float) -> float:
    """Heuristic confidence in [0, 1].

    0.4 * retrieval_relevance + 0.6 * answer_support
      retrieval_relevance: mean cosine score of the used chunks, rescaled so that
                           `threshold` -> 0 and SCORE_CEILING -> 1 (clamped).
      answer_support:      1.0 / 0.5 / 0.0 from the LLM fact-checker.
    No relevant chunks or unsupported answer -> 0.0. This is a calibrated-by-hand
    heuristic, NOT a probability.
    """
    if not scores or support <= 0.0:
        return 0.0
    span = max(SCORE_CEILING - threshold, 1e-6)
    relevance = min(max((sum(scores) / len(scores) - threshold) / span, 0.0), 1.0)
    return round(WEIGHT_RETRIEVAL * relevance + WEIGHT_SUPPORT * support, 2)


# ---- nodes ----------------------------------------------------------------------
def retrieve(state: AgentState) -> AgentState:
    """Embed the question, query Pinecone, keep only chunks above the threshold."""
    s = get_settings()
    vector = _embeddings().embed_query(state["question"])
    result = _index().query(vector=vector, top_k=s.top_k, include_metadata=True)
    context = [
        {
            "text": m.metadata["text"],
            "page": int(m.metadata["page"]),
            "source": m.metadata["source"],
            "score": float(m.score),
        }
        for m in result.matches
        if float(m.score) >= s.relevance_threshold
    ]
    logger.info(
        "Retrieved %d matches, %d above threshold %.2f",
        len(result.matches), len(context), s.relevance_threshold,
    )
    return {"context": context}


def route_after_retrieve(state: AgentState) -> Literal["generate", "refuse"]:
    return "generate" if state.get("context") else "refuse"


def refuse(state: AgentState) -> AgentState:
    """No relevant context: do NOT call the LLM at all."""
    return {"answer": REFUSAL_MESSAGE, "confidence_score": 0.0, "grounded": False, "context": []}


def generate(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content=GENERATION_SYSTEM_PROMPT),
        HumanMessage(
            content=GENERATION_USER_TEMPLATE.format(
                context=format_context(state["context"]), question=state["question"]
            )
        ),
    ]
    answer = _llm().invoke(messages).content
    return {"answer": str(answer).strip()}


def grounding_check(state: AgentState) -> AgentState:
    """Second LLM call verifies the answer against the context, then scores confidence."""
    s = get_settings()
    answer = state["answer"]

    if is_refusal(answer):  # the model itself found nothing usable
        return {"answer": REFUSAL_MESSAGE, "confidence_score": 0.0, "grounded": False}

    judge = _llm().with_structured_output(GroundingVerdict)
    verdict: GroundingVerdict = judge.invoke(
        [
            SystemMessage(content=GROUNDING_SYSTEM_PROMPT),
            HumanMessage(
                content=GROUNDING_USER_TEMPLATE.format(
                    context=format_context(state["context"]),
                    question=state["question"],
                    answer=answer,
                )
            ),
        ]
    )
    support = SUPPORT_VALUES[verdict.support]
    if verdict.unsupported_claims:
        logger.info("Unsupported claims: %s", verdict.unsupported_claims)

    if verdict.support == "not_supported":  # never return an ungrounded answer
        return {"answer": REFUSAL_MESSAGE, "confidence_score": 0.0, "grounded": False}

    scores = [c["score"] for c in state["context"]]
    return {
        "confidence_score": compute_confidence(scores, support, s.relevance_threshold),
        "grounded": verdict.support == "fully_supported",
    }


# ---- graph ------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("grounding_check", grounding_check)
    g.add_node("refuse", refuse)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", "grounding_check")
    g.add_edge("grounding_check", END)
    g.add_edge("refuse", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def run_rag(question: str) -> AgentState:
    """Public entry point used by the API, tests and the benchmark script."""
    result = get_graph().invoke({"question": question})
    result.setdefault("context", [])
    return result
