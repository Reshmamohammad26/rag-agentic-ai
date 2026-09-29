"""Tests.

Unit tests   : run offline, no API keys needed (mocks / pure functions).
Integration  : marked `integration`; need real keys + an ingested index. They are
               skipped automatically when keys are missing.

Run all:            pytest -v
Unit only:          pytest -v -m "not integration"
Integration only:   pytest -v -m integration -s
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import app as app_module
from src import config, graph, ingestion
from src.prompts import REFUSAL_MESSAGE

# --------------------------------------------------------------------------- unit
def test_missing_env_raises(monkeypatch):
    for name in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"):
        monkeypatch.setenv(name, "")
    config.get_settings.cache_clear()
    with pytest.raises(config.ConfigError):
        config.get_settings()
    config.get_settings.cache_clear()


def test_chunk_id_is_deterministic_and_unique():
    from langchain_core.documents import Document

    meta = {"source": "a.pdf", "page": 1, "chunk_index": 0}
    d1 = Document(page_content="hello", metadata=dict(meta))
    d2 = Document(page_content="hello", metadata=dict(meta))
    d3 = Document(page_content="hello!", metadata=dict(meta))
    assert ingestion.chunk_id(d1) == ingestion.chunk_id(d2)
    assert ingestion.chunk_id(d1) != ingestion.chunk_id(d3)


def test_route_after_retrieve():
    assert graph.route_after_retrieve({"context": []}) == "refuse"
    assert graph.route_after_retrieve({"context": [{"text": "x"}]}) == "generate"


def test_refuse_node_returns_zero_confidence():
    out = graph.refuse({"question": "What is the capital of France?"})
    assert out["answer"] == REFUSAL_MESSAGE
    assert out["confidence_score"] == 0.0
    assert out["grounded"] is False


def test_confidence_bounds_and_ordering():
    t = 0.30
    assert graph.compute_confidence([], 1.0, t) == 0.0
    assert graph.compute_confidence([0.9], 0.0, t) == 0.0
    high = graph.compute_confidence([0.8, 0.8], 1.0, t)
    mid = graph.compute_confidence([0.4, 0.4], 1.0, t)
    partial = graph.compute_confidence([0.8, 0.8], 0.5, t)
    assert 0.0 <= mid < high <= 1.0
    assert partial < high


def test_is_refusal():
    assert graph.is_refusal("I cannot answer based on the provided document.")
    assert not graph.is_refusal("Agentic AI is ...")


client = TestClient(app_module.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_chat_rejects_blank_query():
    assert client.post("/chat", json={"query": "   "}).status_code == 422


def test_chat_response_shape_with_mocked_pipeline(monkeypatch):
    fake = {
        "answer": "Agentic AI is ...",
        "context": [{"text": "chunk one", "page": 2, "source": "x.pdf", "score": 0.6}],
        "confidence_score": 0.8,
        "grounded": True,
    }
    monkeypatch.setattr(app_module, "run_rag", lambda q: fake)
    body = client.post("/chat", json={"query": "What is Agentic AI?"}).json()
    assert set(body) == {"query", "final_answer", "retrieved_context_chunks", "confidence_score"}
    assert body["retrieved_context_chunks"] == ["chunk one"]


def test_chat_upstream_failure_returns_502(monkeypatch):
    def boom(q):
        raise RuntimeError("openai down")

    monkeypatch.setattr(app_module, "run_rag", boom)
    assert client.post("/chat", json={"query": "hi"}).status_code == 502


# ------------------------------------------------------------------- integration
HAVE_KEYS = all(os.getenv(k) for k in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"))
integration = pytest.mark.skipif(not HAVE_KEYS, reason="API keys not configured")

IN_SCOPE = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "What real-world use cases for Agentic AI are discussed?",
]


@pytest.mark.integration
@integration
@pytest.mark.parametrize("question", IN_SCOPE)
def test_in_scope_questions_are_answered_from_document(question):
    result = graph.run_rag(question)
    print(f"\nQ: {question}\nA: {result['answer']}\nconfidence={result['confidence_score']}")
    assert result["answer"] != REFUSAL_MESSAGE
    assert len(result["context"]) > 0
    assert result["confidence_score"] > 0.0


@pytest.mark.integration
@integration
@pytest.mark.parametrize(
    "question",
    ["What is the capital of France?", "Who won the FIFA World Cup in 2018?", "Write a Python function to sort a list."],
)
def test_out_of_scope_questions_are_refused(question):
    result = graph.run_rag(question)
    assert "paris" not in result["answer"].lower()
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["confidence_score"] <= 0.2
