# Interview Prep

## Core concepts (simple language)

1. **What is RAG?** Instead of asking an LLM to answer from memory, we first *retrieve* relevant passages from our own documents, then ask the LLM to *generate* an answer using only those passages. It gives up-to-date, private, checkable answers.
2. **Why Pinecone?** It is a managed vector database: fast nearest-neighbour search, metadata storage, scaling and persistence without running infrastructure. Alternatives: pgvector, Qdrant, Chroma, FAISS.
3. **Why embeddings?** They turn text into vectors where similar *meaning* is close together, so "What is an autonomous agent?" matches a passage that says "software that acts on its own" even without shared keywords.
4. **Why LangGraph over a simple chain?** We need a *conditional branch* (refuse vs. generate) and explicit shared state. LangGraph makes the flow a visible graph that is easy to extend (retries, rerank, human review) and to trace.
5. **Nodes.** `retrieve`: embed the question, query Pinecone, drop chunks below the threshold. `generate`: answer strictly from context at temperature 0. `grounding_check`: a second LLM call verifies the answer against the context and computes confidence. `refuse`: fixed refusal when nothing relevant was found (no LLM call).
6. **Hallucination prevention.** Layers: relevance threshold, refuse without calling the LLM when nothing is relevant, strict prompt, temperature 0, LLM-based grounding check that replaces unsupported answers with a refusal. This reduces hallucination; it does not eliminate it.
7. **Confidence.** `0.4 * retrieval_relevance + 0.6 * answer_support`. Relevance = mean cosine score of the used chunks rescaled between the threshold and 0.75; support = 1 / 0.5 / 0 from the judge. Refusals = 0. It is a heuristic, not a probability.
8. **Irrelevant retrieval?** Pinecone always returns the top-k nearest vectors, even if they are poor. We filter by score; if nothing passes, the `refuse` node returns "I cannot answer based on the provided document." with confidence 0.
9. **Why chunk overlap?** A sentence or idea can straddle a chunk boundary. 200 characters of overlap keeps that context in at least one chunk, at the price of a bit more storage.
10. **Why cosine similarity?** It compares vector *direction* (meaning), not length, so it is robust to text length differences. OpenAI embeddings are normalised, so cosine, dot product and Euclidean rank identically; cosine is the conventional, readable choice.
11. **Why FastAPI?** Type-hinted Pydantic models give validation and auto docs for free, it is fast, and it is the standard for Python ML services.
12. **Production improvements.** Reranking, hybrid search, an evaluation set with automated regression tests, tracing (LangSmith), caching, streaming, auth/rate limits, conversation memory, citations with page numbers, threshold calibration on labelled data, async clients, CI/CD and secrets management.

## 10 likely questions with model answers

1. **Walk me through the request flow.** `/chat` validates input, invokes the graph. `retrieve` embeds and queries Pinecone (top 3), filters by threshold. If any chunk survives, `generate` answers from those chunks, then `grounding_check` verifies and scores; otherwise `refuse` returns the refusal. FastAPI maps state to the response model.
2. **How do you stop "capital of France" from returning Paris?** Its embedding is far from every eBook chunk, so no chunk passes the threshold and the `refuse` node answers without calling the LLM. If a chunk did slip through, the strict prompt and grounding check are the second and third line of defence.
3. **How did you choose the threshold (0.30)?** It is a starting default for `text-embedding-3-small`, not a proven value. The right way is to run in-scope and out-of-scope questions, look at the score distributions, and pick a cut-off that separates them; it is configurable for that reason.
4. **Why a second LLM call for grounding?** Retrieval score says the context is relevant, not that the answer stayed inside it. An independent verification pass catches added claims. Cost: extra latency and tokens.
5. **Can the judge be wrong?** Yes. It is an LLM. It is a mitigation, so I call the score a heuristic and would measure judge accuracy against human labels in production.
6. **How is ingestion idempotent?** Deterministic IDs from a hash of source, page, chunk index and text; upsert overwrites, and existing IDs are fetched first so they are not re-embedded.
7. **Why not `langchain-pinecone`?** Calling the Pinecone SDK directly gives control over IDs, metadata and raw scores, which the threshold logic needs, and it means fewer moving parts to explain.
8. **Why temperature 0?** Determinism and less creative drift; for a strictly extractive task we want the most likely, reproducible output.
9. **What are the weaknesses?** Uncalibrated thresholds/weights, no reranking or hybrid search, text-only parsing, no memory, judge errors, and 2 LLM calls per answer.
10. **How would you evaluate it?** Build a labelled set of in-scope and out-of-scope questions; measure retrieval hit rate, faithfulness, answer correctness, refusal precision/recall; run it in CI whenever prompts, chunking or the threshold change.
