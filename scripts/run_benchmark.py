"""Manually run all benchmark queries.   Usage: python -m scripts.run_benchmark"""
from src.graph import run_rag

BENCHMARK_QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "What real-world use cases for Agentic AI are discussed?",
    "What is the capital of France?",
]


def main() -> None:
    for i, q in enumerate(BENCHMARK_QUERIES, 1):
        r = run_rag(q)
        print("=" * 80)
        print(f"[{i}] Query:      {q}")
        print(f"    Answer:     {r['answer']}")
        print(f"    Confidence: {r.get('confidence_score', 0.0)}   Grounded: {r.get('grounded')}")
        chunks = r.get("context", [])
        print(f"    Retrieved chunks: {len(chunks)}")
        for j, c in enumerate(chunks, 1):
            print(f"      ({j}) page {c['page']} score {c['score']:.3f}: {c['text'][:200].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
