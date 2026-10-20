import json
import sys
from .config import ROOT_DIR, LLM_PROVIDER, CHAT_MODEL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from .qa import answer_question
from .vector_store import RAGError

SAMPLE_QUESTIONS = [
    "How many depository libraries did GPO staff visit in FY 2018?",
    "What new digital system replaced FDsys, and when was it fully deployed?",
    "How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies?",
]

def main() -> int:
    results = []
    for question in SAMPLE_QUESTIONS:
        print("=" * 80)
        print(f"QUESTION: {question}")
        try:
            result = answer_question(question)
        except (RAGError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        pages = sorted({s["page_number"] for s in result["sources"]})
        print(f"ANSWER: {result['answer']}\nRETRIEVED PAGES: {pages}\nCITED PAGES: {result['cited_pages']}")
        if not result["cited_pages"]:
            print("WARNING: answer has no page citations.")
        if result["unsupported_citations"]:
            print(f"WARNING: answer cites pages that were not retrieved: {result['unsupported_citations']}")
        for rank, source in enumerate(result["sources"], 1):
            print(f"  {rank}. [PDF p. {source['page_number']}] distance={source['distance']:.3f} {source['text'][:140]}...")
        results.append({
            "question": question,
            "answer": result["answer"],
            "retrieved_pages": pages,
            "cited_pages": result["cited_pages"],
            "unsupported_citations": result["unsupported_citations"],
            "sources": result["sources"],
        })
    output = ROOT_DIR / "evaluation_results.json"
    run_config = {
        "provider": LLM_PROVIDER,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
    }
    output.write_text(json.dumps({"config": run_config, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
