import sys
from .qa import answer_question

def print_answer(question: str) -> bool:
    try:
        result = answer_question(question)
    except Exception as exc:
        print(f"Error: {exc}")
        return False
    print(f"\nAnswer:\n{result['answer']}")
    pages = sorted({source["page_number"] for source in result["sources"]})
    print(f"Retrieved PDF pages: {', '.join(map(str, pages))}")
    if result["unsupported_citations"]:
        print(f"Warning: answer cites pages that were not retrieved: {result['unsupported_citations']}")
    return True

def main() -> int:
    if len(sys.argv) > 1:
        return 0 if print_answer(" ".join(sys.argv[1:])) else 1
    print("GPO Annual Report RAG — type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print_answer(question)
    return 0

if __name__ == "__main__":
    sys.exit(main())
