import re
from .config import CHAT_MODEL, TOP_K
from .vector_store import get_openai_client, openai_errors, retrieve

SYSTEM_PROMPT = """You answer questions using only the supplied excerpts from the GPO FY 2018 Annual Report.
- Use only facts supported by excerpts; do not guess or use outside knowledge.
- Preserve exact quantities, units, names, and dates.
- If the excerpts do not support an answer, say: 'I couldn't verify that from the retrieved report excerpts.'
- Cite factual claims using page citations such as [PDF p. 12], based on the labels supplied with excerpts.
  Cite each page separately, e.g. [PDF p. 5] [PDF p. 21].
- Be concise. If evidence is ambiguous, state that clearly.
"""

CITATION_PATTERN = re.compile(r"\[PDF pp?\.\s*([^\]]+)\]")

def cited_pages(answer: str) -> list[int]:
    return sorted({int(number) for group in CITATION_PATTERN.findall(answer) for number in re.findall(r"\d+", group)})

def answer_question(question: str, top_k: int = TOP_K) -> dict:
    hits = retrieve(question, top_k=top_k)
    context = "\n\n".join(f"[PDF p. {hit['page_number']}]\n{hit['text']}" for hit in hits)
    client = get_openai_client()
    with openai_errors(CHAT_MODEL):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nRetrieved report excerpts:\n{context}"},
            ],
        )
    answer = response.choices[0].message.content or ""
    retrieved = {hit["page_number"] for hit in hits}
    citations = cited_pages(answer)
    return {
        "question": question,
        "answer": answer,
        "cited_pages": citations,
        # Pages the answer cites that were not among the retrieved excerpts (should always be empty).
        "unsupported_citations": [page for page in citations if page not in retrieved],
        "sources": [
            {"chunk_id": h["chunk_id"], "page_number": h["page_number"], "distance": h["distance"], "text": h["text"]}
            for h in hits
        ],
    }
