import streamlit as st
from src.config import CHAT_MODEL, EMBEDDING_MODEL, PROVIDER_NAME, TOP_K
from src.evaluate import SAMPLE_QUESTIONS
from src.qa import answer_question
from src.vector_store import RAGError

st.set_page_config(page_title="GPO Annual Report Q&A", page_icon="📄")
st.title("GPO FY 2018 Annual Report Q&A")
st.caption("Answers use only the report and cite physical PDF page numbers (the page a PDF viewer shows).")

def escape_dollars(text: str) -> str:
    # Streamlit renders $...$ as LaTeX, which would garble amounts like "$375 million ... $375.7 million".
    return text.replace("$", "\\$")

def show_result(result: dict) -> None:
    st.markdown(escape_dollars(result["answer"]))
    pages = sorted({source["page_number"] for source in result["sources"]})
    st.caption(f"Retrieved PDF pages: {', '.join(map(str, pages))}")
    if result["unsupported_citations"]:
        st.warning(f"Answer cites pages that were not retrieved: {result['unsupported_citations']}")
    with st.expander("Retrieved passages"):
        for rank, source in enumerate(result["sources"], 1):
            st.markdown(f"**{rank}. PDF p. {source['page_number']}** · distance {source['distance']:.3f}")
            st.text(source["text"])

def show_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if "result" in message:
            show_result(message["result"])
        elif "error" in message:
            st.error(message["error"])
        else:
            st.markdown(escape_dollars(message["content"]))

messages = st.session_state.setdefault("messages", [])

with st.sidebar:
    st.subheader("Sample questions")
    for sample in SAMPLE_QUESTIONS:
        if st.button(sample):
            st.session_state.pending = sample
    st.divider()
    st.caption(f"{PROVIDER_NAME} · chat `{CHAT_MODEL}` · embeddings `{EMBEDDING_MODEL}` · top {TOP_K} passages")
    if st.button("Clear chat"):
        messages.clear()

for message in messages:
    show_message(message)

if question := st.chat_input("Ask a question about the report") or st.session_state.pop("pending", None):
    messages.append({"role": "user", "content": question})
    show_message(messages[-1])
    with st.spinner("Searching the report..."):
        try:
            reply = {"role": "assistant", "result": answer_question(question)}
        except (RAGError, ValueError) as exc:
            reply = {"role": "assistant", "error": str(exc)}
    messages.append(reply)
    show_message(reply)
