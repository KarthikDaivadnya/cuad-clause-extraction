"""
app.py
------
Optional interactive demo

Two tabs:
    1. "Analyze a contract" — upload any contract PDF, pick a provider, and
       see extracted clauses + summary in real time.
    2. "Search processed contracts" — semantic search over whatever the
       batch pipeline (main.py) has already written to output/results.json.

Run with: streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

from src import config
from src.chunker import chunk_text
from src.clause_extractor import extract_all_clauses
from src.data_loader import _extract_pdf_text
from src.llm_provider import get_provider, LLMError
from src.preprocessor import normalize_text
from src.retriever import ChunkRetriever
from src.semantic_search import ClauseSearchIndex
from src.summarizer import summarize_contract

st.set_page_config(page_title="Contract Clause Extractor", page_icon="📄", layout="wide")
st.title("📄 LLM Contract Clause Extraction & Summarization")
st.caption("Built on the CUAD (Contract Understanding Atticus Dataset) clause taxonomy")

tab_analyze, tab_search = st.tabs(["🔍 Analyze a contract", "🔎 Search processed contracts"])

# ---------------------------------------------------------------------------
# TAB 1 — single-contract analysis
# ---------------------------------------------------------------------------
with tab_analyze:
    col_settings, col_main = st.columns([1, 2])

    with col_settings:
        st.subheader("Settings")
        provider_name = st.selectbox("LLM provider", ["groq", "openai", "anthropic"], index=0)
        default_model = config.DEFAULT_MODELS[provider_name]
        model = st.text_input("Model", value=default_model)
        uploaded = st.file_uploader("Upload a contract PDF", type=["pdf"])
        run_btn = st.button("Run analysis", type="primary", disabled=uploaded is None)

    with col_main:
        if run_btn and uploaded is not None:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            with st.spinner("Extracting text from PDF..."):
                raw_text = _extract_pdf_text(tmp_path)
                normalized = normalize_text(raw_text)
                chunks = chunk_text(normalized)

            if not chunks:
                st.error("Could not extract any text from this PDF (it may be a scanned image with no text layer).")
            else:
                st.success(f"Extracted {len(normalized):,} characters across {len(chunks)} chunk(s).")

                try:
                    provider = get_provider(provider_name, model)
                except LLMError as e:
                    st.error(str(e))
                    st.stop()

                with st.spinner("Building semantic index for retrieval..."):
                    retriever = ChunkRetriever(chunks)

                with st.spinner(f"Extracting clauses with {provider_name}/{model} ..."):
                    clauses = extract_all_clauses(provider, retriever)

                with st.spinner("Generating summary (map-reduce)..."):
                    summary = summarize_contract(provider, chunks)

                st.markdown("### Summary")
                st.info(summary)

                st.markdown("### Extracted Clauses")
                for clause_type, label in [
                    ("termination_clause", "⏹️ Termination"),
                    ("confidentiality_clause", "🔒 Confidentiality"),
                    ("liability_clause", "⚖️ Liability"),
                ]:
                    data = clauses[clause_type]
                    with st.expander(f"{label} — {'✅ found' if data.get('found') else '❌ not found'}", expanded=True):
                        if data.get("found"):
                            st.write(data.get("clause_text", ""))
                            if data.get("key_terms"):
                                st.json(data["key_terms"])
                        else:
                            st.write("No relevant clause was identified in this contract.")
        elif uploaded is None:
            st.info("Upload a contract PDF and click **Run analysis** to get started.")

# ---------------------------------------------------------------------------
# TAB 2 — semantic search over a previously-run batch
# ---------------------------------------------------------------------------
with tab_search:
    st.subheader("Semantic search across all processed contracts")
    st.caption("Run `python main.py` first to populate output/results.json")

    results_path = config.OUTPUT_DIR / "results.json"
    if not results_path.exists():
        st.warning(f"No results found at {results_path}. Run the batch pipeline first.")
    else:
        query = st.text_input("Search query", placeholder="e.g. what happens if a party misses a payment?")
        k = st.slider("Number of results", 1, 20, 5)
        if query:
            with st.spinner("Searching..."):
                index = ClauseSearchIndex.from_results_json(results_path)
                hits = index.search(query, k=k)
            if not hits:
                st.write("No matching clauses found.")
            for hit in hits:
                with st.container(border=True):
                    st.markdown(f"**{hit.contract_id}** · `{hit.clause_type}` · score {hit.score:.2f}")
                    st.write(hit.clause_text)
