"""
app.py
------
Optional interactive demo (not required by the assignment, but a fast way
for a reviewer to see the pipeline work without touching the CLI).

Two tabs:
    1. "Analyze a contract" — upload any contract PDF, pick a provider, and
       see extracted clauses + summary in real time, PLUS:
         - a confidence score (retrieval similarity) for every extraction,
           which works for ANY uploaded contract, and
         - an accuracy check against CUAD's own gold labels, which only
           applies when the upload happens to match a known CUAD contract
           (accuracy requires ground truth -- an arbitrary uploaded PDF has
           none, so this never fakes a number for a document CUAD has no
           annotation for).
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

# Reused unmodified from scripts/evaluate_accuracy.py -- one implementation
# of the gold-label logic, shared by main.py, the notebook, and this app.
from scripts.evaluate_accuracy import (
    build_gold_lookup, load_cuad_gold, match_to_gold, normalize, text_similarity,
    GOLD_CATEGORY_MAP, NOT_EVALUABLE_CLAUSES,
)

st.set_page_config(page_title="Contract Clause Extractor", page_icon="📄", layout="wide")
st.title("📄 LLM Contract Clause Extraction & Summarization")
st.caption("Built on the CUAD (Contract Understanding Atticus Dataset) clause taxonomy")

tab_analyze, tab_search = st.tabs(["🔍 Analyze a contract", "🔎 Search processed contracts"])


@st.cache_resource(show_spinner="Loading CUAD gold labels (first time only, then cached)...")
def get_gold_lookup():
    cuad_data = load_cuad_gold(config.DATA_DIR / ".cuad_cache")
    return build_gold_lookup(cuad_data)


def confidence_badge(score: float) -> str:
    """Qualitative label for a retrieval similarity score. Thresholds are a
    reasonable heuristic for sentence-transformer cosine similarity on prose
    retrieval, not a calibrated probability -- shown as a signal, not a fact."""
    if score >= 0.5:
        return "🟢 High"
    if score >= 0.35:
        return "🟡 Medium"
    return "🔴 Low"


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

                # Retrieval similarity scores, recomputed here purely for the
                # confidence display -- cheap (embedding lookup, no LLM call),
                # not returned by extract_all_clauses() itself.
                confidence_scores = {}
                for clause_type in config.CLAUSE_TYPES:
                    query = config.CLAUSE_RETRIEVAL_QUERIES[clause_type]
                    top = retriever.top_k(query, k=config.TOP_K_CHUNKS_PER_CLAUSE)
                    confidence_scores[clause_type] = max((s for _c, s in top), default=0.0)

                st.markdown("### Summary")
                st.info(summary)

                st.markdown("### Extracted Clauses")
                for clause_type, label in [
                    ("termination_clause", "⏹️ Termination"),
                    ("confidentiality_clause", "🔒 Confidentiality"),
                    ("liability_clause", "⚖️ Liability"),
                ]:
                    data = clauses[clause_type]
                    score = confidence_scores[clause_type]
                    badge = confidence_badge(score)
                    found_label = "✅ found" if data.get("found") else "❌ not found"
                    with st.expander(f"{label} — {found_label} — confidence: {badge} ({score:.2f})", expanded=True):
                        if data.get("found"):
                            st.write(data.get("clause_text", ""))
                            if data.get("key_terms"):
                                st.json(data["key_terms"])
                        else:
                            st.write("No relevant clause was identified in this contract.")
                        st.caption(
                            f"Confidence = top retrieval similarity score for this clause type "
                            f"({score:.2f}). This reflects how strongly the retrieved passages matched "
                            f"the query, not a guarantee of correctness -- treat 🔴 Low as 'review this one'."
                        )

                # -----------------------------------------------------------
                # Accuracy check -- ONLY meaningful if this upload happens to
                # be a contract that's actually in CUAD's gold-labeled set.
                # -----------------------------------------------------------
                st.markdown("### Accuracy check against CUAD gold labels")
                with st.expander("Run accuracy check (only applies if this is a known CUAD contract)"):
                    st.caption(
                        "Accuracy requires ground truth. This looks up whether the uploaded "
                        "file matches one of CUAD's 510 expert-annotated contracts; for any "
                        "other document (most real-world uploads), there is no gold label to "
                        "compare against, and no number is shown."
                    )
                    if st.button("Check against CUAD gold labels"):
                        with st.spinner("Loading CUAD gold labels and matching this contract..."):
                            gold_lookup = get_gold_lookup()
                            gold_titles = list(gold_lookup.keys())
                            norm_titles = {t: normalize(t) for t in gold_titles}
                            matched_title = match_to_gold(uploaded.name, gold_titles, norm_titles)

                        if matched_title is None:
                            st.warning(
                                "No confident match to a known CUAD contract -- accuracy isn't "
                                "computable for this upload. This is expected for any contract "
                                "outside the CUAD dataset; the confidence scores above are the "
                                "right signal to use instead."
                            )
                        else:
                            st.success(f"Matched to CUAD contract: **{matched_title}**")
                            gold_entry = gold_lookup[matched_title]
                            rows = []
                            for clause_type in config.CLAUSE_TYPES:
                                if clause_type in NOT_EVALUABLE_CLAUSES:
                                    rows.append({
                                        "clause_type": clause_type, "gold_present": "N/A",
                                        "extracted_found": clauses[clause_type].get("found", False),
                                        "correct": "not evaluable (no CUAD gold category)", "overlap": "",
                                    })
                                    continue
                                gold_text = gold_entry[clause_type]
                                gold_present = gold_text is not None
                                found = bool(clauses[clause_type].get("found", False))
                                overlap = (
                                    text_similarity(clauses[clause_type].get("clause_text", ""), gold_text)
                                    if (found and gold_present) else None
                                )
                                rows.append({
                                    "clause_type": clause_type,
                                    "gold_present": gold_present,
                                    "extracted_found": found,
                                    "correct": found == gold_present,
                                    "overlap": overlap if overlap is not None else "",
                                })
                            st.table(rows)
        elif uploaded is None:
            st.info("Upload a contract PDF and click **Run analysis** to get started.")

# ---------------------------------------------------------------------------
# TAB 2 — semantic search over a previously-run batch (bonus feature)
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