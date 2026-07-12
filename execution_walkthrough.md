# How This Project Actually Runs — Step-by-Step Execution Walkthrough

This traces **exactly what happens, in order**, when you run:

```bash
python main.py --data-dir data/full_contract_pdf_sample --limit 50
```

Every section below is a real step in that execution, with the real code and
what it does. Read top to bottom and you're reading the actual call stack.

---

## 0. Entry point: `main.py`

This is what actually runs when you type the command. It parses CLI flags,
checks the data folder exists, then hands off everything to `run_pipeline()`.

```python
def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, ...)

    if not args.data_dir.exists():
        print(f"[!] Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    df = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        provider_name=args.provider,
        model=args.model,
        use_cache=not args.no_cache,
    )
```

Nothing clever here — its whole job is: validate inputs, call the real
pipeline, print a final tally. Everything interesting happens inside
`run_pipeline()`.

---

## 1. `run_pipeline()` in `src/pipeline.py` — the orchestrator

This is the top of the real call stack. Four things happen here, in order:

```python
def run_pipeline(data_dir, output_dir, limit=50, provider_name=None, model=None, use_cache=True):
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = get_provider(provider_name, model)          # Step 2
    contracts = list(load_contracts(data_dir, limit=limit)) # Step 3

    results = []
    for contract in tqdm(contracts, desc="Processing contracts"):
        result = process_contract(provider, contract, use_cache=use_cache)  # Steps 4-8
        results.append(result)

    # write CSV/JSON                                        # Step 9
```

Note the `try/except` around each contract inside `process_contract` (not
shown above) — one bad PDF or one LLM hiccup logs an error and moves to the
next contract instead of killing the whole 50-contract batch.

---

## 2. `get_provider()` in `src/llm_provider.py` — pick the LLM backend

```python
def get_provider(name: str = None, model: str = None) -> LLMProvider:
    name = (name or config.DEFAULT_PROVIDER).lower()   # "groq" by default
    model = model or config.DEFAULT_MODELS[name]
    return _PROVIDER_MAP[name](model)                  # e.g. GroqProvider("llama-3.3-70b-versatile")
```

`_PROVIDER_MAP` is just `{"groq": GroqProvider, "openai": OpenAIProvider, "anthropic": AnthropicProvider}`.
Whichever one gets picked, its `__init__` reads the matching API key from
`.env` and constructs the real SDK client:

```python
class GroqProvider(LLMProvider):
    def __init__(self, model: str):
        super().__init__(model)
        from groq import Groq
        api_key = config.API_KEYS["groq"]
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set. Add it to your .env file.")
        self.client = Groq(api_key=api_key)
```

From this point on, the rest of the pipeline only ever calls
`provider.complete(prompt, system, max_tokens, temperature)` — it never
touches `GroqProvider`/`OpenAIProvider`/`AnthropicProvider` directly. That's
the whole point of this abstraction: swap the provider in one line, nothing
downstream changes.

`complete()` wraps the provider-specific `_call()` in retry logic:

```python
def complete(self, prompt, system="...", max_tokens=800, temperature=0.0):
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return self._call(system, prompt, max_tokens, temperature)
        except Exception as exc:
            wait = min(2 ** attempt, 20)   # exponential backoff: 2s, 4s, 8s, 16s, capped at 20s
            time.sleep(wait)
    raise LLMError(...)
```

---

## 3. `load_contracts()` in `src/data_loader.py` — real PDF → text

```python
def load_contracts(data_dir: Path, limit=50):
    pdf_paths = sorted(data_dir.rglob("*.PDF")) + sorted(data_dir.rglob("*.pdf"))
    pdf_paths = sorted(set(pdf_paths))[:limit]

    for pdf_path in pdf_paths:
        text = _extract_pdf_text(pdf_path)
        if not text.strip():
            continue   # skip scanned/image-only PDFs with no text layer
        yield Contract(contract_id=pdf_path.stem, source_path=str(pdf_path), raw_text=text)
```

The actual text extraction, page by page, with `pdfplumber`:

```python
def _extract_pdf_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""   # one bad page doesn't kill the whole document
            pages.append(text)
    return "\n".join(pages)
```

This is a **generator** (`yield`, not `return`) — contracts are read and
processed one at a time rather than all loaded into memory upfront, which
matters once you're pointed at hundreds of PDFs.

---

## 4. `process_contract()` in `src/pipeline.py` — per-contract work + cache check

Every contract funnels through this function. First thing it does: check the
disk cache before doing any real work.

```python
def process_contract(provider, contract, use_cache=True):
    cache_file = config.CACHE_DIR / f"{contract.contract_id}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())   # skip re-processing entirely

    normalized = normalize_text(contract.raw_text)      # Step 5
    chunks = chunk_text(normalized)                      # Step 6
    retriever = ChunkRetriever(chunks)                    # Step 7

    clauses = extract_all_clauses(provider, retriever)   # Step 8
    summary = summarize_contract(provider, chunks)        # Step 8b

    result = {"contract_id": ..., "summary": summary, **clauses}
    cache_file.write_text(json.dumps(result, indent=2))   # save for next run
    return result
```

This is why re-running `main.py` after tweaking a prompt is fast — anything
already processed just gets read off disk instead of re-billing the API.

---

## 5. `normalize_text()` in `src/preprocessor.py` — cleanup before chunking

```python
def normalize_text(raw_text: str) -> str:
    text = unicodedata.normalize("NFKC", raw_text)
    text = text.replace("\u2019", "'").replace("\u201c", '"')...   # smart quotes -> plain

    text = _HYPHEN_WRAP_RE.sub(r"\1\2", text)   # "termina-\ntion" -> "termination"
    text = _PAGE_NUM_RE.sub("\n", text)          # strip standalone page-number lines
    text = _MULTI_SPACE_RE.sub(" ", text)        # collapse "   " -> " "
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)   # collapse "\n\n\n\n" -> "\n\n"
    return text.strip()
```

Why this matters concretely: PDF text extraction often produces
`termina-\ntion` where a word wrapped across a line. Without this step,
neither keyword search nor embedding similarity would recognize that as the
word "termination".

---

## 6. `chunk_text()` in `src/chunker.py` — sliding window split

```python
def chunk_text(text, chunk_size=1800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # snap the boundary to a newline or ". " near `end` so we don't
        # split a sentence in half when we don't have to
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary == -1 or boundary <= start + chunk_size // 2:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(Chunk(chunk_id=len(chunks), text=text[start:end].strip(), start_char=start))
        start = max(end - overlap, start + 1)   # step forward, but overlap by 200 chars
    return chunks
```

The **overlap** matters: if a clause sentence happens to fall right at a
chunk boundary, overlap means it still appears whole in at least one chunk
instead of being cut in half between two.

A 3,000-character contract → 1-2 chunks. An 80-page contract → 40+ chunks.
Same code path either way.

---

## 7. `ChunkRetriever` in `src/retriever.py` — embed + similarity search

Two things happen here: embed every chunk once, then answer "which chunks
are most relevant to X" via cosine similarity.

```python
class ChunkRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self._embeddings = embed_texts([c.text for c in chunks])  # one embedding per chunk

    def top_k(self, query, k=4):
        query_vec = embed_texts([query])[0]
        scores = self._embeddings @ query_vec       # dot product = cosine similarity
                                                       # (embeddings are pre-normalized)
        top_idx = np.argsort(-scores)[:k]             # indices of the k highest scores
        return [(self.chunks[i], float(scores[i])) for i in top_idx]
```

`embed_texts` lazily loads `sentence-transformers` (`all-MiniLM-L6-v2`) once
per process and reuses it:

```python
def embed_texts(texts):
    model = _get_model()   # loads once, cached in a module-level singleton
    return np.asarray(model.encode(texts, normalize_embeddings=True))
```

This is the step that makes retrieval *semantic* rather than keyword-based —
a query like `"termination of agreement, notice period"` will match a chunk
saying `"either party may end this contract with 30 days' notice"` even
though it shares almost no exact words with the query.

---

## 8. `extract_all_clauses()` in `src/clause_extractor.py` — Part A

```python
def extract_all_clauses(provider, retriever):
    return {ct: extract_clause(provider, retriever, ct) for ct in config.CLAUSE_TYPES}
    # CLAUSE_TYPES = ["termination_clause", "confidentiality_clause", "liability_clause"]
```

Three calls to `extract_clause`, one per clause type. Each one:

```python
def extract_clause(provider, retriever, clause_type):
    query = config.CLAUSE_RETRIEVAL_QUERIES[clause_type]   # e.g. "termination of agreement, notice period..."
    top_chunks = retriever.top_k(query, k=4)                # Step 7 in action

    if not top_chunks:
        return {"found": False, "clause_text": "", "key_terms": {}}

    passages = "\n---\n".join(c.text for c, _score in top_chunks)
    example = FEW_SHOT_EXAMPLES[clause_type][0]              # from prompts/few_shot_examples.py

    prompt = _INSTRUCTION_TEMPLATE.format(
        clause_label=clause_type, example_passage=example["passage"],
        example_answer=json.dumps(example["answer"]), passages=passages,
    )
    raw_response = provider.complete(prompt=prompt, system=_SYSTEM_PROMPT, max_tokens=500)
    return _extract_json(raw_response)    # parses the LLM's JSON reply defensively
```

The **only** text ever sent to the LLM for this clause type is: the few-shot
example + the 4 retrieved chunks (not the whole contract). That's the
"handling large text efficiently" design point in practice.

`_extract_json` exists because LLMs sometimes wrap JSON in markdown fences or
add stray commentary — it pulls out the first `{...}` block and parses it,
falling back to `{"found": false, ..., "parse_error": True}` rather than
crashing the batch if parsing fails.

---

## 8b. `summarize_contract()` in `src/summarizer.py` — Part B (map-reduce)

Two phases:

**MAP** — summarize each chunk individually:
```python
def summarize_contract(provider, chunks):
    notes = []
    for chunk in chunks:
        bullets = provider.complete(prompt=_MAP_PROMPT.format(chunk_text=chunk.text), system=_MAP_SYSTEM, max_tokens=150)
        if bullets.strip().upper() != "N/A":
            notes.append(bullets.strip())
```

**REDUCE** — combine all the bullet notes into one summary:
```python
    reduce_prompt = _REDUCE_PROMPT.format(min_words=100, max_words=150, notes="\n\n".join(notes))
    return provider.complete(prompt=reduce_prompt, system=_REDUCE_SYSTEM, max_tokens=350)
```

Why not just paste the whole contract into one summarization prompt? Because
on an 80-page contract that either truncates (losing content) or blows the
model's context window. Map-reduce costs more API calls but scales safely to
any contract length.

---

## 9. Back in `run_pipeline()` — writing the deliverable

```python
json_path.write_text(json.dumps(results, indent=2))    # full nested structure

flat_rows = [{
    "contract_id": r["contract_id"], "summary": r.get("summary", ""),
    "termination_clause": r.get("termination_clause", {}).get("clause_text", ""),
    "confidentiality_clause": r.get("confidentiality_clause", {}).get("clause_text", ""),
    "liability_clause": r.get("liability_clause", {}).get("clause_text", ""),
} for r in results]
pd.DataFrame(flat_rows).to_csv(csv_path, index=False)   # the assignment's required column format
```

That's the entire pipeline. Every one of the 50 contracts goes through steps
3 → 4 → 5 → 6 → 7 → 8 → 8b, and the loop in `run_pipeline` just repeats that
per contract, catching and logging any per-contract failure so one bad
document doesn't stop the batch.

---

## Bonus feature A: semantic search (`src/semantic_search.py`)

Runs *after* the main pipeline, over `output/results.json`:

```python
class ClauseSearchIndex:
    @classmethod
    def from_results_json(cls, results_path):
        idx = cls()
        for record in json.loads(Path(results_path).read_text()):
            for clause_type in (...):
                text = record.get(clause_type, {}).get("clause_text")
                if text:
                    idx._entries.append({"contract_id": ..., "clause_type": ..., "clause_text": text})
        idx._build()   # embed every extracted clause + build a FAISS index
        return idx

    def search(self, query, k=5):
        query_vec = embed_texts([query]).astype("float32")
        scores, ids = self._index.search(query_vec, k)   # FAISS nearest-neighbor search
        return [SearchResult(...) for score, idx in zip(scores[0], ids[0])]
```

This is a *second*, separate index from the per-contract `ChunkRetriever` in
Step 7 — that one retrieves raw contract chunks during extraction; this one
searches the already-extracted clauses across your whole processed corpus.

## Bonus feature B: model comparison (`compare_models.py`)

Runs the exact same `extract_all_clauses()` from Step 8, twice, with two
different providers, on the same small sample, and diffs the results:

```python
rows_a = run_one_provider(args.provider_a, args.model_a, contracts)   # e.g. groq/llama-3.3-70b
rows_b = run_one_provider(args.provider_b, args.model_b, contracts)   # e.g. openai/gpt-4o-mini

for ra, rb in zip(rows_a, rows_b):
    entry[f"{ct}_agree_found"] = ra[f"{ct}_found"] == rb[f"{ct}_found"]
    entry[f"{ct}_text_similarity"] = text_similarity(ra[f"{ct}_text"], rb[f"{ct}_text"])
```

`text_similarity` is `difflib.SequenceMatcher` — a simple ratio of matching
character sequences between the two providers' extracted clause text, used
purely as a rough agreement signal, not a rigorous eval metric.

## Bonus feature C: Streamlit demo (`app.py`)

Same functions, wired to a UI instead of a CLI loop. Upload button →
`_extract_pdf_text()` (Step 3) → `normalize_text()` (Step 5) →
`chunk_text()` (Step 6) → `ChunkRetriever()` (Step 7) →
`extract_all_clauses()` + `summarize_contract()` (Step 8) → rendered in
Streamlit's `st.expander()` widgets instead of written to CSV. It's the
identical pipeline, just triggered by a button click instead of a `for`
loop over 50 files.

---

## The one-sentence version

**PDF → text → clean text → overlapping chunks → embed chunks → retrieve
the 4 most relevant chunks per clause type → ask the LLM to extract
structured JSON from just those chunks → separately map-reduce-summarize
all chunks → write CSV/JSON → cache the result so it's free to skip next
time.**

Every stage above is a separate, independently testable function — that's
what the 21 unit tests in `tests/` exercise (with a fake retriever and a
fake LLM provider standing in for Steps 2 and 7, so the tests run in
milliseconds with no API key or network needed).
