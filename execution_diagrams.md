# Execution Diagrams

Companion to `docs/execution_walkthrough.md`. These render natively on GitHub
(no image export needed — just open this file in the repo).

## 1. Sequence diagram — who calls whom, in what order

This is the exact call order for processing one contract (the loop in
`run_pipeline()` repeats this per contract).

```mermaid
sequenceDiagram
    participant CLI as main.py
    participant PL as pipeline.py
    participant LLM as llm_provider.py
    participant DL as data_loader.py
    participant PP as preprocessor.py
    participant CH as chunker.py
    participant RT as retriever.py
    participant CE as clause_extractor.py
    participant SM as summarizer.py
    participant API as Groq / OpenAI / Anthropic

    CLI->>PL: run_pipeline(data_dir, limit=50)
    PL->>LLM: get_provider()
    LLM-->>PL: provider instance
    PL->>DL: load_contracts(data_dir, limit)
    DL-->>PL: yields Contract objects (real PDF text)

    loop for each contract
        alt cached result exists
            PL->>PL: read .cache/contract_id.json
        else no cache
            PL->>PP: normalize_text(raw_text)
            PP-->>PL: normalized text
            PL->>CH: chunk_text(normalized)
            CH-->>PL: list of Chunks
            PL->>RT: ChunkRetriever(chunks)
            RT->>RT: embed_texts(chunk texts)

            PL->>CE: extract_all_clauses(provider, retriever)
            loop termination / confidentiality / liability
                CE->>RT: top_k(query, k=4)
                RT-->>CE: 4 most relevant chunks
                CE->>API: complete(few-shot prompt + chunks)
                API-->>CE: JSON clause extraction
            end
            CE-->>PL: clauses dict

            PL->>SM: summarize_contract(provider, chunks)
            loop MAP: each chunk
                SM->>API: complete(map prompt)
                API-->>SM: bullet notes
            end
            SM->>API: complete(reduce prompt, all notes)
            API-->>SM: 100-150 word summary
            SM-->>PL: summary text

            PL->>PL: write .cache/contract_id.json
        end
    end

    PL->>PL: write output/results.csv + results.json
    PL-->>CLI: DataFrame
```

## 2. Decision flowchart — cache and error-handling logic

This is the branch logic inside the per-contract loop that the sequence
diagram glosses over.

```mermaid
flowchart TD
    A[New contract from load_contracts] --> B{Cached result<br/>exists?}
    B -->|Yes| C[Read .cache/contract_id.json]
    B -->|No| D[normalize_text]
    D --> E[chunk_text]
    E --> F[ChunkRetriever - embed chunks]
    F --> G[extract_all_clauses]
    G --> H[summarize_contract]
    H --> I{Any step<br/>raised an error?}
    I -->|Yes| J[Log error, record<br/>contract_id + error message]
    I -->|No| K[Assemble result record]
    K --> L[Write .cache/contract_id.json]
    C --> M[Append to results list]
    L --> M
    J --> M
    M --> N{More contracts<br/>in batch?}
    N -->|Yes| A
    N -->|No| O[Write results.csv + results.json]
```

The key thing this flowchart captures that the sequence diagram doesn't:
**one failed contract never stops the batch.** Step `I` catches any
exception from normalization, chunking, retrieval, extraction, or
summarization and routes to `J` instead of crashing — the contract just
shows up in the output with an `error` column filled in instead of a
summary.

## Why two diagrams instead of one

A single diagram trying to show both "call order" and "branch logic" gets
cluttered fast. Splitting them keeps each one answerable in one glance:

- **Sequence diagram** answers: *"what talks to what, and in what order?"*
  — the one to use when explaining the architecture.
- **Flowchart** answers: *"what happens if X fails, or Y is already cached?"*
  — the one to use when debugging or explaining resilience/reproducibility.
