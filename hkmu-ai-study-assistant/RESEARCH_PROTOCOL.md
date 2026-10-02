# Research Protocol — Multi-document RAG Study Assistant

## Research objective

Study whether a persistent multi-document RAG pipeline can provide traceable answers over university course PDFs while remaining robust to document lifecycle operations and scanned-PDF inputs.

## Research questions

- **RQ1 — Retrieval:** how often does the retriever return at least one relevant chunk within top-k?
- **RQ2 — Citation traceability:** when an answer cites evidence, how precise/complete are the cited document-page references relative to labelled references?
- **RQ3 — Robustness:** how does performance differ between text-native PDFs and OCR-dependent pages?
- **RQ4 — Persistence correctness:** do indexing, reload, duplicate detection, and deletion preserve consistent retrieval state across sessions?

## Hypotheses

- H1: persistent indexing can preserve retrieval behavior after process restart.
- H2: explicit page metadata plus citation sanitization improves traceability over uncited generation.
- H3: OCR fallback enables otherwise unreachable scanned content, but with lower retrieval quality than text-native PDFs.

These are **testable hypotheses**, not current measured conclusions.

## Evaluation dataset design

Create a small, legally shareable benchmark from public or self-authored PDFs. Each evaluation item should contain:

- query;
- relevant chunk IDs and/or document-page references;
- content type (`text` or `ocr`);
- retrieved chunk IDs;
- predicted citation IDs;
- labelled citation IDs.

Keep training/course material out of the public benchmark unless redistribution is permitted.

## Primary metrics

- Retrieval Recall@k
- Hit Rate@k
- Mean Reciprocal Rank (MRR)
- Citation precision / recall / F1
- Persistence regression pass rate

## Secondary measurements

- latency by stage (parse/embed/retrieve/generate);
- token usage and estimated request cost;
- OCR failure rate;
- duplicate-detection correctness;
- deletion/index-consistency failures.

## Ablations

1. query rewriting on vs off;
2. different retrieval `k` values;
3. text-only vs OCR pages;
4. citation sanitizer on vs raw model citation markers;
5. optional future reranker vs base FAISS retrieval.

## Evidence boundary

The repository currently validates core system behavior with deterministic tests. It does **not** yet claim a measured retrieval benchmark or answer-faithfulness score. Those should only be reported after the labelled benchmark is created and evaluated.
