# Verification Notes

## Local checks performed for this portfolio revision

The dependency-light core test suite passes in the packaging environment:

- citation marker sanitization and coverage calculation;
- SQLite document/chunk persistence and cascade deletion;
- chat-message persistence;
- offline retrieval/citation metric calculations.

The full end-to-end RAG integration tests remain in `tests/test_system.py`. They exercise PDF parsing, FAISS persistence, duplicate detection, retrieval, citations, and selective query rewriting. Those tests automatically skip when optional runtime packages such as LangChain/FAISS are absent and run normally in GitHub Actions after `requirements.txt` is installed.

## Original project verification evidence

The uploaded project artifact reported successful checks for:

- Python compilation for `app.py`, `src/`, and `tests/`;
- 3 offline integration tests;
- text-PDF parsing with PyMuPDF;
- FAISS indexing/search/persistence/deletion;
- SQLite persistence;
- SHA-256 duplicate detection;
- selective conversation query rewriting;
- page-level citation metadata and token accounting;
- image-only PDF OCR using Tesseract;
- Streamlit health endpoint.

## Security/privacy checks

- `.env`, runtime databases, FAISS indexes, uploaded PDFs, logs, caches, and bytecode are excluded by `.gitignore`.
- No API key is included in the repository bundle.
- SQLite foreign-key enforcement is explicitly enabled on each connection.

## Live-provider boundary

The deterministic tests use fake LLM and embedding providers. Live OpenAI behavior still depends on the user's API key, quota, selected model availability, and network access.
