from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def evaluate_record(record: dict) -> dict[str, float]:
    relevant = list(dict.fromkeys(record.get("relevant_chunk_ids", []) or []))
    retrieved = list(record.get("retrieved_chunk_ids", []) or [])
    relevant_set = set(relevant)
    retrieved_set = set(retrieved)
    recall = _safe_div(len(relevant_set & retrieved_set), len(relevant_set))
    hit = float(bool(relevant_set & retrieved_set))
    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(retrieved, 1):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    expected = set(record.get("expected_citation_ids", []) or [])
    predicted = set(record.get("predicted_citation_ids", []) or [])
    overlap = len(expected & predicted)
    citation_precision = _safe_div(overlap, len(predicted))
    citation_recall = _safe_div(overlap, len(expected))
    citation_f1 = _safe_div(2 * citation_precision * citation_recall, citation_precision + citation_recall)
    return {
        "retrieval_recall": recall,
        "hit_rate": hit,
        "mrr": reciprocal_rank,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_f1": citation_f1,
    }


def evaluate_records(records: list[dict]) -> dict[str, float]:
    if not records:
        raise ValueError("evaluation set is empty")
    rows = [evaluate_record(record) for record in records]
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate precomputed RAG retrieval/citation outputs.")
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    result = evaluate_records(load_jsonl(args.jsonl))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
