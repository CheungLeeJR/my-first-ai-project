import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_retrieval.py"
spec = importlib.util.spec_from_file_location("evaluate_retrieval", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_retrieval_and_citation_metrics():
    row = module.evaluate_record({
        "relevant_chunk_ids": ["c1", "c3"],
        "retrieved_chunk_ids": ["c2", "c1", "c9"],
        "expected_citation_ids": ["doc:p1", "doc:p3"],
        "predicted_citation_ids": ["doc:p1", "doc:p8"],
    })
    assert row["retrieval_recall"] == 0.5
    assert row["hit_rate"] == 1.0
    assert row["mrr"] == 0.5
    assert row["citation_precision"] == 0.5
    assert row["citation_recall"] == 0.5
    assert row["citation_f1"] == 0.5


def test_aggregate_requires_non_empty_records():
    try:
        module.evaluate_records([])
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")
