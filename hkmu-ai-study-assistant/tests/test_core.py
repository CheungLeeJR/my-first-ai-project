from __future__ import annotations

import numpy as np

from src.citations import CitationService
from src.database import Database
from src.models import ChatMessage, DocumentRecord
from src.repository import Repository


class DummyDocument:
    def __init__(self, text: str, metadata: dict):
        self.page_content = text
        self.metadata = metadata


def test_citation_sanitization_and_coverage():
    service = CitationService()
    answer = "The model uses labelled data [1]. It predicts classes [7]."
    cleaned, invalid = service.sanitize(answer, count=2)
    assert invalid == [7]
    assert "[7]" not in cleaned
    assert service.coverage(cleaned) == 0.5


def test_repository_document_roundtrip_and_cascade_delete(tmp_path):
    repo = Repository(Database(tmp_path / "app.db"))
    record = DocumentRecord(
        document_id="doc-1",
        file_name="sample.pdf",
        file_size=123,
        page_count=1,
        chunk_count=1,
    )
    doc = DummyDocument(
        "A short chunk",
        {
            "chunk_id": "doc-1:0",
            "document_id": "doc-1",
            "file_name": "sample.pdf",
            "page_number": 1,
        },
    )
    repo.save_document(record, [doc], np.array([[0.1, 0.2, 0.3]], dtype="float32"))
    assert repo.has_document("doc-1")
    assert len(repo.documents()) == 1
    assert len(repo.chunk_rows()) == 1

    repo.delete_document("doc-1")
    assert not repo.has_document("doc-1")
    assert repo.chunk_rows() == []


def test_message_persistence(tmp_path):
    repo = Repository(Database(tmp_path / "chat.db"))
    repo.save_message(ChatMessage(role="user", content="hello"))
    repo.save_message(ChatMessage(role="assistant", content="hi"))
    assert [m.role for m in repo.messages()] == ["user", "assistant"]
    assert len(repo.messages(limit=1)) == 1
    repo.clear_messages()
    assert repo.messages() == []
