import pytest
pytest.importorskip("langchain_core")
pytest.importorskip("faiss")

from pathlib import Path
import pymupdf
from src.database import Database
from src.repository import Repository
from src.rag_engine import RAGEngine
from tests.fakes import FakeEmbeddings,FakeLLM
class Upload:
 def __init__(self,name,data):self.name=name;self._data=data
 def getvalue(self):return self._data
def pdf_bytes(text):
 d=pymupdf.open();p=d.new_page();p.insert_text((72,72),text);b=d.tobytes();d.close();return b
def engine(tmp_path,monkeypatch):
 import src.config as cfg,src.pdf_service as ps,src.vector_index as vi
 up=tmp_path/'uploads';up.mkdir();idx=tmp_path/'x.faiss';monkeypatch.setattr(ps,'UPLOAD_DIR',up);monkeypatch.setattr(vi,'INDEX_PATH',idx)
 repo=Repository(Database(tmp_path/'test.db'));return RAGEngine(FakeEmbeddings(),FakeLLM(),repo)
def test_end_to_end_persistence_delete(tmp_path,monkeypatch):
 e=engine(tmp_path,monkeypatch);added,errors=e.add_uploads([Upload('a.pdf',pdf_bytes('Supervised learning uses labelled data for classification.'))]);assert len(added)==1 and not errors
 r=e.ask('What does supervised learning use?');assert r.citations[0].page_number==1 and r.usage.total_tokens==19
 e2=RAGEngine(FakeEmbeddings(),FakeLLM(),e.repo);assert len(e2.documents)==1 and e2.index.index.ntotal>0
 e2.remove(added[0].document_id);assert not e2.documents and e2.index.index is None
def test_duplicate(tmp_path,monkeypatch):
 e=engine(tmp_path,monkeypatch);u=Upload('a.pdf',pdf_bytes('Enough text content for parsing and indexing.'));e.add_uploads([u]);a,errs=e.add_uploads([u]);assert not a and 'duplicate' in errs[0]
def test_selective_rewrite(tmp_path,monkeypatch):
 e=engine(tmp_path,monkeypatch);e.repo.save_message(__import__('src.models',fromlist=['ChatMessage']).ChatMessage(role='user',content='Explain supervised learning'))
 assert e._query('What are its limits?')[1] is True
 assert e._query('Define regression')[1] is False
