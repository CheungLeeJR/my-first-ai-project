import json
from pathlib import Path
from src.database import Database
from src.models import ChatMessage,DocumentRecord,TokenUsage
class Repository:
 def __init__(self,db=None):self.db=db or Database()
 def documents(self):
  with self.db.connect() as c:return [DocumentRecord(**dict(r)) for r in c.execute("SELECT * FROM documents ORDER BY uploaded_at")]
 def has_document(self,did):
  with self.db.connect() as c:return c.execute("SELECT 1 FROM documents WHERE document_id=?",(did,)).fetchone() is not None
 def save_document(self,r,chunks,embeddings):
  with self.db.connect() as c:
   c.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?)",(r.document_id,r.file_name,r.file_size,r.page_count,r.chunk_count,r.ocr_pages,r.uploaded_at))
   c.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?)",[(d.metadata['chunk_id'],r.document_id,r.file_name,d.metadata['page_number'],d.page_content,e.astype('float32').tobytes(),len(e)) for d,e in zip(chunks,embeddings)])
 def delete_document(self,did):
  with self.db.connect() as c:
   c.execute("DELETE FROM chunks WHERE document_id=?",(did,)); c.execute("DELETE FROM documents WHERE document_id=?",(did,))
 def chunk_rows(self):
  with self.db.connect() as c:return list(c.execute("SELECT * FROM chunks ORDER BY chunk_id"))
 def save_message(self,m):
  with self.db.connect() as c:c.execute("INSERT INTO messages(role,content,citations_json,metrics_json) VALUES(?,?,?,?)",(m.role,m.content,json.dumps([x.model_dump() for x in m.citations]),json.dumps(m.metrics)))
 def messages(self,limit=None):
  q="SELECT * FROM messages ORDER BY id"; params=()
  if limit:q="SELECT * FROM (SELECT * FROM messages ORDER BY id DESC LIMIT ?) ORDER BY id";params=(limit,)
  with self.db.connect() as c:return [ChatMessage(role=r['role'],content=r['content'],citations=json.loads(r['citations_json']),metrics=json.loads(r['metrics_json'])) for r in c.execute(q,params)]
 def clear_messages(self):
  with self.db.connect() as c:c.execute("DELETE FROM messages")
 def save_usage(self,q,model,u,latency,k):
  with self.db.connect() as c:c.execute("INSERT INTO usage(question,model,input_tokens,output_tokens,total_tokens,latency_ms,retrieved_chunks) VALUES(?,?,?,?,?,?,?)",(q,model,u.input_tokens,u.output_tokens,u.total_tokens,latency,k))
 def usage_rows(self):
  with self.db.connect() as c:return [dict(r) for r in c.execute("SELECT * FROM usage ORDER BY id DESC")]
