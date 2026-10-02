import faiss,numpy as np
from langchain_core.documents import Document
from src.config import INDEX_PATH
from src.errors import KnowledgeBaseError
class VectorIndex:
 def __init__(self,repository,embeddings):self.repo=repository;self.embeddings=embeddings;self.index=None;self.rows=[];self.load()
 def load(self):
  self.rows=self.repo.chunk_rows()
  if not self.rows:self.index=None;INDEX_PATH.unlink(missing_ok=True);return
  dim=self.rows[0]['dimension']
  if INDEX_PATH.exists():
   try:self.index=faiss.read_index(str(INDEX_PATH));return
   except Exception:pass
  self.rebuild()
 def rebuild(self):
  self.rows=self.repo.chunk_rows()
  if not self.rows:self.index=None;INDEX_PATH.unlink(missing_ok=True);return
  vectors=np.vstack([np.frombuffer(r['embedding'],dtype='float32') for r in self.rows]).astype('float32')
  faiss.normalize_L2(vectors); self.index=faiss.IndexFlatIP(vectors.shape[1]);self.index.add(vectors);faiss.write_index(self.index,str(INDEX_PATH))
 def add(self,record,chunks):
  try:v=np.asarray(self.embeddings.embed_documents([x.page_content for x in chunks]),dtype='float32')
  except Exception as exc:raise KnowledgeBaseError(f"Embedding request failed: {exc}") from exc
  self.repo.save_document(record,chunks,v);self.rebuild()
 def delete_document(self,did):self.repo.delete_document(did);self.rebuild()
 def search(self,query,k):
  if self.index is None:raise KnowledgeBaseError("Add at least one PDF first.")
  q=np.asarray([self.embeddings.embed_query(query)],dtype='float32');faiss.normalize_L2(q)
  scores,ids=self.index.search(q,min(k,len(self.rows)));out=[]
  for score,i in zip(scores[0],ids[0]):
   if i<0:continue
   r=self.rows[i];out.append((Document(page_content=r['content'],metadata={"chunk_id":r['chunk_id'],"document_id":r['document_id'],"file_name":r['file_name'],"page_number":r['page_number']}),float(score)))
  return out
