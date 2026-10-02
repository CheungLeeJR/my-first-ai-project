import re,time
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from src.citations import CitationService
from src.config import CHAT_MODEL,EMBEDDING_MODEL,MEMORY_MESSAGES,OPENAI_API_KEY,MAX_FILES,UPLOAD_DIR
from src.errors import ConfigurationError,DocumentError,KnowledgeBaseError,ProviderError
from src.logging_setup import get_logger
from src.models import ChatMessage,RAGResponse,TokenUsage
from src.pdf_service import PDFService
from src.repository import Repository
from src.vector_index import VectorIndex
log=get_logger(__name__)
SYSTEM="""You are HKMU AI Study Assistant. Use only the supplied course context. Every factual sentence must end with one or more valid citations such as [1] or [1][2]. If evidence is insufficient, say so. Never invent citation numbers. Answer in the user's language."""
DEPENDENT=re.compile(r"\b(it|its|they|them|those|this|that|he|she|former|latter)\b|它|这个|这些|上述|前者|后者",re.I)
class RAGEngine:
 def __init__(self,embeddings=None,llm=None,repository=None):
  if (embeddings is None or llm is None) and not OPENAI_API_KEY:raise ConfigurationError("OPENAI_API_KEY is missing. Copy .env.example to .env.")
  self.repo=repository or Repository();self.pdf=PDFService();self.embeddings=embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL,api_key=OPENAI_API_KEY);self.llm=llm or ChatOpenAI(model=CHAT_MODEL,temperature=0,api_key=OPENAI_API_KEY,max_retries=2,timeout=60);self.index=VectorIndex(self.repo,self.embeddings);self.citations=CitationService()
 @property
 def documents(self):return self.repo.documents()
 @property
 def messages(self):return self.repo.messages()
 def add_uploads(self,uploads):
  added=[];errors=[]
  if len(self.documents)+len(uploads)>MAX_FILES:return [],[f"Knowledge base limit is {MAX_FILES} PDFs."]
  for f in uploads:
   did=self.pdf.document_id(f.getvalue())
   if self.repo.has_document(did):errors.append(f"{f.name}: duplicate document");continue
   try:
    record,chunks=self.pdf.parse(f.name,f.getvalue());self.index.add(record,chunks);added.append(record);log.info("document added id=%s pages=%s",did,record.page_count)
   except Exception as exc:errors.append(f"{f.name}: {exc}");log.exception("document add failed id=%s",did)
  return added,errors
 def remove(self,did):
  self.index.delete_document(did);(UPLOAD_DIR/f"{did}.pdf").unlink(missing_ok=True);log.info("document removed id=%s",did)
 def _query(self,q):
  recent=self.repo.messages(MEMORY_MESSAGES)
  if not recent or not DEPENDENT.search(q):return q,False
  history="\n".join(f"{m.role}: {m.content}" for m in recent)
  prompt=f"Rewrite only the latest question as a standalone search query. Return only the query.\nHistory:\n{history}\nQuestion: {q}"
  try:
   r=self.llm.invoke([HumanMessage(content=prompt)])
   return str(r.content).strip() or q,True
  except Exception:return q,False
 def ask(self,q):
  if not q.strip():raise KnowledgeBaseError("Question cannot be empty.")
  start=time.perf_counter();query,rewritten=self._query(q);results=self.index.search(query,5);citations=self.citations.build(results)
  blocks=[f"[{c.citation_id}] {c.file_name}, PDF page {c.page_number}\n{d.page_content}" for c,(d,s) in zip(citations,results)]
  context="\n\n".join(blocks)
  try:
   resp=self.llm.invoke([SystemMessage(content=SYSTEM),HumanMessage(content=f"Context:\n{context}\n\nQuestion: {q}")])
  except Exception as exc:
   log.exception("LLM failure");raise ProviderError(f"Language model request failed: {exc}") from exc
  answer,invalid=self.citations.sanitize(str(resp.content),len(citations));coverage=self.citations.coverage(answer)
  usage_raw=getattr(resp,'usage_metadata',None) or {};usage=TokenUsage(input_tokens=int(usage_raw.get('input_tokens',0)),output_tokens=int(usage_raw.get('output_tokens',0)),total_tokens=int(usage_raw.get('total_tokens',0)))
  latency=round((time.perf_counter()-start)*1000);metrics={"tokens":usage.total_tokens,"latency_ms":latency,"retrieved":len(results),"rewritten":rewritten,"citation_coverage":coverage}
  self.repo.save_message(ChatMessage(role="user",content=q));self.repo.save_message(ChatMessage(role="assistant",content=answer,citations=citations,metrics=metrics));self.repo.save_usage(q,CHAT_MODEL,usage,latency,len(results))
  return RAGResponse(answer=answer,citations=citations,retrieval_query=query,usage=usage,latency_ms=latency,retrieved_chunks=len(results),citation_coverage=coverage)
 def clear_conversation(self):self.repo.clear_messages()
