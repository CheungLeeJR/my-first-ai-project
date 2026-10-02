from datetime import datetime,timezone
from pydantic import BaseModel,Field
class DocumentRecord(BaseModel):
 document_id:str; file_name:str; file_size:int; page_count:int; chunk_count:int; ocr_pages:int=0; uploaded_at:str=Field(default_factory=lambda:datetime.now(timezone.utc).isoformat())
class Citation(BaseModel):
 citation_id:int; document_id:str; file_name:str; page_number:int; chunk_id:str; excerpt:str; score:float=0
class TokenUsage(BaseModel): input_tokens:int=0; output_tokens:int=0; total_tokens:int=0
class ChatMessage(BaseModel): role:str; content:str; citations:list[Citation]=Field(default_factory=list); metrics:dict=Field(default_factory=dict)
class RAGResponse(BaseModel):
 answer:str; citations:list[Citation]; retrieval_query:str; usage:TokenUsage; latency_ms:int; retrieved_chunks:int; citation_coverage:float
