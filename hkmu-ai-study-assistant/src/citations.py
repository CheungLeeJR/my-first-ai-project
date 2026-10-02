import re
from src.models import Citation
STOP={"the","and","for","that","with","this","from","are","was","were","have","has","not","but","what","how","why","which","into","your","about"}
class CitationService:
 def build(self,results):
  out=[]
  for doc,score in results:
   out.append(Citation(citation_id=len(out)+1,document_id=doc.metadata['document_id'],file_name=doc.metadata['file_name'],page_number=doc.metadata['page_number'],chunk_id=doc.metadata['chunk_id'],excerpt=doc.page_content[:650].strip(),score=round(score,3)))
  return out
 def sanitize(self,answer,count):
  invalid={int(x) for x in re.findall(r"\[(\d+)\]",answer) if int(x)<1 or int(x)>count}
  for i in invalid:answer=answer.replace(f"[{i}]","")
  return answer,sorted(invalid)
 def coverage(self,answer):
  sentences=[s.strip() for s in re.split(r'(?<=[.!?。！？])\s*',answer) if len(s.strip())>20]
  if not sentences:return 1.0
  cited=sum(bool(re.search(r"\[\d+\]",s)) for s in sentences)
  return round(cited/len(sentences),2)
