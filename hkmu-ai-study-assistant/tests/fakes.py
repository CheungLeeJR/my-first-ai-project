import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
class FakeEmbeddings(Embeddings):
 def _v(self,t):
  v=np.array([len(t)%11+1,sum(map(ord,t))%13+1,1.0],dtype=float);return v.tolist()
 def embed_documents(self,texts):return [self._v(t) for t in texts]
 def embed_query(self,text):return self._v(text)
class FakeLLM:
 def invoke(self,msgs):
  x=str(msgs[-1].content)
  if 'Rewrite only' in x:return AIMessage(content='standalone supervised learning query')
  return AIMessage(content='Supervised learning uses labelled data [1].',usage_metadata={'input_tokens':12,'output_tokens':7,'total_tokens':19})
