import streamlit as st
from src.errors import StudyAssistantError
from src.rag_engine import RAGEngine
st.set_page_config(page_title="HKMU AI Study Assistant",page_icon="📚",layout="wide")
st.markdown("""<style>.block-container{max-width:1200px;padding-top:2rem}.stChatMessage{border:1px solid #e2e8f0;border-radius:14px;padding:.6rem}.hero{padding:1.2rem;border-radius:18px;background:linear-gradient(120deg,#1d4ed8,#4f46e5);color:white;margin-bottom:1rem}</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📚 HKMU AI Study Assistant</h1><p>Persistent multi-PDF conversational RAG with OCR, citations and observability</p></div>',unsafe_allow_html=True)
if 'engine' not in st.session_state:
 try:st.session_state.engine=RAGEngine()
 except StudyAssistantError as e:st.error(str(e));st.code("cp .env.example .env\n# Add OPENAI_API_KEY, then restart");st.stop()
e=st.session_state.engine
chat,pdfs,usage=st.tabs(["💬 Chat","📄 PDF Manager","📊 Observability"])
with pdfs:
 uploads=st.file_uploader("Upload course PDFs",type=['pdf'],accept_multiple_files=True)
 if st.button("Process PDFs",type="primary",disabled=not uploads):
  with st.status("Building persistent knowledge base...",expanded=True) as s:
   added,errors=e.add_uploads(uploads)
   for r in added:st.write(f"✅ {r.file_name}: {r.page_count} pages, {r.chunk_count} chunks, {r.ocr_pages} OCR pages")
   for x in errors:st.error(x)
   s.update(label="Finished",state="complete")
 st.subheader("Managed documents")
 if not e.documents:st.info("Upload one or more PDFs to begin.")
 for r in e.documents:
  a,b,c,d,act=st.columns([4,1,1,1,1]);a.markdown(f"**{r.file_name}**");b.write(f"{r.page_count} pages");c.write(f"{r.chunk_count} chunks");d.write(f"OCR {r.ocr_pages}")
  if act.button("Delete",key=r.document_id):e.remove(r.document_id);st.rerun()
with chat:
 docs=e.documents;a,b,c,d=st.columns(4);a.metric("Documents",len(docs));b.metric("Pages",sum(x.page_count for x in docs));c.metric("Chunks",sum(x.chunk_count for x in docs));d.metric("Messages",len(e.messages))
 if st.button("Clear conversation"):e.clear_conversation();st.rerun()
 if not docs:st.info("Go to PDF Manager and upload course material first.")
 for m in e.messages:
  with st.chat_message(m.role):
   st.markdown(m.content)
   if m.citations:
    with st.expander(f"Sources ({len(m.citations)})"):
     for c in m.citations:st.markdown(f"**[{c.citation_id}] {c.file_name}, PDF page {c.page_number}** · similarity {c.score}");st.caption(c.excerpt)
   if m.metrics:st.caption(f"{m.metrics.get('tokens',0)} tokens · {m.metrics.get('latency_ms',0)/1000:.2f}s · {m.metrics.get('retrieved',0)} chunks · citation coverage {m.metrics.get('citation_coverage',0):.0%}")
 q=st.chat_input("Ask about your course materials...",disabled=not docs)
 if q:
  try:
   with st.spinner("Retrieving and validating evidence..."):e.ask(q)
   st.rerun()
  except StudyAssistantError as x:st.error(str(x))
with usage:
 rows=e.repo.usage_rows();total=sum(x['total_tokens'] for x in rows);avg=sum(x['latency_ms'] for x in rows)/len(rows) if rows else 0
 a,b,c,d=st.columns(4);a.metric("Questions",len(rows));b.metric("Total tokens",total);c.metric("Average latency",f"{avg/1000:.2f}s");d.metric("Index persistence","Active")
 if rows:st.dataframe(rows,use_container_width=True,hide_index=True)
 st.caption("Application logs use rotation and exclude API keys, prompts and PDF content.")
