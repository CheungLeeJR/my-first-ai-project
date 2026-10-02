import hashlib, shutil
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_OVERLAP,CHUNK_SIZE,MAX_FILE_MB,MIN_TEXT_PER_PAGE,OCR_LANGUAGES,UPLOAD_DIR
from src.errors import DocumentError,OCRUnavailableError
from src.models import DocumentRecord
from src.logging_setup import get_logger
log=get_logger(__name__)
class PDFService:
 @staticmethod
 def document_id(data:bytes)->str:return hashlib.sha256(data).hexdigest()
 def parse(self,file_name:str,data:bytes)->tuple[DocumentRecord,list[Document]]:
  if not file_name.lower().endswith('.pdf'):raise DocumentError("Only PDF files are supported.")
  if not data:raise DocumentError("The uploaded file is empty.")
  if len(data)>MAX_FILE_MB*1024*1024:raise DocumentError(f"File exceeds {MAX_FILE_MB} MB.")
  did=self.document_id(data); path=UPLOAD_DIR/f"{did}.pdf"; path.write_bytes(data)
  try:
   pdf=pymupdf.open(path)
   if pdf.needs_pass:raise DocumentError("Encrypted PDFs are not supported.")
   pages=[]; ocr_count=0
   for i,page in enumerate(pdf):
    text=page.get_text("text").strip()
    if len(text)<MIN_TEXT_PER_PAGE:
     try:
      tp=page.get_textpage_ocr(language=OCR_LANGUAGES,dpi=200,full=True)
      text=page.get_text("text",textpage=tp).strip(); ocr_count+=1
     except Exception as exc:
      log.warning("OCR unavailable document=%s page=%s error=%s",did,i+1,type(exc).__name__)
    if text:
     pages.append(Document(page_content=text,metadata={"document_id":did,"file_name":file_name,"page_number":i+1}))
   page_count=len(pdf); pdf.close()
   if not pages:raise OCRUnavailableError("No text could be extracted. Install Tesseract with the required language pack for scanned PDFs.")
   chunks=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP).split_documents(pages)
   for i,c in enumerate(chunks):c.metadata["chunk_id"]=f"{did}:{i}"
   return DocumentRecord(document_id=did,file_name=file_name,file_size=len(data),page_count=page_count,chunk_count=len(chunks),ocr_pages=ocr_count),chunks
  except DocumentError:
   path.unlink(missing_ok=True); raise
  except Exception as exc:
   path.unlink(missing_ok=True); log.exception("PDF parse failure document=%s",did); raise DocumentError(f"Could not parse PDF: {exc}") from exc
