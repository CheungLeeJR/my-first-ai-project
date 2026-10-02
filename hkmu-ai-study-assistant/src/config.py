from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
BASE_DIR=Path(__file__).resolve().parent.parent
DATA_DIR=BASE_DIR/"data"; UPLOAD_DIR=DATA_DIR/"uploads"; INDEX_DIR=DATA_DIR/"indexes"; LOG_DIR=BASE_DIR/"logs"
DB_PATH=DATA_DIR/"app.db"; INDEX_PATH=INDEX_DIR/"course_index.faiss"
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY","")
CHAT_MODEL=os.getenv("CHAT_MODEL","gpt-4o-mini")
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL","text-embedding-3-small")
OCR_LANGUAGES=os.getenv("OCR_LANGUAGES","eng")
CHUNK_SIZE=1000; CHUNK_OVERLAP=180; RETRIEVAL_K=5; MEMORY_MESSAGES=6
MAX_FILE_MB=25; MAX_FILES=10; MIN_TEXT_PER_PAGE=20
for p in (DATA_DIR,UPLOAD_DIR,INDEX_DIR,LOG_DIR): p.mkdir(parents=True,exist_ok=True)
