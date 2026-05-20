"""
ScholarSync Backend - Complete RAG Pipeline
Pure Python: FastAPI + FAISS + sentence-transformers + OpenAI
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os, io, re, hashlib, json
from dataclasses import dataclass, asdict
from typing import List, Optional
from dotenv import load_dotenv

# Document parsing
import pdfplumber  # PDF text extraction
from docx import Document as DocxDocument  # DOCX parsing

import mysql.connector
from mysql.connector import Error
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec

# Embeddings & vector search

# LLM
from openai import OpenAI

# Load local environment variables if testing locally
load_dotenv()

# ─── MySQL Configuration ────────────────────────────────────
# Credentials will be loaded from environment variables on Vercel
MYSQL_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "scholarsync"),
    "port": int(os.getenv("DB_PORT", 4000)),
    "ssl_verify_cert": True
}

# ─── Pinecone Configuration ─────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ─── Google AI Configuration ────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
google_ai_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

# ─── Configuration ──────────────────────────────────────────
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 100   # overlap between chunks
TOP_K = 5             # number of chunks to retrieve
EMBED_MODEL_NAME = "models/text-embedding-004"
VECTOR_DIMENSION = 768 # Dimension for text-embedding-004

# ─── AI Model Configuration ─────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ai_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_API_KEY else None

# ─── Global State ───────────────────────────────────────────
pc = None       # Pinecone client instance

# ─── Data Models ────────────────────────────────────────────
@dataclass
class TextChunk:
    text: str
    page: int
    start_idx: int

@dataclass
class DocumentData:
    id: str
    name: str
    type: str
    page_count: int
    word_count: int
    pages: dict  # page_num -> text

class AskRequest(BaseModel):
    doc_id: str
    query: str

class AskResponse(BaseModel):
    answer: str
    citations: List[dict]
    highlighted_phrases: List[str]

# ─── State Management ───────────────────────────────────────
def init_db():
    """Initialize MySQL Database and Tables if they don't exist."""
    try:
        # First, connect without the database name to create it if it doesn't exist
        admin_config = MYSQL_CONFIG.copy()
        db_name = admin_config.pop("database", "scholarsync")
        
        conn = mysql.connector.connect(**admin_config)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        cursor.close()
        conn.close()

        # Now connect to the specific database to create tables
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255), type VARCHAR(50),
                page_count INT, word_count INT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_pages (
                doc_id VARCHAR(255), page_number INT, text LONGTEXT,
                PRIMARY KEY (doc_id, page_number),
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INT AUTO_INCREMENT PRIMARY KEY,
                doc_id VARCHAR(255), role VARCHAR(50),
                content LONGTEXT, citations JSON, highlighted_phrases JSON,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"Error initializing MySQL: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup logic."""
    global pc
    print("Application startup: Initializing database and clients...")
    init_db()
    if PINECONE_API_KEY and GOOGLE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Application startup complete.")
    yield
    # Code below yield runs on shutdown, not typically used in serverless
    print("Application shutting down.")

app = FastAPI(title="ScholarSync RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Document Parsing ───────────────────────────────────────
def parse_pdf(file_bytes: bytes) -> dict:
    """Extract text page-by-page from PDF."""
    pages = {}
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                pages[i] = text.strip()
    return pages

def parse_docx(file_bytes: bytes) -> dict:
    """Extract text from DOCX with approximate pagination."""
    doc = DocxDocument(io.BytesIO(file_bytes))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    # Approximate: ~500 words per page
    words = full_text.split()
    words_per_page = 500
    pages = {}
    for i in range(0, len(words), words_per_page):
        page_num = (i // words_per_page) + 1
        page_text = " ".join(words[i:i + words_per_page])
        pages[page_num] = page_text
    return pages if pages else {1: full_text}

def parse_txt(file_bytes: bytes) -> dict:
    """Parse TXT with approximate pagination."""
    text = file_bytes.decode("utf-8")
    words = text.split()
    words_per_page = 500
    pages = {}
    for i in range(0, len(words), words_per_page):
        page_num = (i // words_per_page) + 1
        page_text = " ".join(words[i:i + words_per_page])
        pages[page_num] = page_text
    return pages if pages else {1: text}

# ─── Text Chunking ──────────────────────────────────────────
def chunk_text(pages: dict) -> List[TextChunk]:
    """Split document into overlapping semantic chunks."""
    chunks = []
    for page_num, text in sorted(pages.items()):
        # Split by sentences for semantic boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < CHUNK_SIZE:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(TextChunk(
                        text=current_chunk.strip(),
                        page=page_num,
                        start_idx=current_start
                    ))
                current_chunk = sentence
                current_start = len(current_chunk)
                # Basic semantic overlap: retain the last sentence of the previous chunk if it fits
                overlap = current_chunk.split(". ")[-1] + ". " if ". " in current_chunk else ""
                current_chunk = overlap + sentence if len(overlap) < CHUNK_OVERLAP else sentence
                current_start += len(current_chunk) - len(sentence)
        
        if current_chunk:
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                page=page_num,
                start_idx=current_start
            ))
    return chunks

# ─── Vector Indexing ────────────────────────────────────────
def index_chunks_in_pinecone(doc_id: str, chunks: List[TextChunk]):
    """Create a Pinecone index and upsert text chunks."""
    if not pc or not chunks:
        return

    if doc_id not in pc.list_indexes().names():
        pc.create_index(name=doc_id, dimension=VECTOR_DIMENSION, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    
    index = pc.Index(doc_id)
    
    # Batch embed all chunk texts using Google's API
    chunk_texts = [c.text for c in chunks]
    try:
        result = google_ai_client.models.embed_content(
            model=EMBED_MODEL_NAME,
            contents=chunk_texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        embeddings = [e.values for e in result.embeddings]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    # Prepare vectors for upsert to Pinecone
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        vectors_to_upsert.append({
            "id": f"chunk-{i}",
            "values": embeddings[i],
            "metadata": {"text": chunk.text, "page": chunk.page}
        })
    
    # Upsert in batches to avoid request size limits
    for i in range(0, len(vectors_to_upsert), 100):
        batch = vectors_to_upsert[i:i+100]
        index.upsert(vectors=batch)

# ─── RAG Retrieval ──────────────────────────────────────────
def retrieve_chunks(query: str, doc_id: str, top_k: int = TOP_K):
    """Find most relevant chunks using semantic search."""
    if not pc or doc_id not in pc.list_indexes().names():
        return []
    
    index = pc.Index(doc_id)
    result = google_ai_client.models.embed_content(
        model=EMBED_MODEL_NAME,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_embedding = result.embeddings[0].values
    
    query_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    results = []
    for match in query_results.get('matches', []):
        results.append({
            "text": match['metadata']['text'],
            "page": match['metadata']['page'],
            "score": match['score']
        })
    return results

# ─── AI Answer Generation ───────────────────────────────────
def build_ai_answer(query: str, chunks: List[dict]) -> AskResponse:
    """
    Build answer using an AI model to summarize the exact passages.
    """
    if not chunks:
        return AskResponse(
            answer="Answer not found in document.",
            citations=[],
            highlighted_phrases=[]
        )
    
    parts = []
    citations = []
    highlighted = []
    
    for i, chunk in enumerate(chunks):
        parts.append(chunk["text"])
        highlighted.append(chunk["text"])
        citations.append({
            "id": f"c-{i}",
            "page": chunk["page"],
            "text": chunk["text"],
            "document_id": ""
        })
    
    context = "\n\n".join([f"Passage from page {c['page']}:\n>>> {c['text']}" for c in chunks])
    prompt = f"""You are an expert academic assistant. Based on the following passages from a document, provide a clear and well-structured answer to the user's question.
Use markdown for formatting, including headings, bullet points, and bold text to organize the information.
Your response should be comprehensive but based *only* on the information in the provided passages.

CONTEXT:
{context}

QUESTION: {query}"""
    
    if ai_client:
        try:
            response = ai_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
    else:
        # Fallback to exact concatenation if no API key is provided
        answer = " ".join(parts)
    
    return AskResponse(
        answer=answer,
        citations=citations,
        highlighted_phrases=highlighted
    )

# ─── API Endpoints ──────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document."""
    contents = await file.read()
    doc_id = hashlib.md5(contents).hexdigest()[:12]
    
    # Parse based on file type
    ext = file.filename.split(".")[-1].lower()
    if ext == "pdf":
        pages = parse_pdf(contents)
    elif ext in ("docx", "doc"):
        pages = parse_docx(contents)
    elif ext == "txt":
        pages = parse_txt(contents)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    
    # Store document
    full_text = " ".join(pages.values())
    word_count = len(full_text.split())
    
    doc = DocumentData(
        id=doc_id,
        name=file.filename,
        type=ext,
        page_count=max(pages.keys()) if pages else 0,
        word_count=word_count,
        pages=pages
    )
    
    # Chunk and index
    chunks = chunk_text(pages)
    index_chunks_in_pinecone(doc_id, chunks)
    
    # Persist metadata to MySQL
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO documents (id, name, type, page_count, word_count) VALUES (%s, %s, %s, %s, %s)", (doc.id, doc.name, doc.type, doc.page_count, doc.word_count))
        for page_num, text in pages.items():
            cursor.execute("INSERT INTO document_pages (doc_id, page_number, text) VALUES (%s, %s, %s)", (doc.id, page_num, text))
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"MySQL Insert Error: {e}")

    return {
        "id": doc_id,
        "name": file.filename,
        "type": ext,
        "page_count": doc.page_count,
        "word_count": word_count,
        "pages": [{"page_number": k, "text": v} for k, v in sorted(pages.items())]
    }

@app.post("/ask")
async def ask_question(req: AskRequest):
    """Ask a question about an uploaded document."""
    chunks = retrieve_chunks(req.query, req.doc_id)
    response = build_ai_answer(req.query, chunks)
    
    # Add document ID to citations
    for c in response.citations:
        c["document_id"] = req.doc_id
        
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chats (doc_id, role, content, citations, highlighted_phrases) VALUES (%s, %s, %s, %s, %s)", (req.doc_id, "user", req.query, "[]", "[]"))
        cursor.execute("INSERT INTO chats (doc_id, role, content, citations, highlighted_phrases) VALUES (%s, %s, %s, %s, %s)", (req.doc_id, "assistant", response.answer, json.dumps(response.citations), json.dumps(response.highlighted_phrases)))
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"MySQL Chat Error: {e}")
    
    return response

@app.get("/documents/{doc_id}/chats")
async def get_chats(doc_id: str):
    """Get chat history for a document from the database."""
    chat_history = []
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT role, content, citations, highlighted_phrases FROM chats WHERE doc_id = %s ORDER BY id ASC", (doc_id,))
        for row in cursor.fetchall():
            chat_history.append({
                "role": row["role"],
                "content": row["content"],
                "citations": json.loads(row["citations"]) if row["citations"] else [],
                "highlighted_phrases": json.loads(row["highlighted_phrases"]) if row["highlighted_phrases"] else []
            })
        cursor.close()
        conn.close()
    except Error as e:
        print(f"MySQL Get Chats Error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch chat history.")
    return chat_history

@app.delete("/documents/{doc_id}/chats")
async def clear_chat_history(doc_id: str):
    """Clear chat history for a document from memory and the database."""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chats WHERE doc_id = %s", (doc_id,))
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"MySQL Chat Clear Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear chat history from database.")
    return {"status": "chat history cleared"}

@app.get("/documents")
async def list_documents():
    """List all uploaded documents from the database."""
    docs = []
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, type, page_count, word_count FROM documents")
        docs = cursor.fetchall()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"MySQL List Error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch documents.")
    return docs

@app.get("/documents/{doc_id}/pages")
async def get_document_pages(doc_id: str):
    """Get all pages of a document from the database."""
    pages = []
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT page_number, text FROM document_pages WHERE doc_id = %s ORDER BY page_number ASC", (doc_id,))
        pages = cursor.fetchall()
        if not pages: # Check if the document existed at all
            cursor.execute("SELECT id FROM documents WHERE id = %s", (doc_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Document not found")
        cursor.close()
        conn.close()
    except Error as e:
        print(f"MySQL Get Pages Error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch document pages.")
    return {"pages": pages}

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    if pc and doc_id in pc.list_indexes().names():
        pc.delete_index(doc_id)
            
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"MySQL Delete Error: {e}")
        
    return {"status": "deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)