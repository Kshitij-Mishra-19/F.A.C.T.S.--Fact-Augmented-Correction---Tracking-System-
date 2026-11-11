from pathlib import Path
import hashlib

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Project paths
ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_DIR = ROOT / "data" / "documents"
DB_DIR = ROOT / "chroma_db"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

# Persistent ChromaDB client
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="halluguard_knowledge",
    metadata={"hnsw:space": "cosine"}
)


def extract_text(file_path):
    """
    Extract text from supported document formats.
    """

    suffix = file_path.suffix.lower()

    if suffix in [".txt", ".md"]:
        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    elif suffix == ".pdf":
        reader = PdfReader(str(file_path))

        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    return ""


import spacy

nlp = spacy.load("en_core_web_sm")


def chunk_text(text):
    """
    Split documents into sentence-level evidence chunks.
    """

    doc = nlp(text)

    chunks = []

    for sentence in doc.sents:

        sentence_text = sentence.text.strip()

        # Ignore very short headings and fragments
        if len(sentence_text.split()) < 4:
            continue

        chunks.append(sentence_text)

    return chunks


def create_chunk_id(source, index, content):
    """
    Generate a stable ID for each chunk.
    """

    raw = f"{source}:{index}:{content}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def ingest_documents():
    """
    Read documents, create embeddings and store them in ChromaDB.
    """

    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        path for path in DOCUMENT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in [".txt", ".md", ".pdf"]
    ]

    if not files:
        print("No supported documents found.")
        return

    total_chunks = 0

    for file_path in files:

        print(f"\nProcessing: {file_path.name}")

        text = extract_text(file_path)

        if not text.strip():
            print("No extractable text. Skipping.")
            continue

        chunks = chunk_text(text)

        if not chunks:
            continue

        embeddings = model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        ids = []
        metadatas = []

        source = str(file_path.relative_to(ROOT))

        for index, chunk in enumerate(chunks):

            ids.append(
                create_chunk_id(source, index, chunk)
            )

            metadatas.append({
                "source": source,
                "chunk_index": index
            })

        collection.delete(
            where={"source": source}
        )
        
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        total_chunks += len(chunks)

        print(f"Chunks stored: {len(chunks)}")

    print("\nDocument ingestion completed.")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Database location: {DB_DIR}")


if __name__ == "__main__":
    ingest_documents()