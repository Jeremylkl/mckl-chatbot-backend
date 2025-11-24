import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from PyPDF2 import PdfReader
import docx

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

DATA_DIR = Path("data")          # put your PDFs / DOCX / TXT files here
INDEX_DIR = Path("chroma_db")    # Chroma will persist its DB here

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


# ------------ Load raw text from files ------------

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def load_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents() -> List[Document]:
    docs: List[Document] = []

    for path in DATA_DIR.glob("**/*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            text = load_pdf(path)
        elif suffix in {".docx", ".doc"}:
            text = load_docx(path)
        elif suffix in {".txt", ".md"}:
            text = load_txt(path)
        else:
            # skip unsupported files
            continue

        if not text.strip():
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path.relative_to(DATA_DIR))},
            )
        )

    return docs


# ------------ Chunking ------------

def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


# ------------ Build & persist Chroma DB ------------

def create_vectorstore(chunks: List[Document]) -> None:
    embeddings = OpenAIEmbeddings()

    INDEX_DIR.mkdir(exist_ok=True)

    db = Chroma.from_documents(
        documents=chunks,
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
    )
    db.persist()


def main() -> None:
    print(f"📁 Looking for documents in: {DATA_DIR.resolve()}")

    docs = load_documents()
    if not docs:
        print("⚠️  No documents found in 'data/'. Put PDFs / DOCX / TXT there and run again.")
        return

    print(f"✅ Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(docs)
    print(f"🔹 Total chunks: {len(chunks)}")

    print("📦 Building Chroma index...")
    create_vectorstore(chunks)
    print(f"✅ Ingestion complete. Index stored in {INDEX_DIR.resolve()}")


if __name__ == "__main__":
    main()
