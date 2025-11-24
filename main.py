import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in .env")
    sys.exit(1)
else:
    print(f"✅ Loaded API Key: {OPENAI_API_KEY[:8]}... (hidden)")
    print(f"✅ Using model: {MODEL}")

def load_pdf_with_ocr(file_path):
    try:
        pages = convert_from_path(file_path)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page)
        return text
    except Exception as e:
        print(f"⚠️ OCR failed for {file_path}: {e}")
        return ""

def load_documents():
    docs = []
    data_path = "data"
    if not os.path.exists(data_path):
        print("❌ ERROR: 'data/' folder not found. Please create it and add your files.")
        sys.exit(1)

    print(f"📂 Found data folder: {data_path}")
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
                if not loaded_docs or not any(d.page_content.strip() for d in loaded_docs):
                    text = load_pdf_with_ocr(file_path)
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"source": file_path}))
                        print(f"📄 Loaded (OCR) {file}")
                        continue
                docs.extend(loaded_docs)
                print(f"📄 Loaded {file}")
            except Exception as e:
                print(f"⚠️ Failed to load PDF {file}: {e}")
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs.extend(loader.load())
            print(f"📄 Loaded {file}")
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
            docs.extend(loader.load())
            print(f"📄 Loaded {file}")
        elif file.endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
                text = df.to_string(index=False)
                docs.append(Document(page_content=text, metadata={"source": file_path}))
                print(f"📄 Loaded CSV {file}")
            except Exception as e:
                print(f"⚠️ Failed to load CSV {file}: {e}")
        else:
            print(f"⚠️ Skipping unsupported file: {file}")
            continue
    return docs

def ingest():
    try:
        print("📚 Loading documents...")
        documents = load_documents()

        if not documents:
            print("❌ No documents found in data/. Please add some files.")
            sys.exit(1)

        print("✂️ Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")

        print("🔎 Creating embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        print("💾 Saving FAISS index...")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local("vectorstore")
        print("✅ Ingestion complete!")

    except Exception as e:
        print(f"❌ ERROR during ingestion: {e}")

def ask():
    try:
        print("🤖 AI Assistant ready! Ask me anything (type 'exit' to quit).")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("📂 Loading FAISS index from vectorstore...")
        db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        print("✅ Vectorstore loaded!")

        llm = ChatOpenAI(model=MODEL, temperature=0)
        qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever())

        chat_history = []

        while True:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            result = qa.invoke({"question": query, "chat_history": chat_history})
            answer = result["answer"]
            print(f"Assistant: {answer}")
            chat_history.append((query, answer))

    except Exception as e:
        print(f"❌ ERROR in ask(): {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [ingest|ask]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        ingest()
    elif command == "ask":
        ask()
    else:
        print("Unknown command. Use 'ingest' or 'ask'.")
