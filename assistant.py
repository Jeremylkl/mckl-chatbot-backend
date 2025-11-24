import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA

load_dotenv()

# Folder where the vector store will live
INDEX_DIR = Path("chroma_db")

# Model + temperature from env, with safe defaults
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))


class AIAssistant:
    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        model_name: str = OPENAI_MODEL,
        temperature: float = TEMPERATURE,
    ) -> None:
        if not index_dir.exists():
            raise RuntimeError(
                "Chroma index not found. Run ingest.py first to build the index."
            )

        # Embeddings + Chroma vector store
        embeddings = OpenAIEmbeddings()

        self.db = Chroma(
            persist_directory=str(index_dir),
            embedding_function=embeddings,
        )

        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6},
        )

        # Chat model
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

        # QA chain that returns sources
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
        )

    def query(self, question: str) -> Dict:
        """Run a query against the vector store + LLM."""
        result = self.qa_chain({"query": question})
        answer: str = result["result"]
        sources: List[Document] = result.get("source_documents", [])

        source_strings: List[str] = []
        for i, doc in enumerate(sources, start=1):
            meta = doc.metadata or {}
            label = (
                meta.get("source")
                or meta.get("file_path")
                or meta.get("filename")
                or f"Document {i}"
            )
            source_strings.append(f"{i}. {label}")

        return {
            "answer": answer,
            "sources": source_strings,
        }
