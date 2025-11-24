import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

CHROMA_DIR = Path("chroma_db")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))


class AIAssistant:
    def __init__(
        self,
        db_dir: Path = CHROMA_DIR,
        model_name: str = OPENAI_MODEL,
        temperature: float = TEMPERATURE,
    ):
        if not db_dir.exists():
            raise RuntimeError(
                "Chroma DB not found. Run ingest.py first to build the vector store."
            )

        embeddings = OpenAIEmbeddings()

        self.db = Chroma(
            embedding_function=embeddings,
            persist_directory=str(db_dir),
        )

        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6},
        )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
        )

        self.chat_history = []

    def ask(self, query: str) -> dict:
        """
        Returns dict with 'answer' (str) and 'sources' (list of file paths).
        """
        result = self.qa_chain({"query": query})
        answer = result["result"]
        docs = result.get("source_documents", []) or []
        sources = [d.metadata.get("source", "unknown") for d in docs]

        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {"answer": answer, "sources": sources}

    def show_history(self) -> str:
        out = []
        for turn in self.chat_history:
            role = "You" if turn["role"] == "user" else "Assistant"
            out.append(f"{role}: {turn['content']}")
        return "\n".join(out)
