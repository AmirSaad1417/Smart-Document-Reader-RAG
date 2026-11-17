from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import settings
from src.core.logger import get_logger

logger = get_logger("vectorstore")

class VectorStoreManager:
    """Manages embedding generation, FAISS indexing, persistence, and retrieval."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        if not key:
            raise ValueError("Gemini API key is required to initialize GoogleGenerativeAIEmbeddings.")

        self.model = model or settings.EMBEDDING_MODEL
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model,
            google_api_key=key
        )
        self.vector_store: Optional[FAISS] = None

    def build_from_documents(self, documents: List[Document]) -> FAISS:
        """Embeds a list of Document chunks and constructs an in-memory FAISS index.

        Args:
            documents: List of chunked Document instances.

        Returns:
            The initialized FAISS vector store.
        """
        if not documents:
            raise ValueError("No documents provided to build vector index.")

        logger.info(f"Generating embeddings for {len(documents)} chunks using {self.model}...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info(f"FAISS index built successfully. Total entries: {self.vector_store.index.ntotal}")
        return self.vector_store

    def save_local(self, folder_path: Optional[Path] = None) -> None:
        """Persists the current FAISS index and docstore to disk."""
        if self.vector_store is None:
            raise ValueError("No active vector store index to persist.")

        save_path = Path(folder_path or settings.VECTOR_STORE_DIR)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(save_path))
        logger.info(f"FAISS index persisted to disk at: {save_path}")

    def load_local(self, folder_path: Optional[Path] = None) -> FAISS:
        """Loads a persisted FAISS index from disk."""
        load_path = Path(folder_path or settings.VECTOR_STORE_DIR)
        if not load_path.exists():
            raise FileNotFoundError(f"No FAISS index found at: {load_path}")

        logger.info(f"Loading existing FAISS index from {load_path}...")
        self.vector_store = FAISS.load_local(
            folder_path=str(load_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"FAISS index loaded. Total entries: {self.vector_store.index.ntotal}")
        return self.vector_store

    def as_retriever(self, k: int = settings.RETRIEVER_K) -> VectorStoreRetriever:
        """Returns a retriever instance configured for similarity search."""
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Build or load an index first.")

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
