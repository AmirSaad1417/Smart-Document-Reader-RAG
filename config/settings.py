import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Detect Streamlit Cloud secrets if available
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for secret_key in ["GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
            if secret_key in st.secrets and not os.environ.get(secret_key):
                os.environ[secret_key] = str(st.secrets[secret_key])
except Exception:
    pass

class Settings(BaseSettings):
    """Central application settings and environment variables configuration."""

    # API Keys
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # Model Configuration
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    GEMINI_LLM_MODEL: str = "gemini-3.5-flash-lite"
    GROQ_LLM_MODEL: str = "openai/gpt-oss-120b"
    LLM_TEMPERATURE: float = 0.0

    # Document Chunking Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval Settings
    RETRIEVER_K: int = 4

    # Storage Paths
    DATA_DIR: Path = PROJECT_ROOT / "data"
    VECTOR_STORE_DIR: Path = PROJECT_ROOT / "storage" / "faiss_index"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def validate_keys(self) -> None:
        """Validate at least one LLM key and embedding key are present."""
        gemini_key = self.GEMINI_API_KEY or self.GOOGLE_API_KEY
        if not gemini_key and not self.GROQ_API_KEY:
            raise ValueError("No API keys found. Please configure GEMINI_API_KEY or GROQ_API_KEY in secrets or .env.")

# Global singleton settings instance
settings = Settings()
