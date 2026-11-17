from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.core.logger import get_logger

logger = get_logger("pdf_loader")

class PDFProcessor:
    """Handles PDF ingestion, validation, and chunking with metadata preservation."""

    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.last_page_count: int = 0
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_and_split(self, file_path: Union[str, Path]) -> List[Document]:
        """Loads a PDF file from disk and splits it into manageable chunks.

        Args:
            file_path: The filesystem path to the PDF document.

        Returns:
            List of chunked Document objects containing page content and metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        logger.info(f"Loading PDF document: {path.name}")
        loader = PyPDFLoader(str(path))
        pages = loader.load()

        if not pages:
            raise ValueError(f"No readable pages found in PDF: {path.name}")

        self.last_page_count = len(pages)
        logger.info(f"Loaded {len(pages)} pages. Splitting into chunks...")
        chunks = self.text_splitter.split_documents(pages)

        # Enhance chunk metadata with filename and total page count
        for chunk in chunks:
            chunk.metadata["filename"] = path.name
            chunk.metadata["total_pages"] = len(pages)

        logger.info(f"Successfully generated {len(chunks)} chunks from {path.name}")
        return chunks
