"""Command Line Interface (CLI) runner for RAG pipeline."""
import sys
from pathlib import Path
from src.core.logger import get_logger
from src.document_loaders.pdf_loader import PDFProcessor
from src.vectorstore.faiss_store import VectorStoreManager
from src.rag.chain import RAGService

logger = get_logger("cli")

def run_pipeline(pdf_path: str, query: str = "What is the main topic of this document?"):
    """Executes the document answering pipeline headlessly."""
    path = Path(pdf_path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)

    logger.info("=== Reading Document ===")
    processor = PDFProcessor()
    chunks = processor.load_and_split(path)

    logger.info("=== Indexing Sections ===")
    vector_manager = VectorStoreManager()
    vector_manager.build_from_documents(chunks)

    logger.info("=== Preparing Assistant ===")
    retriever = vector_manager.as_retriever()
    rag_service = RAGService(retriever=retriever)

    logger.info(f"=== Answering Question: '{query}' ===")
    answer, docs = rag_service.ask(query)

    print("\n" + "="*50)
    print(f"💬 Answer:\n{answer}")
    print("="*50)
    print("\n📍 Verified page references:")
    for doc in docs:
        page_raw = doc.metadata.get("page", 0)
        page_num = int(page_raw) + 1 if isinstance(page_raw, (int, float)) else str(page_raw)
        snippet = doc.page_content[:180].replace("\n", " ").strip()
        print(f"\n• Page {page_num}: \"{snippet}...\"")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        user_query = sys.argv[2] if len(sys.argv) > 2 else "What is the summary of this document?"
        run_pipeline(pdf_file, user_query)
    else:
        print("Usage: python main.py <path_to_pdf> [optional_question]")
