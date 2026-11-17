"""Unit and integration tests for RAG components."""
import unittest
from langchain_core.documents import Document

from config.settings import settings
from src.rag.memory import ConversationMemory
from src.rag.prompts import RAG_PROMPT_TEMPLATE
from src.document_loaders.pdf_loader import PDFProcessor

class TestRAGComponents(unittest.TestCase):

    def test_settings_loaded(self):
        """Verify settings load properly from .env or defaults."""
        self.assertIsNotNone(settings.GROQ_API_KEY)
        self.assertIsNotNone(settings.GEMINI_API_KEY)
        self.assertEqual(settings.CHUNK_SIZE, 1000)
        self.assertEqual(settings.CHUNK_OVERLAP, 200)

    def test_conversation_memory(self):
        """Test conversation memory storage and formatting."""
        memory = ConversationMemory(max_turns=2)
        memory.add_interaction("Hello", "Hi there!")
        memory.add_interaction("My name is Alice", "Nice to meet you Alice!")

        history = memory.get_formatted_history()
        self.assertIn("Human: Hello", history)
        self.assertIn("Human: My name is Alice", history)

        # Test max turns trimming
        memory.add_interaction("What is today's topic?", "AI RAG architecture.")
        trimmed_history = memory.get_formatted_history()
        self.assertNotIn("Human: Hello", trimmed_history)
        self.assertIn("Human: My name is Alice", trimmed_history)

    def test_prompt_template_formatting(self):
        """Verify prompt template populates variables correctly."""
        prompt_val = RAG_PROMPT_TEMPLATE.format(
            chat_history="None",
            context="This is document context.",
            question="What is this?"
        )
        self.assertIn("This is document context.", prompt_val)
        self.assertIn("What is this?", prompt_val)

    def test_text_splitting_logic(self):
        """Verify text splitter creates appropriate chunks."""
        processor = PDFProcessor(chunk_size=100, chunk_overlap=20)
        sample_doc = Document(page_content="Sentence one. " * 30, metadata={"page": 0})
        chunks = processor.text_splitter.split_documents([sample_doc])
        self.assertGreater(len(chunks), 1)

if __name__ == "__main__":
    unittest.main()
