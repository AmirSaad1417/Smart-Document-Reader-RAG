from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from config.settings import settings
from src.core.logger import get_logger
from src.rag.memory import ConversationMemory
from src.rag.prompts import RAG_PROMPT_TEMPLATE

logger = get_logger("rag_service")

class RAGService:
    """Orchestrates RAG with intelligent dual-provider failover (Gemini <-> Groq)."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        memory: Optional[ConversationMemory] = None
    ):
        self.retriever = retriever
        self.memory = memory or ConversationMemory()

        # 1. Initialize Gemini LLM Chain
        self.gemini_chain = None
        gemini_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        if gemini_key:
            try:
                gemini_llm = ChatGoogleGenerativeAI(
                    model=settings.GEMINI_LLM_MODEL,
                    google_api_key=gemini_key
                )
                self.gemini_chain = RAG_PROMPT_TEMPLATE | gemini_llm | StrOutputParser()
                logger.info(f"Initialized Gemini LLM ({settings.GEMINI_LLM_MODEL})")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini LLM: {e}")

        # 2. Initialize Groq LLM Chain
        self.groq_chain = None
        if settings.GROQ_API_KEY:
            try:
                groq_llm = ChatGroq(
                    model=settings.GROQ_LLM_MODEL,
                    groq_api_key=settings.GROQ_API_KEY,
                    temperature=settings.LLM_TEMPERATURE
                )
                self.groq_chain = RAG_PROMPT_TEMPLATE | groq_llm | StrOutputParser()
                logger.info(f"Initialized Groq LLM ({settings.GROQ_LLM_MODEL})")
            except Exception as e:
                logger.warning(f"Could not initialize Groq LLM: {e}")

        if not self.gemini_chain and not self.groq_chain:
            raise ValueError("Neither Gemini nor Groq LLM could be initialized. Please check your API keys.")

    def ask(self, question: str) -> Tuple[str, List[Document]]:
        """Processes a question using dual-provider failover.

        Tries the primary LLM first; if a rate limit (429) or error occurs,
        automatically falls back to the second provider seamlessly.
        """
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        # Contextual search query expansion for follow-up or short questions
        search_query = clean_question
        if self.memory.messages:
            last_human, _ = self.memory.messages[-1]
            if len(clean_question.split()) <= 6:
                search_query = f"{last_human} {clean_question}"

        logger.info(f"Retrieving context for query: '{search_query}'")
        retrieved_docs: List[Document] = self.retriever.invoke(search_query)

        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        history_text = self.memory.get_formatted_history()

        payload = {
            "chat_history": history_text,
            "context": context_text,
            "question": clean_question
        }

        answer: Optional[str] = None
        last_error: Optional[Exception] = None

        # Chain execution order: Gemini -> Groq (or fallback)
        providers = []
        if self.gemini_chain:
            providers.append(("Google Gemini", self.gemini_chain))
        if self.groq_chain:
            providers.append(("Groq", self.groq_chain))

        for provider_name, chain in providers:
            try:
                logger.info(f"Attempting response generation with {provider_name}...")
                answer = chain.invoke(payload)
                if answer and isinstance(answer, str) and answer.strip():
                    logger.info(f"Successfully generated answer using {provider_name}.")
                    break
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{provider_name} encountered an error: {str(e)}. "
                    f"Failing over to alternative provider..."
                )

        if not answer:
            error_msg = f"All LLM providers failed. Last error: {str(last_error)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Save turn to memory
        self.memory.add_interaction(clean_question, answer)
        return answer, retrieved_docs

    def clear_memory(self) -> None:
        """Resets the conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")
