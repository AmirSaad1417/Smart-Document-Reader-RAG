from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import gradio as gr

from src.core.logger import get_logger
from src.document_loaders.pdf_loader import PDFProcessor
from src.vectorstore.faiss_store import VectorStoreManager
from src.rag.chain import RAGService
from src.rag.memory import ConversationMemory

logger = get_logger("ui")

# Minimalist, High-Contrast CSS for maximum readability and clean layout
CLEAN_CSS = """
<style>
* {
    box-sizing: border-box;
}

body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* Minimalist Top Header Bar */
.navbar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 17px;
    font-weight: 700;
    color: #0f172a;
}
.navbar-subtitle {
    font-size: 13px;
    color: #64748b;
}

/* Sidebar Box */
.sidebar-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}
.sidebar-title {
    font-size: 13px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Document Status Indicators */
.doc-status-badge {
    padding: 9px 12px;
    border-radius: 6px;
    font-size: 13.5px;
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.doc-status-waiting {
    background-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    color: #64748b;
}
.doc-status-ready {
    background-color: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
    font-weight: 600;
}

/* Chatbot Custom Styling & Optimal Text Readability */
#doc-chatbot {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background-color: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
}

/* Ensure Chat text is high contrast, larger font, and clean line-height */
#doc-chatbot .message, 
#doc-chatbot [data-testid="user"], 
#doc-chatbot [data-testid="bot"],
#doc-chatbot [class*="message"] {
    font-size: 15.5px !important;
    line-height: 1.68 !important;
    color: #0f172a !important;
}

/* High contrast user prompt styling */
#doc-chatbot [data-testid="user"], 
#doc-chatbot .user {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}

/* High contrast assistant response styling */
#doc-chatbot [data-testid="bot"], 
#doc-chatbot .bot {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}

/* Hide default Gradio footer completely */
footer, .gradio-container footer, .footer {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Action Buttons */
.btn-primary {
    background-color: #2563eb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
}
.btn-primary:hover {
    background-color: #1d4ed8 !important;
}
.btn-secondary {
    background-color: #f8fafc !important;
    color: #475569 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}
.btn-secondary:hover {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
}
</style>
"""

# JavaScript injected into <head> to anchor viewport to the start of the question
HEAD_SCRIPT = """
<script>
(function() {
    let lastUserCount = 0;

    function anchorToLatestQuestion() {
        const chatbot = document.querySelector('#doc-chatbot');
        if (!chatbot) return;

        // Find user question bubble in the chatbot
        const userBubbles = chatbot.querySelectorAll('[data-testid="user"], .user, [class*="user"]');
        if (userBubbles.length > lastUserCount) {
            lastUserCount = userBubbles.length;
            const latestUser = userBubbles[userBubbles.length - 1];
            if (latestUser) {
                // Smoothly anchor to the start of the user question
                latestUser.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
    }

    const observer = new MutationObserver(() => {
        anchorToLatestQuestion();
    });

    window.addEventListener('DOMContentLoaded', () => {
        const target = document.querySelector('#doc-chatbot') || document.body;
        observer.observe(target, { childList: true, subtree: true });
    });
})();
</script>
"""

class RAGSession:
    """Manages active user session, document loading, and question answering."""

    def __init__(self):
        self.rag_service: Optional[RAGService] = None
        self.vector_manager: Optional[VectorStoreManager] = None
        self.current_filename: Optional[str] = None
        self.total_pages: int = 0
        self.memory = ConversationMemory()

    def process_pdf(self, file_path: str) -> str:
        """Processes an uploaded PDF file and sets up the document for answering questions."""
        if not file_path:
            return """
            <div class="doc-status-badge doc-status-waiting">
                <span>⚠️</span>
                <span>No PDF selected. Please choose a file first.</span>
            </div>
            """

        try:
            path = Path(file_path)
            logger.info(f"Ingesting PDF: {path.name}")

            # 1. Read and chunk the document
            processor = PDFProcessor()
            chunks = processor.load_and_split(path)
            self.total_pages = processor.last_page_count or 1

            # 2. Index in vector store
            self.vector_manager = VectorStoreManager()
            self.vector_manager.build_from_documents(chunks)

            # 3. Initialize RAG Service
            retriever = self.vector_manager.as_retriever()
            self.memory.clear()
            self.rag_service = RAGService(
                retriever=retriever,
                memory=self.memory
            )
            self.current_filename = path.name

            return f"""
            <div class="doc-status-badge doc-status-ready">
                <span>🟢</span>
                <span><strong>Ready:</strong> {path.name} ({self.total_pages} pages)</span>
            </div>
            """
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}", exc_info=True)
            return f"""
            <div class="doc-status-badge doc-status-waiting" style="background:#fef2f2; border-color:#fca5a5; color:#991b1b;">
                <span>❌</span>
                <span>Could not read file: {str(e)}</span>
            </div>
            """

    def chat(self, question: str, history: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], str]:
        """Answers a user question with collapsible, clean page references."""
        history_list: List[Dict[str, Any]] = list(history or [])

        if not question or not question.strip():
            return history_list, ""

        if self.rag_service is None:
            warning_msg = (
                "Please upload a PDF document in the left sidebar and click **'Read Document'** first."
            )
            history_list.append({"role": "user", "content": question})
            history_list.append({"role": "assistant", "content": warning_msg})
            return history_list, ""

        try:
            answer, docs = self.rag_service.ask(question)

            # Format source references inside a clean collapsible details accordion with explicit line breaks
            source_items = []
            page_numbers = []
            for doc in docs:
                page_raw = doc.metadata.get("page", 0)
                page_num = int(page_raw) + 1 if isinstance(page_raw, (int, float)) else str(page_raw)
                page_numbers.append(f"Page {page_num}")
                snippet = doc.page_content[:180].replace("\n", " ").strip()
                source_items.append(
                    f"<div style='margin-bottom: 8px; line-height: 1.5;'>"
                    f"<b>• Page {page_num}:</b> <i>\"{snippet}...\"</i>"
                    f"</div>"
                )

            pages_label = ", ".join(dict.fromkeys(page_numbers))
            sources_block = "\n".join(source_items)

            # Clean layout: Crisp Answer, followed by an unobtrusive collapsible sources bar
            full_response = (
                f"{answer}\n\n"
                f"<details style='margin-top:14px; padding:10px 14px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; font-size:13px; color:#475569;'>"
                f"<summary style='cursor:pointer; font-weight:600; color:#2563eb;'>📍 View Document Sources ({pages_label})</summary>"
                f"<div style='margin-top: 10px;'>"
                f"{sources_block}"
                f"</div>"
                f"</details>"
            )

            history_list.append({"role": "user", "content": question})
            history_list.append({"role": "assistant", "content": full_response})
            return history_list, ""
        except Exception as e:
            logger.error(f"Inference error: {str(e)}", exc_info=True)
            error_response = f"Sorry, could not retrieve an answer: {str(e)}"
            history_list.append({"role": "user", "content": question})
            history_list.append({"role": "assistant", "content": error_response})
            return history_list, ""

    def clear(self) -> Tuple[List[Dict[str, Any]], str]:
        """Clears chat history and conversational memory."""
        if self.rag_service:
            self.rag_service.clear_memory()
        return [], ""


def create_gradio_app() -> gr.Blocks:
    """Builds the clean, minimalist, high-readability Document Chat interface."""
    session = RAGSession()

    with gr.Blocks(title="Smart Document Reader") as demo:
        # Inject Clean CSS
        gr.HTML(CLEAN_CSS)

        # Minimalist Header Bar
        gr.HTML(
            """
            <div class="navbar-header">
                <div class="navbar-brand">
                    <span>📑</span> Smart Document Reader
                </div>
                <div class="navbar-subtitle">
                    Turn any document into a smart assistant — ask for summaries, key facts, or detailed explanations.
                </div>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            # Left Sidebar: Document Management
            with gr.Column(scale=3):
                with gr.Column(elem_classes=["sidebar-panel"]):
                    gr.HTML('<div class="sidebar-title"><span>📂</span> Upload Document</div>')

                    pdf_input = gr.File(
                        label="Select PDF File",
                        file_types=[".pdf"],
                        type="filepath",
                        show_label=False
                    )

                    read_btn = gr.Button(
                        "Read Document",
                        variant="primary",
                        elem_classes=["btn-primary"]
                    )

                    status_display = gr.HTML(
                        """
                        <div class="doc-status-badge doc-status-waiting">
                            <span>⚪</span>
                            <span>No document loaded</span>
                        </div>
                        """
                    )

                    gr.HTML("<div style='height: 14px;'></div>")

                    clear_btn = gr.Button(
                        "🗑️ Clear Conversation",
                        variant="secondary",
                        size="sm",
                        elem_classes=["btn-secondary"]
                    )

            # Right Main Column: Chat Interface
            with gr.Column(scale=9):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    show_label=False,
                    height=560,
                    elem_id="doc-chatbot",
                    autoscroll=False,
                    placeholder="👋 Upload a document on the left and ask your first question here."
                )

                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Type your question here and press Enter...",
                        show_label=False,
                        scale=9,
                        lines=1,
                        autofocus=True
                    )
                    ask_btn = gr.Button(
                        "Ask",
                        variant="primary",
                        scale=1,
                        elem_classes=["btn-primary"]
                    )

        # Event Bindings
        read_btn.click(
            fn=session.process_pdf,
            inputs=[pdf_input],
            outputs=[status_display]
        )

        # Question Submission (via Enter key or Ask button)
        question_input.submit(
            fn=session.chat,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input]
        )

        ask_btn.click(
            fn=session.chat,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input]
        )

        # Clear Event
        clear_btn.click(
            fn=session.clear,
            inputs=[],
            outputs=[chatbot, question_input]
        )

    return demo
