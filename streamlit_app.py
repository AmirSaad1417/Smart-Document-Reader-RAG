import os
import streamlit as st
from pathlib import Path
from tempfile import NamedTemporaryFile

from config.settings import settings
from src.document_loaders.pdf_loader import PDFProcessor
from src.vectorstore.faiss_store import VectorStoreManager
from src.rag.chain import RAGService
from src.rag.memory import ConversationMemory

# Page Configuration
st.set_page_config(
    page_title="Smart Document Reader",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Readability CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 20px;
    }
    .stChatMessage {
        font-size: 15.5px !important;
        line-height: 1.68 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="main-header">📑 Smart Document Reader</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Turn any document into a smart assistant — ask for summaries, key facts, or detailed explanations.</div>',
    unsafe_allow_html=True
)

# Initialize Session State
if "rag_service" not in st.session_state:
    st.session_state.rag_service = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "page_count" not in st.session_state:
    st.session_state.page_count = 0

# Sidebar: Document Upload
with st.sidebar:
    st.subheader("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("📖 Read Document", type="primary", use_container_width=True):
            with st.spinner("Reading and indexing document..."):
                try:
                    # Save temporarily to parse
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # 1. Load and split
                    processor = PDFProcessor()
                    chunks = processor.load_and_split(tmp_path)
                    st.session_state.page_count = processor.last_page_count or 1

                    # 2. Embed and build FAISS index
                    vector_manager = VectorStoreManager()
                    vector_manager.build_from_documents(chunks)

                    # 3. Initialize RAG Service
                    retriever = vector_manager.as_retriever()
                    memory = ConversationMemory()
                    st.session_state.rag_service = RAGService(retriever=retriever, memory=memory)
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.messages = []

                    # Cleanup temporary file
                    os.unlink(tmp_path)
                    st.success(f"✅ Ready: {uploaded_file.name} ({st.session_state.page_count} pages)")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")

    if st.session_state.current_file:
        st.info(f"🟢 **Active:** `{st.session_state.current_file}` ({st.session_state.page_count} pages)")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        if st.session_state.rag_service:
            st.session_state.rag_service.clear_memory()
        st.session_state.messages = []
        st.rerun()

# Main Chat Area
if not st.session_state.rag_service:
    st.info("👈 Please upload your PDF in the left sidebar and click **'Read Document'** to start asking questions.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask any question about your document..."):
    if not st.session_state.rag_service:
        st.warning("Please upload a PDF document first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    answer, docs = st.session_state.rag_service.ask(prompt)

                    # Format page sources in a clean collapsible expander
                    page_numbers = []
                    source_items = []
                    for doc in docs:
                        page_raw = doc.metadata.get("page", 0)
                        page_num = int(page_raw) + 1 if isinstance(page_raw, (int, float)) else str(page_raw)
                        page_numbers.append(f"Page {page_num}")
                        snippet = doc.page_content[:180].replace("\n", " ").strip()
                        source_items.append(f"- **Page {page_num}**: *\"{snippet}...\"*")

                    pages_label = ", ".join(dict.fromkeys(page_numbers))
                    sources_text = "\n".join(source_items)

                    full_response = (
                        f"{answer}\n\n"
                        f"<details style='margin-top:12px; padding:8px 12px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; font-size:13.5px; color:#475569;'>"
                        f"<summary style='cursor:pointer; font-weight:600; color:#2563eb;'>📍 View Document Sources ({pages_label})</summary>\n\n"
                        f"{sources_text}\n"
                        f"</details>"
                    )

                    st.markdown(full_response, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    error_msg = f"Sorry, could not generate an answer: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
