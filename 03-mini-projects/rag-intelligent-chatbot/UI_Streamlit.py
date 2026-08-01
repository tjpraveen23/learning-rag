# streamlit_app.py
import streamlit as st
import logging
from chat_interface import ChatInterface
from datetime import datetime

# -------------------
# Logger Configuration
# -------------------
log_file = "chat_ui.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------
# Chat Session Setup
# -------------------
if "chat" not in st.session_state:
    st.session_state.chat = ChatInterface()

if "history" not in st.session_state:
    st.session_state.history = []

chat = st.session_state.chat

# -------------------
# Sidebar
# -------------------
st.sidebar.title("📌 Multi-LOB RAG Assistant")
st.sidebar.write("**Available LOBs:**")
st.sidebar.write(", ".join(chat.rag_framework.get_available_lobs()))
st.sidebar.write(f"**Current LOB:** {chat.current_lob}")

lob_choice = st.sidebar.selectbox(
    "Switch LOB",
    chat.rag_framework.get_available_lobs(),
    index=chat.rag_framework.get_available_lobs().index(chat.current_lob)
)
if lob_choice != chat.current_lob:
    chat.current_lob = lob_choice
    if lob_choice not in chat.rag_framework.get_initialized_lobs():
        st.sidebar.write(f"🔄 Initializing {lob_choice}...")
        if chat.rag_framework.initialize_lob(lob_choice):
            st.sidebar.success(f"{lob_choice} initialized successfully")
            logger.info(f"Initialized LOB: {lob_choice}")
        else:
            st.sidebar.error(f"Failed to initialize {lob_choice}")
            logger.error(f"Failed to initialize LOB: {lob_choice}")

# -------------------
# Main Chat Display
# -------------------
st.title("🤖 Multi-LOB RAG Chat")

for item in st.session_state.history:
    st.markdown(f"**🧑 You:** {item['question']}")
    if item["result"]["success"]:
        st.markdown(f"**🤖 Answer:** {item['result']['answer']}")
        st.markdown(f"**📚 Sources:**")
        for src in item["result"]["context_sources"]:
            st.markdown(f"- {src}")
    else:
        st.error(f"❌ Error: {item['result']['error']}")
    st.markdown("---")

# -------------------
# Input Form
# -------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question or enter a command (/init, /status, etc.)")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    logger.info(f"User input: {user_input}")  # Log every input
    
    if user_input.startswith("/"):
        chat._handle_command(user_input)
        st.success("✅ Command executed. Check sidebar/status.")
        logger.info(f"Executed command: {user_input}")
    else:
        with st.spinner("🤔 Thinking..."):
            result = chat.rag_framework.query(user_input, chat.current_lob)
            st.session_state.history.append({
                "question": user_input,
                "result": result
            })
        logger.info(f"Query executed in LOB '{chat.current_lob}': {user_input}")
        logger.info(f"Query success: {result['success']}")
        if result["success"]:
            logger.info(f"Answer: {result['answer']}")
        else:
            logger.error(f"Error: {result['error']}")
        st.rerun()
