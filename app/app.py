"""
Streamlit Web Interface for Annual Report Analyzer
"""

import streamlit as st
import os
from rag_system import AnnualReportRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Annual Report Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

# Header
st.title("📊 Company Annual Report Analyzer")
st.markdown("""
**Upload an annual report PDF and ask questions to get accurate, citation-backed answers.**

This system uses RAG (Retrieval-Augmented Generation) to ensure numerical accuracy by:
- Retrieving exact figures from the document
- Never inventing or estimating numbers
- Providing source citations for transparency
""")

# Sidebar for file upload and initialization
with st.sidebar:
    st.header("1️⃣ Upload Annual Report")
    
    pdf_file = st.file_uploader(
        "Upload PDF",
        type=['pdf'],
        help="Upload a company annual report in PDF format"
    )
    
    if st.button("Initialize System", type="primary"):
        if pdf_file is None:
            st.error("⚠️ Please upload a PDF file first.")
        else:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                st.error("⚠️ Google API key not found. Please set GOOGLE_API_KEY in .env file.")
            else:
                try:
                    with st.spinner("Initializing RAG system..."):
                        st.session_state.rag_system = AnnualReportRAG(
                            google_api_key=api_key,
                            persist_directory="./chroma_db"
                        )
                    
                    # Save uploaded file temporarily
                    with st.spinner("Loading PDF document..."):
                        temp_pdf_path = f"temp_{pdf_file.name}"
                        with open(temp_pdf_path, "wb") as f:
                            f.write(pdf_file.getvalue())
                    
                    with st.spinner("Creating embeddings and vector store... This may take a few minutes."):
                        st.session_state.rag_system.load_and_index_documents([temp_pdf_path])
                    
                    # Clean up temp file
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)
                    
                    st.session_state.current_pdf = pdf_file.name
                    st.success(f"✅ Successfully loaded and indexed: {pdf_file.name}")
                    st.success("You can now ask questions about the annual report!")
                    
                except Exception as e:
                    st.error(f"❌ Error initializing system: {str(e)}")
    
    if st.session_state.current_pdf:
        st.info(f"📄 Currently loaded: {st.session_state.current_pdf}")

# Main area for Q&A
st.header("2️⃣ Ask Questions")

# Question input
question = st.text_input(
    "Your Question",
    placeholder="e.g., What was the total revenue in 2024?",
    help="Ask any question about the annual report"
)

# Options
show_sources = st.checkbox("Show source documents", value=True)

# Ask button
if st.button("Ask Question", type="primary"):
    if not question or question.strip() == "":
        st.warning("⚠️ Please enter a question.")
    elif st.session_state.rag_system is None:
        st.warning("⚠️ Please initialize the system with a PDF first.")
    else:
        try:
            with st.spinner("Searching for answer..."):
                result = st.session_state.rag_system.ask_question(question)
            
            # Display answer
            st.markdown("### Answer")
            st.write(result['answer'])
            
            # Display sources if requested
            if show_sources and result['source_documents']:
                st.markdown("### Source Documents")
                for i, doc in enumerate(result['source_documents'][:3], 1):
                    page = doc.metadata.get('page', 'Unknown')
                    content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    
                    with st.expander(f"Source {i} (Page {page})"):
                        st.text(content)
                        
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")

# Example questions
st.markdown("### 💡 Example Questions")
examples = [
    "What was the total revenue for the year?",
    "What is the net income or profit?",
    "How much did the company spend on research and development?",
    "What is the operating cash flow?",
    "How many employees does the company have?",
    "What were the total assets at year end?",
    "What is the company's debt-to-equity ratio?",
]

cols = st.columns(2)
for i, example in enumerate(examples):
    with cols[i % 2]:
        if st.button(example, key=f"example_{i}"):
            st.rerun()
