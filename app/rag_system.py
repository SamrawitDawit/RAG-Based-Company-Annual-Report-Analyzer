"""
RAG System for Annual Report Analysis
Implements retrieval-augmented generation for accurate question answering.
"""

import os
from typing import List, Optional, Any
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pdf_processor import PDFProcessor


class AnnualReportRAG:
    """RAG system for analyzing annual reports with numerical accuracy."""
    
    def __init__(
        self,
        google_api_key: str,
        persist_directory: str = "./chroma_db",
        model_name: str = "gemini-flash-latest"
    ):
        """
        Initialize RAG system.
        
        Args:
            google_api_key: Google API key
            persist_directory: Directory to persist vector store
            model_name: Gemini model to use
        """
        self.google_api_key = google_api_key
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Initialize components
        # Use HuggingFace embeddings (free, local, no API limits)
        print(f"Loading embedding model... (first time may take a minute to download)")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,  # Low temperature for factual accuracy
            google_api_key=google_api_key
        )
        
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain: Optional[Any] = None
        self.retriever: Optional[Any] = None
        self.pdf_processor = PDFProcessor()
        
    def load_and_index_documents(self, pdf_paths: List[str]) -> None:
        """
        Load PDF documents and create vector store.
        
        Args:
            pdf_paths: List of paths to PDF files
        """
        all_chunks = []
        
        for pdf_path in pdf_paths:
            # Load and chunk each PDF
            documents = self.pdf_processor.load_pdf(pdf_path)
            chunks = self.pdf_processor.chunk_documents(documents)
            all_chunks.extend(chunks)
        
        print(f"Total chunks to index: {len(all_chunks)}")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"Vector store created and persisted to {self.persist_directory}")
        
        # Create QA chain
        self._create_qa_chain()
    
    def load_existing_vectorstore(self) -> None:
        """Load existing vector store from disk."""
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"Vector store not found at {self.persist_directory}")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        print(f"Loaded existing vector store from {self.persist_directory}")
        self._create_qa_chain()
    
    def _create_qa_chain(self) -> None:
        """Create the QA chain with custom prompt."""
        
        # Custom prompt for financial accuracy
        template = """You are a financial analyst assistant analyzing company annual reports. 
Your task is to answer questions based ONLY on the provided context from the annual report.

CRITICAL RULES:
1. Use ONLY the exact numbers and figures found in the context
2. NEVER make up, estimate, or calculate numbers that aren't explicitly stated
3. If a specific figure is not in the context, clearly state "This information is not available in the provided context"
4. When citing numbers, quote them exactly as they appear in the source
5. Preserve units (millions, billions, percentages, etc.) exactly as stated
6. If asked about trends or comparisons, only use data explicitly present in the context

Context from annual report:
{context}

Question: {question}

Detailed Answer (with exact figures from the context):"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
        )
        
        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create chain using LCEL (LangChain Expression Language)
        self.qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Store retriever for getting source documents
        self.retriever = retriever
    
    def ask_question(self, question: str) -> dict:
        """
        Ask a question about the annual report.
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Load documents first.")
        
        # Get the answer
        answer = self.qa_chain.invoke(question)
        
        # Get source documents
        source_documents = self.retriever.invoke(question)
        
        return {
            "question": question,
            "answer": answer,
            "source_documents": source_documents
        }
    
    def get_similar_chunks(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve similar document chunks for a query.
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def validate_numerical_answer(self, answer: str, context: str) -> bool:
        """
        Validate that numerical claims in answer are found in context.
        
        Args:
            answer: Generated answer
            context: Source context
            
        Returns:
            True if all numbers in answer are found in context
        """
        # Extract numbers from answer and context
        answer_figures = self.pdf_processor.extract_financial_data(answer)
        context_figures = self.pdf_processor.extract_financial_data(context)
        
        # Check if all answer figures are in context
        for figure in answer_figures:
            if figure not in context_figures:
                return False
        
        return True
