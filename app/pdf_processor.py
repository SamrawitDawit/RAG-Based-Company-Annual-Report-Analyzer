"""
PDF Document Processor for Annual Reports
Handles loading, chunking, and preprocessing of PDF documents.
"""

from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re


class PDFProcessor:
    """Process PDF documents for RAG system."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load and process a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects with text chunks
        """
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")
        
        # Clean and preprocess text
        for doc in documents:
            doc.page_content = self._clean_text(doc.page_content)
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from documents")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep financial symbols
        # Keep: numbers, letters, common punctuation, $, %, etc.
        text = re.sub(r'[^\w\s\$\%\.\,\-\(\)\:\;\/]', '', text)
        
        return text.strip()
    
    def extract_financial_data(self, text: str) -> List[str]:
        """
        Extract financial figures and numbers from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of extracted financial figures
        """
        # Pattern for currency amounts: $123,456.78 or $123.4 million/billion
        currency_pattern = r'\$[\d,]+\.?\d*\s*(?:million|billion|trillion)?'
        
        # Pattern for percentages: 12.5%
        percentage_pattern = r'\d+\.?\d*\s*%'
        
        # Pattern for large numbers with commas
        number_pattern = r'\d{1,3}(?:,\d{3})+(?:\.\d+)?'
        
        figures = []
        figures.extend(re.findall(currency_pattern, text, re.IGNORECASE))
        figures.extend(re.findall(percentage_pattern, text))
        figures.extend(re.findall(number_pattern, text))
        
        return figures
