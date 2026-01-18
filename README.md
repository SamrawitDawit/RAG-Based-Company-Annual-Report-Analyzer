# RAG Pipeline Notebook

This project includes an interactive Jupyter Notebook (`rag_pipeline.ipynb`) that implements a Retrieval-Augmented Generation (RAG) pipeline for analyzing company annual reports using Google's Gemini models and ChromaDB.

## Features

- **PDF Processing**: Loads, cleans, and chunks PDF documents.
- **RAG System**: Uses HuggingFace embeddings and ChromaDB for vector storage.
- **LLM Integration**: Connects to Google Gemini for generating accurate answers based on the report context.
- **Financial Focus**: Specialized prompts to ensure factual accuracy with extraction of financial figures.

## Prerequisites

- Python 3.8+
- A Google Cloud API Key for Gemini.

## Setup

1. **Environment Variables**:
   Create a `.env` file in the root directory and add your Google API key (refer to the env example)

2. **Dependencies**:
   The notebook first cell includes a cell to install requirements

## Usage

1. Open `rag_pipeline.ipynb` in VS Code or Jupyter Lab.
2. Run the "Setup & Dependencies" cells to install libraries and load environment variables.
3. The notebook defines two main classes:
   - `PDFProcessor`: Handles text extraction and cleaning.
   - `AnnualReportRAG`: Manages the embedding, storage, and retrieval process.
4. **Main Execution**:
   - Update the `pdf_path` variable in the final cells to point to your target PDF annual report.
   - Run the cells to index the document and ask financial questions.
