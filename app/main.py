"""
Main CLI Application for Annual Report Analyzer
Command-line interface for document analysis and querying.
"""

import os
import argparse
from dotenv import load_dotenv
from rag_system import AnnualReportRAG


def main():
    """Main CLI application."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Analyze company annual reports with RAG-based Q&A"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to PDF annual report"
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Load existing vector store instead of creating new one"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask (for non-interactive mode)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch web UI with Gradio"
    )
    
    args = parser.parse_args()
    
    # Launch web UI if requested
    if args.ui:
        import subprocess
        print("Launching Streamlit web interface...")
        subprocess.run(["streamlit", "run", "streamlit_app.py"])
        return
    
   
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file with your Google API key.")
        return
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = AnnualReportRAG(google_api_key=api_key)
    
    # Load or create vector store
    if args.load_existing:
        try:
            rag.load_existing_vectorstore()
            print("Loaded existing vector store.")
        except ValueError as e:
            print(f"Error: {e}")
            return
    elif args.pdf:
        print(f"Processing PDF: {args.pdf}")
        rag.load_and_index_documents([args.pdf])
    else:
        print("Error: Please provide --pdf or --load-existing")
        return
    
    # Single question mode
    if args.question:
        result = rag.ask_question(args.question)
        print("\n" + "="*80)
        print(f"Question: {result['question']}")
        print("="*80)
        print(f"\nAnswer:\n{result['answer']}\n")
        print("="*80)
        print("\nSource Documents:")
        for i, doc in enumerate(result['source_documents'], 1):
            page = doc.metadata.get('page', 'Unknown')
            print(f"\n[Source {i}] Page {page}:")
            print(doc.page_content[:200] + "...")
        return
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*80)
        print("Interactive Annual Report Analyzer")
        print("="*80)
        print("Ask questions about the annual report. Type 'quit' or 'exit' to exit.\n")
        
        while True:
            try:
                question = input("Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                result = rag.ask_question(question)
                
                print("\n" + "-"*80)
                print(f"Answer:\n{result['answer']}")
                print("-"*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("\nUse --interactive for interactive mode or --question 'your question' for single query.")
        print("Or use --ui to launch the web interface.")


if __name__ == "__main__":
    main()
