#!/usr/bin/env python3
"""
Query interface for the RAG system.
Loads existing ChromaDB and provides interactive Q&A without rebuilding the system.
"""

import os
from pathlib import Path

# Load .env file if it exists
def load_env_file():
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Load environment variables
load_env_file()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_existing_rag_system():
    """Load existing RAG system from persisted ChromaDB."""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please add it to your .env file")
        return None
    
    # Check if ChromaDB exists
    chroma_path = Path("./chroma_db")
    if not chroma_path.exists():
        print("Error: ChromaDB not found. Please run setupRAG.py first to create the vector database.")
        return None
    
    print("Loading existing ChromaDB vector store...")
    
    # Setup embeddings (must match the ones used during creation)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    # Load existing vector store
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Create retriever
    print("Setting up retriever...")
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )
    
    # Setup LLM (OpenAI GPT)
    print("Setting up OpenAI LLM...")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=1000
    )
    
    # Create prompt template
    prompt_template = """
You are an assistant for answering questions using provided context.
You are given the extracted parts of documents and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.

Question: {question}
Context: {context}

Answer:"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    
    # Create RAG chain
    print("Creating RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    print("RAG system loaded successfully!")
    return qa_chain


def ask_question(qa_chain, question):
    """Ask a question to the RAG system."""
    if not qa_chain:
        print("RAG system not initialized")
        return None
        
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    try:
        result = qa_chain.invoke({"query": question})
        
        print(f"Answer: {result['result']}")
        print("\nSources:")
        for i, doc in enumerate(result['source_documents'], 1):
            source = doc.metadata.get('source', 'Unknown')
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"{i}. {source}: {preview}")
        
        return result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def main():
    """Main function for interactive querying."""
    print("Loading RAG system...")
    
    # Load existing RAG system
    qa_chain = load_existing_rag_system()
    
    if not qa_chain:
        print("\nSetup Instructions:")
        print("1. Run 'python setupRAG.py' first to create the vector database")
        print("2. Ensure your .env file contains OPENAI_API_KEY")
        return
    
    # Interactive Q&A loop
    print("\n" + "="*60)
    print("RAG System Ready! Ask questions about your documents.")
    print("Type 'quit' to exit.")
    print("="*60)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not question:
            continue
            
        ask_question(qa_chain, question)


if __name__ == "__main__":
    main()