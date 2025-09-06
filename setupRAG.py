#!/usr/bin/env python3
"""
Setup complete RAG system with OpenAI API.
Creates vector store, retriever, and QA chain using ChromaDB and OpenAI models.
"""

import os
from pathlib import Path
from setupLangchain import load_chunked_files

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
from langchain.vectorstores import utils as chromautils
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def setup_rag_system():
    """Setup complete RAG system with OpenAI models."""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return None
    
    print("Loading chunked documents...")
    # Load LangChain documents from chunked files
    documents = load_chunked_files()
    
    if not documents:
        print("No documents found. Please run chunking.py first.")
        return None
        
    print(f"Loaded {len(documents)} documents")
    
    # Filter complex metadata for ChromaDB compatibility
    print("Filtering metadata for ChromaDB...")
    docs = chromautils.filter_complex_metadata(documents)
    
    # Setup embeddings (OpenAI)
    print("Setting up OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # More cost-effective than ada-002
    )
    
    # Create vector store
    print("Creating ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory="./chroma_db"  # Persist the database
    )
    
    # Create retriever
    print("Setting up retriever...")
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}  # Retrieve top 3 most similar chunks
    )
    
    # Setup LLM (OpenAI GPT)
    print("Setting up OpenAI LLM...")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # Cost-effective option
        temperature=0.2,        # Low temperature for factual responses
        max_tokens=200          # Reasonable response length
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
        return_source_documents=True  # Include source documents in response
    )
    
    print("RAG system setup complete!")
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
    """Main function to setup and test RAG system."""
    print("Setting up RAG system with OpenAI...")
    
    # Setup RAG system
    qa_chain = setup_rag_system()
    
    if not qa_chain:
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


def get_rag_chain():
    """Utility function to get RAG chain for import by other modules."""
    return setup_rag_system()


if __name__ == "__main__":
    main()