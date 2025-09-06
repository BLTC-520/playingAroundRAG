# convert Unstructured elements to LangChain documents
import json
from pathlib import Path
from langchain_core.documents import Document

def load_chunked_files():
    """Load all chunked JSON files and convert to LangChain Documents."""
    chunked_dir = Path(__file__).parent / 'chunked_output'
    documents = []
    
    for json_file in chunked_dir.glob('*.chunks.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            metadata["source"] = metadata.get("original_filename", json_file.stem)
            if "languages" in metadata:
                del metadata["languages"]
            
            documents.append(Document(
                page_content=chunk['text'], 
                metadata=metadata
            ))
    
    return documents

# Usage: documents = load_chunked_files()

if __name__ == "__main__":
    documents = load_chunked_files()
    print(f"Loaded {len(documents)} LangChain documents from chunked files")
    
    if documents:
        print(f"First document preview:")
        print(f"Source: {documents[0].metadata.get('source')}")
        print(f"Content: {documents[0].page_content[:200]}...")
        print("\nDocuments ready for RAG!")