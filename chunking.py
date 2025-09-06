#!/usr/bin/env python3
"""
Enhanced chunking script for processing Unstructured JSON output files.
Uses the by_title strategy optimized for financial/government documents.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    Element, Title, NarrativeText, Text, 
    ListItem, Table, CompositeElement, ElementMetadata
)


class ChunkingConfig:
    """Configuration for chunking parameters."""
    
    def __init__(
        self,
        max_characters: int = 1000,
        new_after_n_chars: int = 800,
        combine_text_under_n_chars: int = 200,
        overlap: int = 50,
        overlap_all: bool = False,
        multipage_sections: bool = True
    ):
        self.max_characters = max_characters
        self.new_after_n_chars = new_after_n_chars
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.overlap = overlap
        self.overlap_all = overlap_all
        self.multipage_sections = multipage_sections


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chunking.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def json_to_elements(json_data: List[Dict[str, Any]]) -> List[Element]:
    """Convert JSON data back to Unstructured Element objects."""
    elements = []
    
    # Map element types to their classes
    element_type_map = {
        'Title': Title,
        'NarrativeText': NarrativeText,
        'UncategorizedText': Text,  # Map to generic Text element
        'ListItem': ListItem,
        'Table': Table,
        'CompositeElement': CompositeElement,
        'Text': Text
    }
    
    for item in json_data:
        element_type = item.get('type')
        text = item.get('text', '')
        metadata_dict = item.get('metadata', {})
        element_id = item.get('element_id')
        
        # Create ElementMetadata object
        metadata = ElementMetadata(
            filename=metadata_dict.get('filename'),
            filetype=metadata_dict.get('filetype'),
            languages=metadata_dict.get('languages', []),
            page_number=metadata_dict.get('page_number'),
            coordinates=metadata_dict.get('coordinates')
        )
        
        # Get the appropriate element class, default to NarrativeText
        ElementClass = element_type_map.get(element_type, NarrativeText)
        
        # Create the element
        element = ElementClass(text=text, metadata=metadata)
        # Note: element_id is read-only, so we can't set it directly
        # The chunking process will generate new IDs anyway
        
        elements.append(element)
    
    return elements


def chunk_elements_by_title(elements: List[Element], config: ChunkingConfig) -> List[Element]:
    """Chunk elements using the by_title strategy."""
    return chunk_by_title(
        elements,
        max_characters=config.max_characters,
        new_after_n_chars=config.new_after_n_chars,
        combine_text_under_n_chars=config.combine_text_under_n_chars,
        overlap=config.overlap,
        overlap_all=config.overlap_all,
        multipage_sections=config.multipage_sections
    )


def save_chunks(chunks: List[Element], output_file: Path, original_filename: str) -> None:
    """Save chunked elements to JSON and text files."""
    # Save as JSON
    json_output = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            'chunk_id': i + 1,
            'text': chunk.text,
            'type': chunk.__class__.__name__,
            'metadata': {
                'original_filename': original_filename,
                'chunk_index': i + 1,
                'character_count': len(chunk.text),
                'word_count': len(chunk.text.split()),
                'created_at': datetime.now().isoformat()
            }
        }
        
        # Add original elements info if available
        if hasattr(chunk.metadata, 'orig_elements') and chunk.metadata.orig_elements:
            chunk_data['metadata']['original_elements_count'] = len(chunk.metadata.orig_elements)
            chunk_data['metadata']['original_element_types'] = [
                elem.__class__.__name__ for elem in chunk.metadata.orig_elements
            ]
        
        json_output.append(chunk_data)
    
    # Save JSON file
    json_file = output_file.with_suffix('.chunks.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    # Save text file for easy reading
    txt_file = output_file.with_suffix('.chunks.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"CHUNKED DOCUMENT: {original_filename}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, chunk in enumerate(chunks):
            f.write(f"CHUNK {i + 1}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Characters: {len(chunk.text)}\n")
            f.write(f"Words: {len(chunk.text.split())}\n")
            f.write(f"Type: {chunk.__class__.__name__}\n\n")
            f.write(chunk.text)
            f.write("\n\n" + "=" * 80 + "\n\n")


def process_file(input_file: Path, output_dir: Path, config: ChunkingConfig, logger: logging.Logger) -> bool:
    """Process a single JSON file for chunking."""
    try:
        logger.info(f"Processing file: {input_file.name}")
        
        # Load JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if not json_data:
            logger.warning(f"Empty JSON file: {input_file.name}")
            return False
        
        # Convert JSON to elements
        elements = json_to_elements(json_data)
        logger.info(f"Converted {len(elements)} elements from JSON")
        
        # Chunk the elements
        chunks = chunk_elements_by_title(elements, config)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Save results
        output_file = output_dir / input_file.stem
        save_chunks(chunks, output_file, input_file.name)
        
        logger.info(f"Saved chunks for {input_file.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_file.name}: {str(e)}")
        return False


def main():
    """Main chunking process."""
    logger = setup_logging()
    logger.info("Starting chunking process")
    
    # Setup directories
    current_dir = Path(__file__).parent
    input_dir = current_dir / 'output'
    output_dir = current_dir / 'chunked_output'
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    config = ChunkingConfig(
        max_characters=1000,      # Larger chunks for better context
        new_after_n_chars=800,    # Preferred chunk size
        combine_text_under_n_chars=200,  # Combine small sections
        overlap=50,               # Small overlap for continuity
        overlap_all=False,        # Only overlap split chunks
        multipage_sections=True   # Preserve cross-page sections
    )
    
    logger.info(f"Configuration: max_chars={config.max_characters}, "
                f"new_after={config.new_after_n_chars}, "
                f"combine_under={config.combine_text_under_n_chars}")
    
    # Process all JSON files
    json_files = list(input_dir.glob('*.json'))
    
    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    success_count = 0
    for json_file in json_files:
        if process_file(json_file, output_dir, config, logger):
            success_count += 1
    
    logger.info(f"Chunking complete! Processed {success_count}/{len(json_files)} files successfully")
    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()