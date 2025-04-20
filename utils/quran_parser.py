from typing import List
from langchain.schema import Document

def load_and_chunk_quran(filepath: str, chunk_size: int = 5) -> List[Document]:
    """
    Function to load Quran text from a file and split it into chunks.
    
    Args:
        filepath (str): Path to the file containing Quran text.
        chunk_size (int, optional): Number of verses (ayahs) per chunk. Defaults to 5.
        
    Returns:
        List[Document]: A list of Document objects, each containing a chunk of text with metadata.
    """
    documents = []  # List to store all Document objects
    current_chunk = []  # Temporary storage for the current chunk being built
    current_metadata = {"surah": None, "ayah_range": None}  # Metadata for the current chunk
    
    # Open and read the file
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        if line.strip() == "":
            continue  # Skip empty lines
        
        try:
            # Parse line - expected format: "surah|ayah|text"
            surah, ayah, text = line.strip().split("|", maxsplit=2)
        except ValueError:
            continue  # Skip lines that don't match the expected format
        
        # Add the verse text to the current chunk
        current_chunk.append(text.strip())
        
        # If this is the first verse in the chunk, update metadata
        if len(current_chunk) == 1:
            start_ayah = ayah  # Record the starting ayah number
            current_metadata["surah"] = int(surah)  # Set the surah number
        
        # If chunk is full or this is the last line, create a Document
        if len(current_chunk) == chunk_size or idx == len(lines) - 1:
            end_ayah = ayah  # Record the ending ayah number
            current_metadata["ayah_range"] = f"{start_ayah}-{end_ayah}"  # Set the ayah range
            
            # Join all verses in the chunk into a single text
            chunk_text = " ".join(current_chunk)
            
            # Create a Document object with text and metadata
            doc = Document(
                page_content=chunk_text,
                metadata=current_metadata.copy()  # Use .copy() to avoid reference issues
            )
            documents.append(doc)
            
            # Reset for the next chunk
            current_chunk = []
    
    return documents