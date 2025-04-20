# This system path modification is necessary to import modules from parent directory
# It adds the parent directory of the current file to Python's import path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our custom Quran parsing function from the utils directory
from utils.quran_parser import load_and_chunk_quran

# Import libraries needed for vector embeddings and storage
from langchain.vectorstores import FAISS  # FAISS is a library for efficient similarity search
from langchain.embeddings import HuggingFaceEmbeddings  # For generating text embeddings
from utils.quran_parser import load_and_chunk_quran  # Repeated import (could be removed)

def build_and_save_vectorstore(quran_path: str, index_path: str = "quran_faiss_index", chunk_size: int = 5):
   """
   Build a vector database from Quran text and save it for later retrieval.
   
   Args:
       quran_path (str): Path to the Quran text file
       index_path (str): Directory where the FAISS index will be saved
       chunk_size (int): Number of verses to include in each chunk
   """
   print("📖 Loading and chunking Quran text...")
   documents = load_and_chunk_quran(quran_path, chunk_size=chunk_size)  # Load and split the text

   print("🔍 Generating embeddings (this may take a minute)...")
   # Initialize the embedding model - MiniLM is a lightweight but effective model
   embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

   print("🧠 Building FAISS vector store...")
   # Convert documents into vector embeddings and store in FAISS index
   vectorstore = FAISS.from_documents(documents, embeddings)

   print(f"💾 Saving FAISS index to {index_path}/")
   # Save the vector store to disk for later use
   vectorstore.save_local(index_path)

# This conditional ensures the code only runs when the script is executed directly
# (not when imported as a module)
if __name__ == "__main__":
   build_and_save_vectorstore("data/en.ahmedali.txt")  # Default path to English Quran translation