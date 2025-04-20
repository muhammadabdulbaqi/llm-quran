# backend/retriever.py

from langchain.vectorstores import FAISS  # Import FAISS for vector similarity search
from langchain.embeddings import HuggingFaceEmbeddings  # Import for text embeddings

def get_quran_retriever(index_path: str = "quran_faiss_index"):
    """
    Creates and returns a retriever for semantic search over Quran text.
    
    Args:
        index_path (str): Path to the directory containing the saved FAISS index.
                         Defaults to "quran_faiss_index".
    
    Returns:
        A retriever object that can be used to find relevant Quran passages.
    """
    # Load the same embedding model that was used when creating the index
    # Consistency in embedding models is crucial for accurate retrieval
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the previously saved FAISS index from disk
    vectorstore = FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)

    # Create a retriever from the vector store
    # - search_type="similarity" means it will return results based on vector similarity
    # - k=4 means it will return the top 4 most similar passages
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever