# backend/qa_chain.py

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from backend.retriever import get_quran_retriever

def get_qa_chain():
    """
    Creates and returns a question answering chain that combines retrieval 
    and generation components.
    
    Returns:
        A RetrievalQA chain that can answer questions about the Quran.
    """
    # Get the Quran retriever
    retriever = get_quran_retriever()

    # Initialize a language model that supports text generation
    # Choose a model from the list below based on performance and availability

    # Model 1: GPT-2 (common and free)
    llm = HuggingFaceHub(
        repo_id="gpt2",  # A commonly used free model
        model_kwargs={"temperature": 0.7, "max_length": 256}
    )

    # Model 2: DistilGPT-2 (faster, smaller version)
    # llm = HuggingFaceHub(
    #     repo_id="distilgpt2",  # Smaller, faster version of GPT-2
    #     model_kwargs={"temperature": 0.7, "max_length": 256}
    # )

    # Model 3: Mistral-7B (more advanced and higher quality)
    # llm = HuggingFaceHub(
    #     repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # High quality, might need more resources
    #     model_kwargs={"temperature": 0.7, "max_length": 256}
    # )

    # Model 4: Falcon-7B (high performance, better for larger tasks)
    # llm = HuggingFaceHub(
    #     repo_id="tiiuae/falcon-7b",  # Strong performance for large-scale tasks
    #     model_kwargs={"temperature": 0.7, "max_length": 256}
    # )

    # Model 5: Flan-T5 Base (can be used for text generation tasks)
    # llm = HuggingFaceHub(
    #     repo_id="google/flan-t5-base",  # Google Flan-T5 model
    #     model_kwargs={"temperature": 0.7, "max_length": 256}
    # )

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain
