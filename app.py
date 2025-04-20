# app.py
import streamlit as st
from dotenv import load_dotenv
from backend.qa_chain import get_qa_chain

# Load environment variables
load_dotenv()  # This loads the .env file

# Initialize QA chain
qa = get_qa_chain()

# Streamlit UI components
st.title("Quran Q&A Chatbot")
st.write("Ask a question, and get an answer from the Quran!")

# Input box for user to ask a question
user_input = st.text_input("Your Question:")

# Check if user has entered a question
if user_input:
    with st.spinner("Thinking..."):
        # Use qa() or qa.invoke() instead of qa.run()
        # This properly handles chains that return multiple outputs
        response = qa({"query": user_input})
        
        # Now properly extract the answer and sources
        answer = response["result"]
        sources = response["source_documents"]
        
        # Display the answer
        st.write("Answer:")
        st.write(answer)
        
        # Display the source documents
        st.write("Source Documents:")
        for doc in sources:
            st.write(f"📖 Surah {doc.metadata['surah']}, Ayat {doc.metadata['ayah_range']}:")
            st.write(doc.page_content)
            st.write("---")