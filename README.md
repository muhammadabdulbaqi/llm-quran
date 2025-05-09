# 🕌 Quran Question Answering App (LLM + RAG)

This project is a simple Q&A system for the Quran using Retrieval-Augmented Generation (RAG). It leverages LangChain, a HuggingFace LLM (e.g., GPT2 or similar), and a Quran retriever built from indexed verses and metadata.

## 💡 Features

- Users can ask natural language questions about the Quran.
- Answers are generated based on retrieved Quranic verses using an LLM.
- Source verses are provided to support the answer.

## 🔧 Project Structure

```
quran_rag_app/
├── backend/
│   ├── qa_chain.py           # Builds the QA chain (retriever + LLM)
│   ├── retriever.py          # Loads vectorstore retriever
│   ├── utils.py              # Shared helpers
│
├── data/                     # Quran text and embedding files
│
├── app.py                    # Streamlit frontend
├── requirements.txt
└── README.md
```

## 🛠️ Current Stack

- **Frontend**: Streamlit
- **LLM**: HuggingFaceHub (e.g., gpt2, optionally mistralai/Mistral-7B-Instruct-v0.1, etc.)
- **Embedding & Retrieval**: FAISS vectorstore via LangChain
- **Data**: Quran verses with metadata (English)

## ⚠️ Known Limitations

### ❌ Response Quality
Current LLM (gpt2) is limited in capability and often produces:
- Unclear or hallucinated answers
- Redundant or irrelevant content
- Context isn't always well integrated into the final answer.

### ❌ Cluttered UI
- Answer formatting is verbose and not user-friendly.
- Source documents are displayed inline with no option to hide or expand.

### ❌ One-time Use
- Users can ask only one question per session.
- No easy way to ask a follow-up or new question without restarting the app.

## 🧭 Next Steps

We plan to improve the app in the following ways:

### ✨ UX/UI Improvements
- Add the ability for users to ask multiple questions.
- Add a toggle or expandable section for source verses to reduce clutter.
- Clean up and format answers to be more readable and relevant.

### 🔁 Model & Logic Enhancements
- Try better-performing free LLMs (e.g., mistralai/Mistral-7B-Instruct-v0.1 or tiiuae/falcon-7b) for richer answers.
- Improve retrieval quality to ensure more relevant verses are passed to the LLM.

## ✅ Getting Started

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

You'll need a HuggingFace API token to use the LLM.
