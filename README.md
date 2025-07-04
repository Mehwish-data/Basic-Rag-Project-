#  Retrieval-Augmented Generation (RAG) QA System

This project is a **local RAG (Retrieval-Augmented Generation)** pipeline that lets you upload PDF documents, embed their content using **Ollama embeddings**, and query them through a **Streamlit** UI or a **Flask API** (optional).

It uses:

- 🧱 **LangChain** for chaining LLM logic
  
- 🧠 **Ollama** for local text embedding (`nomic-embed-text`)
  
- 🗃️ **ChromaDB** for vector storage and retrieval
  
- 🌐 **Streamlit** for the front-end UI
  
- 🔑 **Gemini 2.0** (via Google GenAI) as the LLM
  
- 🚀 Optional: **Flask** to serve as an API backend


## 📁 Project Structure

(RAG) QA System

├── pdfs/

│ ├── openai.pdf

│ ├── LLaMA.pdf

│ └── NIPS-2017.pdf

├── chroma_db/ <-- auto-generated, don’t upload this

├── create_embedding.py <-- processes and chunks PDFs

├── st_app.py <-- Streamlit app

├── flask_api.py <-- (optional) Flask API endpoint

├── requirements.txt <-- list of dependencies

└── README.md <-- project description



## ⚙️ Setup Instructions

### 1. 🔧 Create and activate a virtual environment

# Optional but recommended
python -m venv rag_env
.\rag_env\Scripts\activate  # On Windows

## 2. 📦 Install dependencies

pip install -r requirements.txt

✅ make sure Ollama is installed and running:
https://ollama.com

## 🧩 How It Works

Step 1: Embed your PDF documents

python create_embedding.py

*Loads all PDFs in the pdfs/ folder

*Chunks and embeds them

*Saves vector DB in chroma_db/

## Step 2: Launch the Streamlit UI

*Ask a question in the web UI

*See the generated answer + source documents

## Optional: Use the Flask API (instead of Streamlit)

*python flask_api.py

*http://localhost:5000/ask
{
  "query": "What is LLaMA?"
}

## ✅ Requirements (requirements.txt)

langchain

langchain_community

langchain_core

langchain_google_genai

langchain_ollama

chromadb

pypdf

protobuf==3.20.3

streamlit

# Optional if using API

Flask

Flask-CORS

## 📌 Notes

chroma_db/ is auto-generated. You can add it to .gitignore.

GOOGLE_API_KEY should be securely managed in real projects.

Ollama must be running (ollama serve) before embedding or querying.

