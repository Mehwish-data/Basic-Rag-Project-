import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import streamlit as st
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Load Google API Key from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Initialize vector store retriever
vectorstore = Chroma(
    embedding_function=embedding,
    collection_name="rag_demo"
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemma-3n-e2b-it",
    temperature=0.2
)

# Prompt template
prompt_template = """You are a helpful assistant. Use the following context to answer the question:

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate.from_template(prompt_template)

# Build chains
llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"
)

# Streamlit UI
st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("🧠 Retrieval-Augmented QA System")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        docs = retriever.get_relevant_documents(query)
        response = stuff_chain.run(input_documents=docs, question=query)

        st.subheader("🧠 Answer:")
        st.markdown(response)

        st.subheader("📄 Source Documents:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "streamlit run st_app.py --server.port 8502")

            with st.expander(f"Source {i} — 📘 {source}, Page {page}"):
                st.markdown(doc.page_content)
