import os
from dotenv import load_dotenv
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
import chromadb

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Load and split documents
loader = TextLoader("docs.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Initialize embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Initialize Chroma client and vectorstore
import chromadb

# Use PersistentClient (for saving data)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Then use the client with LangChain’s Chroma vectorstore
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="rag_demo",
    client=chroma_client
)
# Create retriever from vectorstore
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

# RAG chain using Runnable
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableMap({
        "context": lambda x: format_docs(x["input_documents"]),
        "question": lambda x: x["question"]
    })
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("🧠 Retrieval-Augmented QA System")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        docs = retriever.get_relevant_documents(query)
        response = rag_chain.invoke({"input_documents": docs, "question": query})

        st.subheader("🧠 Answer:")
        st.markdown(response)

        st.subheader("📄 Source Documents:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            with st.expander(f"Source {i} — 📘 {source}"):
                st.markdown(doc.page_content)

        st.success("Done!")



