import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Path to your PDF folder
PDF_DIR = "./pdfs"

# Load all PDFs
all_documents = []
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        filepath = os.path.join(PDF_DIR, filename)
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        all_documents.extend(documents)
        print(f" Loaded {len(documents)} pages from: {filename}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(all_documents)
print(f" Split into {len(docs)} chunks.")

# Embeddings setup
embed = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

# Store into Chroma
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embed,
    persist_directory="./chroma_db"
)

vectorstore.persist()
print(" All documents embedded and saved to Chroma.")
