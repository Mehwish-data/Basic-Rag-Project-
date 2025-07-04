import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize embedding model
embedding = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

# Load Chroma vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
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

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

stuff_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

# Ask a question
query = input("Ask something: ")
docs = retriever.get_relevant_documents(query)

response = stuff_chain.run(input_documents=docs, question=query)

print("\n🧠 Answer:\n", response)
print("\n📄 Source Documents:")
for i, doc in enumerate(docs, 1):
    print(f"\nSource {i}:\n{doc.page_content}")
