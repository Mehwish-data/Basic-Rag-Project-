import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

try:
    embedding = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://localhost:11434"
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    logger.info("Retriever initialized with Chroma vector store.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2
    )

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

    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )
    
    logger.info("RAG system initialized successfully")

except Exception as e:
    logger.error(f"Error initializing RAG system: {e}")
    # Initialize with None for graceful error handling
    retriever = None
    stuff_chain = None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query_documents():
    """Handle document queries"""
    try:
        if not retriever or not stuff_chain:
            return jsonify({
                'error': 'RAG system not properly initialized. Please check your configuration.'
            }), 500

        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
        
        question = data['question'].strip()
        
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400

        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            return jsonify({
                'answer': 'No relevant documents found for your query.',
                'sources': []
            })
        
        # Generate response
        response = stuff_chain.run(input_documents=docs, question=question)
        
        # Format sources
        sources = []
        for i, doc in enumerate(docs, 1):
            sources.append({
                'id': i,
                'content': doc.page_content[:500] + ('...' if len(doc.page_content) > 500 else ''),
                'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
            })
        
        return jsonify({
            'answer': response,
            'sources': sources
        })
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': 'An error occurred while processing your query'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_system': 'initialized' if retriever and stuff_chain else 'not initialized'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)