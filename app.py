import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import faiss
import logging
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_together import Together

# Constants
WIKI_URL = "https://en.wikipedia.org/wiki/Luke_Skywalker"
TOGETHER_API_KEY = st.secrets["together"]["api_key"]

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Scrape Wikipedia page
@st.cache_data
def scrape_wiki_page(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = "\n".join([para.get_text() for para in paragraphs])
        logger.info(f"Scraped content length: {len(content)}")
        return content
    except Exception as e:
        logger.error(f"Error scraping wiki page: {str(e)}")
        return ""

# Chunk the content
@st.cache_data
def chunk_content(content, chunk_size=2):
    try:
        nltk.download('punkt')
        sentences = sent_tokenize(content)
        chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking content: {str(e)}")
        return []

# Store chunks in Faiss vector database using LangChain
@st.cache_resource
def store_chunks_in_faiss(chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
        vector_store = FAISS.from_texts(chunks, embeddings)
        logger.info("Stored chunks in Faiss index with LangChain")
        return vector_store
    except Exception as e:
        logger.error(f"Error storing chunks in Faiss: {str(e)}")
        return None

# Retrieve relevant chunks
def get_relevant_chunks(question, vector_store, k=3):
    try:
        question_embedding = vector_store.embedding_function(question)
        distances, indices = vector_store.index.search(np.array([question_embedding]), k)
        logger.info(f"Retrieved {len(indices[0])} relevant chunks")
        return [vector_store.texts[i] for i in indices[0]]
    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {str(e)}")
        return []

# Generate answer using Together API with Mistral model
def generate_answer(question, context):
    system_message = """ 
    You are not an AI language model.
    Answer only from chunks"""
    
    messages = [{"role": "system", "content": system_message}]
    prompt = f"{question}\n{context}"
    messages.append({"role": "user", "content": prompt})

    together_client = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        together_api_key=TOGETHER_API_KEY
    )
    
    try:
        response = together_client.invoke(prompt)
        logger.info("Generated answer successfully")
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"

# Main Streamlit app
st.title("QuestQuarry")
st.write("Ask any question about Luke Skywalker:")

# Scrape and process the content
content = scrape_wiki_page(WIKI_URL)
if content:
    chunks = chunk_content(content)
    if chunks:
        vector_store = store_chunks_in_faiss(chunks)

        # User input
        question = st.text_input("Your question:")

        if question:
            relevant_chunks = get_relevant_chunks(question, vector_store)
            context = " ".join(relevant_chunks)
            answer = generate_answer(question, context)
            st.write("Answer:", answer)
            
            # Text summarization
            if len(answer) > 100:
                summary = summarizer(answer, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
                st.write("Summary:", summary)
else:
    st.write("Error loading content.")
