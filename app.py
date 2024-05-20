import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np

WIKI_URL = "https://en.wikipedia.org/wiki/Luke_Skywalker"

@st.cache_data
def scrape_wiki_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    content = "\n".join([para.get_text() for para in paragraphs])
    return content

@st.cache_data
def chunk_content(content, chunk_size=2):
    nltk.download('punkt')
    sentences = sent_tokenize(content)
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

@st.cache_resource
def store_chunks_in_faiss(chunks):
    model_name = "sentence-transformers/all-mpnet-base-v2"  # Lighter model
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    chunk_embeddings = model(**tokenizer(chunks, padding=True, return_tensors="pt")).pooler_output.cpu().detach().numpy()
    d = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(chunk_embeddings))
    return index, chunks, model

def get_relevant_chunks(question, index, chunks, model, k=3):
    question_embedding = model(**tokenizer([question], padding=True, return_tensors="pt")).pooler_output.cpu().detach().numpy()
    distances, indices = index.search(question_embedding, k)
    return [chunks[i] for i in indices[0]]

def generate_answer(question, context):
    # No Together integration here, answer directly from context
    answer = f"Based on the provided passage about Luke Skywalker, here's what I found relevant to your question: {context}"
    return answer

st.title("Luke Skywalker Q&A")
st.write("Ask any question about Luke Skywalker:")

content = scrape_wiki_page(WIKI_URL)
chunks = chunk_content(content)
index, chunks, model = store_chunks_in_faiss(chunks)

question = st.text_input("Your question:")

if question:
    relevant_chunks = get_relevant_chunks(question, index, chunks, model)
    context = " ".join(relevant_chunks)
    answer = generate_answer(question, context)
    st.write("Answer:", answer)
