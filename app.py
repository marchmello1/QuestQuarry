import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertModel, BertTokenizer
import faiss
import numpy as np
from together import Together

WIKI_URL = "https://en.wikipedia.org/wiki/Luke_Skywalker"
TOGETHER_API_KEY = st.secrets["together"]["api_key"]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

@st.cache(suppress_st_warning=True)
def scrape_wiki_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    content = "\n".join([para.get_text() for para in paragraphs])
    return content

@st.cache(suppress_st_warning=True)
def chunk_content(content, chunk_size=5):
    nltk.download('punkt')
    sentences = sent_tokenize(content)
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

@st.cache(hash_funcs={tokenizers.AddedToken: my_hash_func})
def store_chunks_in_faiss(chunks):
    chunk_embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True)
        outputs = model(**inputs)
        pooled_output = outputs[1].detach().numpy()
        chunk_embeddings.append(pooled_output[0])
    d = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(chunk_embeddings))
    return index, chunks

def get_relevant_chunks(question, index, chunks, k=3):
    inputs = tokenizer(question, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    question_embedding = outputs[1].detach().numpy()
    distances, indices = index.search(question_embedding, k)
    return [chunks[i] for i in indices[0]]

def generate_answer(question, context):
    system_message = """ 
    You are not an AI language model.
    Answer only from chunks"""
    
    messages = [{"role": "system", "content": system_message}]
    prompt = f"{question}\n{context}"
    messages.append({"role": "user", "content": prompt})

    together_client = Together(api_key=TOGETHER_API_KEY)
    
    try:
        response = together_client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

st.title("Luke Skywalker Q&A")
st.write("Ask any question about Luke Skywalker:")

content = scrape_wiki_page(WIKI_URL)
chunks = chunk_content(content)
index, chunks = store_chunks_in_faiss(chunks)

question = st.text_input("Your question:")

if question:
    relevant_chunks = get_relevant_chunks(question, index, chunks)
    context = " ".join(relevant_chunks)
    answer = generate_answer(question, context)
    st.write("Answer:", answer)
