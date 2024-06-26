import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from together import Together


TOGETHER_API_KEY = st.secrets["together"]["api_key"]


def scrape_wiki_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find_all('p')
    text = ' '.join([para.text for para in content])
    return text


def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def embed_chunks(chunks):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(chunks).toarray()
    return embeddings, vectorizer


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index


def retrieve_top_k_chunks(question, vectorizer, index, chunks, k=3):
    question_embedding = vectorizer.transform([question]).toarray()
    D, I = index.search(question_embedding.astype(np.float32), k)
    top_k_chunks = [chunks[i] for i in I[0]]
    return top_k_chunks


def generate_answer(question, context):
    system_message = """ 
    You are not an AI language model.
    Answer only from the chunks"""
    
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


def generate_summary(answer):
    system_message = """ 
    You are not an AI language model.
    Summarize the given answer"""
    
    messages = [{"role": "system", "content": system_message}]
    prompt = f"Summarize the following answer:\n{answer}"
    messages.append({"role": "user", "content": prompt})

    together_client = Together(api_key=TOGETHER_API_KEY)
    
    try:
        response = together_client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def main():
    st.title("Qilo Q&A App")

    url = "https://en.wikipedia.org/wiki/Luke_Skywalker"
    text = scrape_wiki_page(url)
    chunks = chunk_text(text)
    embeddings, vectorizer = embed_chunks(chunks)
    index = create_faiss_index(embeddings)

    question = st.text_input("Ask a question about Luke Skywalker:")
    if question:
        top_k_chunks = retrieve_top_k_chunks(question, vectorizer, index, chunks)
        context = ' '.join(top_k_chunks)
        answer = generate_answer(question, context)
        st.write("**Answer:**", answer)

        
        summary = generate_summary(answer)
        st.write("**Summary:**", summary)

if __name__ == "__main__":
    main()
