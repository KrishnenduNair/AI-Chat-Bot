import streamlit as st
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Page setup
st.set_page_config(page_title="AI Q&A App", layout="centered")

# Title
st.title("AI Knowledge Q&A App")
st.markdown("Ask basic questions about AI, ML, NLP, LLMs, and more!")

# Load and embed documents (cache to prevent reloading)
@st.cache_resource
def load_data():
    loader = CSVLoader(file_path="AI_cleaned.csv")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()

retriever = load_data()

# Question input
question = st.text_input("Enter your question:")

if question:
    matches = retriever.get_relevant_documents(question)
    if matches:
        st.success("Answer:")
        st.write(matches[0].page_content)
    else:
        st.warning("No relevant answer found.")

# Optional predefined questions
if st.checkbox("Show sample Q&A"):
    sample_questions = [
        "What is Artificial Intelligence?",
        "What is Machine Learning?",
        "What is Deep Learning?",
        "What is Natural Language Processing?",
        "What are LLMs?",
        "What is the difference between AI and ML?",
        "What is supervised learning?",
        "What is unsupervised learning?",
        "What is reinforcement learning?",
        "What is RAG?",
    ]

    sample_qa = []
    for q in sample_questions:
        matches = retriever.get_relevant_documents(q)
        a = matches[0].page_content if matches else "Not found."
        sample_qa.append({"Question": q, "Answer": a})

    df = pd.DataFrame(sample_qa)
    st.dataframe(df)
    df.to_excel("sample_qa_responses.xlsx", index=False)
    st.download_button("Download Q&A as Excel", data=open("sample_qa_responses.xlsx", "rb"), file_name="sample_qa_responses.xlsx")

