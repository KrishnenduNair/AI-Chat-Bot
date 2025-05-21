import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load and embed
loader = CSVLoader(file_path="AI.csv")
documents = loader.load()

# Smaller chunk size for more precise retrieval
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

def ask(q):
    matches = retriever.get_relevant_documents(q)
    if not matches:
        return "Not found."
    print(f"\nTop 3 matches for query '{q}':")
    for i, match in enumerate(matches[:3]):
        print(f"Match {i+1} content:\n{match.page_content}\n{'-'*50}")
    # Join top 3 chunks to form a fuller answer
    combined_answer = "\n\n".join([m.page_content for m in matches[:3]])
    return combined_answer

qa_pairs = [
    {"Question": "What is Artificial Intelligence?", "Answer": ask("What is Artificial Intelligence?")},
    {"Question": "What is Machine Learning?", "Answer": ask("What is Machine Learning?")},
    {"Question": "What is Deep Learning?", "Answer": ask("What is Deep Learning?")},
    {"Question": "What is Natural Language Processing?", "Answer": ask("What is Natural Language Processing?")},
    {"Question": "What are LLMs?", "Answer": ask("What are LLMs?")},
    {"Question": "What is the difference between AI and ML?", "Answer": ask("What is the difference between AI and ML?")},
    {"Question": "What is supervised learning?", "Answer": ask("What is supervised learning?")},
    {"Question": "What is unsupervised learning?", "Answer": ask("What is unsupervised learning?")},
    {"Question": "What is reinforcement learning?", "Answer": ask("What is reinforcement learning?")},
    {"Question": "What is RAG?", "Answer": ask("What is RAG?")},
]

df = pd.DataFrame(qa_pairs)
df.to_excel("sample_qa_responses.xlsx", index=False)
print("\nQ&A responses saved to sample_qa_responses.xlsx")
