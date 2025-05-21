# AI Knowledge Q&A App

## Project Overview

This is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and LangChain. It answers questions related to AI, ML, NLP, and LLMs by retrieving relevant information from a CSV dataset.

---

## Features

- Loads and processes data from `AI_cleaned.csv`.
- Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) and FAISS for efficient retrieval.
- Streamlit-based user interface for asking questions and displaying answers.
- Provides sample questions and a downloadable Q&A Excel file.

---

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```
2. Create and activate a virtual environment:
    - On Mac/Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```
    - On Windows:
      ```powershell
      python -m venv venv
      .\venv\Scripts\activate
      ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the app locally:
    ```bash
    streamlit run app.py
    ```

---

## Usage

- Enter AI/ML/NLP-related questions in the input box.
- View answers retrieved from the dataset.
- Use the checkbox to view and download sample Q&A pairs.

---

## Files

- `app.py`: Streamlit chatbot application.
- `AI_cleaned.csv`: Dataset used for retrieval.
- `sample_qa_responses.xlsx`: Sample Q&A export.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

---

## Deployed App

You can access the live app here:  
[AI Knowledge Q&A App](https://ai-chat-bot-235.streamlit.app/)




