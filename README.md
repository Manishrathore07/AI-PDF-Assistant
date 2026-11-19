# AI PDF Assistant

An AI-powered PDF chatbot that allows users to upload PDF documents and ask questions about their content. The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate answers with source citations.

## Features

- Upload and process PDF documents.
- Extract text from PDFs.
- Intelligent, heading-aware text chunking.
- Generate text embeddings using Sentence Transformers.
- Semantic search using FAISS.
- Keyword search using BM25.
- Hybrid search combining semantic and keyword retrieval.
- Query expansion to improve information retrieval.
- AI-generated answers using Google Gemini.
- Page and chunk citations in answers.
- Follow-up questions and conversational chat history.
- Document summaries and topic insights.
- Interactive interface built with Streamlit.

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- Sentence Transformers
- FAISS
- BM25
- PyPDF
- NumPy

## Project Structure

```text
AI-PDF-Chatbot/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env
└── venv/
```

Note: `.env` and `venv/` are local files and should not be uploaded to GitHub.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-PDF-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Gemini API key

Create a `.env` file in the project folder and add:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/).

Never share or upload your actual API key.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. The user uploads a PDF document.
2. The application extracts text from the PDF.
3. Extracted text is divided into meaningful chunks.
4. Sentence Transformers generates embeddings for the chunks.
5. FAISS performs semantic similarity search.
6. BM25 performs keyword-based search.
7. Hybrid retrieval combines the search results.
8. Google Gemini generates an answer using the retrieved document context.
9. The application displays the answer with page and chunk citations.

## Project Goal

This project was developed to explore Retrieval-Augmented Generation (RAG), semantic search, vector databases, hybrid retrieval, and Large Language Models through a practical application.

---

**Built with Python, Streamlit, FAISS, Sentence Transformers, and Google Gemini.**
