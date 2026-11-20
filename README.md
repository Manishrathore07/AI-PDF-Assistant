# AI PDF Assistant

An AI-powered PDF chatbot built using **Retrieval-Augmented Generation (RAG)** that allows users to upload PDF documents and ask questions about their content. It retrieves relevant information from the uploaded document and uses Google Gemini to generate context-aware answers with page and chunk citations.

## Features

- **PDF Upload:** Upload and process PDF documents.
- **Text Extraction:** Extract text and page information using PyPDF.
- **Intelligent Chunking:** Divide extracted text into meaningful, heading-aware chunks.
- **Text Embeddings:** Generate vector embeddings using Sentence Transformers.
- **Semantic Search:** Retrieve relevant document sections using FAISS.
- **Keyword Search:** Find relevant information using BM25.
- **Hybrid Retrieval:** Combine semantic and keyword search using Reciprocal Rank Fusion (RRF).
- **Query Expansion:** Generate alternative search queries to improve retrieval.
- **AI-Powered Answers:** Generate document-grounded responses using Google Gemini.
- **Source Citations:** Display page and chunk references to help verify answers.
- **Conversational Chat:** Support follow-up questions with chat history.
- **Document Insights:** Generate document summaries and topic insights.
- **Interactive UI:** Use the application through a Streamlit interface.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| Google Gemini API | AI-powered answer generation and query expansion |
| Sentence Transformers | Text embedding generation |
| FAISS | Semantic vector search |
| BM25 | Keyword-based retrieval |
| Reciprocal Rank Fusion | Combining retrieval results |
| PyPDF | PDF text extraction |
| NumPy | Numerical operations |
| python-dotenv | Environment variable management |

## How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

1. **PDF Upload:** The user uploads a PDF document.
2. **Text Extraction:** PyPDF extracts text from the document.
3. **Text Chunking:** The extracted text is divided into smaller, meaningful chunks.
4. **Embedding Generation:** Sentence Transformers converts chunks into numerical vector representations.
5. **Semantic Retrieval:** FAISS searches for chunks that are semantically similar to the user's question.
6. **Keyword Retrieval:** BM25 identifies chunks containing relevant keywords.
7. **Hybrid Search:** Reciprocal Rank Fusion combines the results from both retrieval methods.
8. **Query Expansion:** Gemini generates alternative versions of the user's question to improve information retrieval.
9. **Answer Generation:** Relevant retrieved context is provided to Gemini to generate an answer.
10. **Source Citations:** The application displays page and chunk references alongside the generated answer.

## RAG Architecture

```text
             PDF Document
                   |
                   v
             Text Extraction
                   |
                   v
              Text Chunking
                   |
                   v
            Embedding Generation
                   |
                   v
              Vector Store
                   |
                   |
User Question -----+
                   |
                   v
              Query Expansion
                   |
                   v
          +--------+--------+
          |                 |
          v                 v
      FAISS Search       BM25 Search
   (Semantic Search)   (Keyword Search)
          |                 |
          +--------+--------+
                   |
                   v
             Hybrid Search
                   |
                   v
          Relevant PDF Chunks
                   |
                   v
            Google Gemini
                   |
                   v
          Answer with Citations
```

## Project Structure

```text
AI-PDF-Assistant/
│
├── app.py                  # Main Streamlit application
├── main.py                 # Initial project entry point
├── chunking.py             # Text chunking
├── embeddings.py            # Embedding generation
├── build_vector_store.py    # FAISS vector store creation
├── search_vector_store.py   # Vector store search
├── similarity_search.py     # Similarity search
├── faiss_search.py          # FAISS search functionality
├── rag_pipeline.py          # RAG pipeline components
├── test_gemini.py           # Gemini API testing
│
├── data/
│   └── sample.pdf
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── .env                     # Local environment file (not uploaded)
```

*The virtual environment (`venv/`) is created locally during installation and is not included in the repository.*

## Installation and Setup

### Prerequisites

- Python 3.12 recommended
- Git
- Google Gemini API key

### 1. Clone the repository

```bash
git clone https://github.com/Manishrathore07/AI-PDF-Assistant.git
```

Navigate into the project directory:

```bash
cd AI-PDF-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Gemini API key

Create a file named `.env` in the root project directory.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/).

**Important:** Never share your actual API key or upload your `.env` file to GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal. By default:

```text
http://localhost:8501
```

## Key Concepts Explored

Through this project, I explored and implemented:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Text embeddings
- Vector similarity search
- FAISS vector indexing
- BM25 keyword retrieval
- Hybrid search and Reciprocal Rank Fusion
- Query expansion
- Context-aware answer generation
- Source attribution and citation validation

## Future Improvements

- Support for multiple PDF documents.
- More advanced reranking techniques.
- Improved retrieval accuracy for complex questions.
- Support for additional document formats.
- Deployment as a publicly accessible web application.

## Project Goal

The goal of this project was to gain practical experience with RAG systems, vector search, information retrieval, embeddings, and Large Language Models by building an AI-powered application that can answer questions from user-provided PDF documents.

---

**Built with Python, Streamlit, FAISS, Sentence Transformers, BM25, and Google Gemini.**
