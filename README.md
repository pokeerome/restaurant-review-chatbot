# Restaurant Review Chatbot API

A REST API that answers questions about a pizza restaurant by searching real customer reviews using vector search and OpenAI's GPT-4o-mini. Built with FastAPI, LangChain, and ChromaDB.

🔗 **Live Demo:** https://restaurant-review-chatbot.onrender.com

📖 **Interactive API Docs:** https://restaurant-review-chatbot.onrender.com/docs

## How it works

1. Customer reviews are loaded from a CSV file
2. Each review is converted into a vector embedding using OpenAI's `text-embedding-3-small`
3. Embeddings are stored in a local ChromaDB vector database
4. When you ask a question, it's converted to a vector and compared against all stored reviews
5. The most relevant reviews are retrieved and passed to `gpt-4o-mini`, which generates a grounded answer
6. Every question and answer is saved to a SQLite database for history tracking

This is a Retrieval Augmented Generation (RAG) system that searches by meaning, not just keywords.

## Features

- **REST API** — send questions via HTTP and get JSON responses
- **Vector search** — finds relevant reviews by meaning using OpenAI embeddings and ChromaDB
- **History tracking** — every question and answer saved to SQLite with timestamps
- **Raw data access** — view all reviews directly without going through the AI
- **Auto docs** — FastAPI generates interactive documentation at `/docs`
- **Live deployment** — hosted on Render, accessible from anywhere

## Requirements

- Python 3.10+
- An OpenAI API key with available credits

## Setup

1. Clone the repository
   ```
   git clone https://github.com/pokeerome/restaurant-review-chatbot.git
   cd restaurant-review-chatbot
   ```

2. Create a virtual environment
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```

3. Install dependencies
   ```
   pip install fastapi uvicorn python-dotenv pandas langchain-core langchain-openai langchain-chroma
   ```

4. Create a `.env` file in the root folder
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   MODEL_NAME=gpt-4o-mini
   CSV_FILE=realistic_restaurant_reviews.csv
   ```

5. The sample reviews CSV (`realistic_restaurant_reviews.csv`) is already included

## Usage

### Run locally

```
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/docs` to test all endpoints interactively.

### Use the live version

Visit https://restaurant-review-chatbot.onrender.com/docs to test the deployed API directly in your browser — no setup required.

> Note: the first local run will take longer than usual since every review needs to be converted into an embedding and stored in ChromaDB. After that, the vector database is reused on future runs.

---

### POST /ask

Ask a question and get an answer synthesized from relevant customer reviews.

```
POST https://restaurant-review-chatbot.onrender.com/ask
Content-Type: application/json

{
  "question": "is the pizza good here?"
}
```

Response:

```json
{
  "answer": "Based on the reviews, customers consistently praise the pizza, particularly the crust and the buffalo mozzarella..."
}
```

---

### GET /reviews

Returns all customer reviews directly from the CSV, with no AI involved.

```
GET https://restaurant-review-chatbot.onrender.com/reviews
```

Response:

```json
[
  {
    "title": "Best pizza in town",
    "review": "The crust was perfectly crispy...",
    "rating": 5,
    "date": "2024-01-15"
  }
]
```

---

### GET /history

Returns all previous questions and answers with timestamps.

```
GET https://restaurant-review-chatbot.onrender.com/history
```

Response:

```json
[
  {
    "id": 1,
    "question": "is the pizza good here?",
    "answer": "Based on the reviews, customers consistently praise...",
    "timestamp": "2026-06-18T12:49:09.044496"
  }
]
```

---

## Project structure

```
restaurant-review-chatbot/
├── main.py                          # FastAPI app — endpoints and request handling
├── vector.py                        # ChromaDB setup, embeddings, and retriever
├── database.py                      # SQLite database logic — init, save, fetch history
├── realistic_restaurant_reviews.csv # Sample customer reviews dataset
├── .gitignore
└── README.md
```

## What I learned building this

- How vector search works — converting text to embeddings and searching by meaning instead of exact keywords
- Building a RAG pipeline with LangChain — loaders, embeddings, vector stores, and retrievers
- Using ChromaDB as a persistent local vector database
- Connecting to the OpenAI API for both embeddings and chat completions
- Designing prompts that stay grounded in source data while still allowing natural synthesis across multiple documents
- Reusing a SQLite history pattern across different projects
- Structuring a FastAPI project with clear separation between API logic, vector search logic, and database logic
- Deploying a FastAPI app to Render with environment variable management