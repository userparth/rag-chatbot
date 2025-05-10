# RAG-Based Product Chatbot

A retrieval-augmented generation (RAG) chatbot system that combines LangChain, GPT-4o, Pinecone, and FastAPI to deliver intelligent, real-time product discovery experiences via natural language interfaces. Designed for integration into existing storefronts, the system supports multilingual interactions, dynamic session memory, and streaming UI updates.

---

## Features

* **RAG Pipeline:** End-to-end retrieval and generation architecture powered by LangChain 0.2 and GPT-4o
* **Multilingual Support:** Accurately processes English, Hindi, and Hinglish with tone-aware response formatting
* **Session Memory:** Context preserved per session using in-memory or persistent mechanisms
* **Streaming Output:** Real-time streaming via FastAPI Server-Sent Events (SSE) for dynamic frontend updates
* **Verified Link Handling:** All product links are derived from a curated dataset and never hallucinated
* **Frontend Widget:** Vanilla JS-based floating widget embeddable on any HTML webpage
* **Vector Database:** Product embeddings indexed and retrieved using Pinecone
* **Configurable Parameters:** Retrieval logic (e.g., top-k), prompt templates, and fallback behavior are customizable

---

## Project Structure

```
├── chatbot.py           # RAG logic with retriever chain, memory, redirect handler
├── api.py               # FastAPI application with SSE-compatible /chat endpoint
├── run.py               # Central launcher for CLI or API modes
├── main.py              # CLI-based interaction testing interface
├── prepare_data.py      # Data pipeline for vectorizing and storing product metadata
├── chat_history.py      # Session-based memory store and history utilities
├── chatbot_widget.html  # Client-side JS widget for embedding in external sites
├── requirements.txt     # Complete list of dependencies
└── .env                 # Local secrets for OpenAI and Pinecone (excluded from VCS)
```

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/rag-chatbot
cd rag-chatbot
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file with the following content:

```
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=products
```

### 5. Generate and Upload Embeddings

```bash
python prepare_data.py  # Parses CSV and uploads embeddings to Pinecone
```

### 6. Launch the API Server

```bash
uvicorn api:app --reload
```

### 7. Optional: Run Command-Line Interface

```bash
python main.py
```

---

## Embeddable Chatbot Widget

Use `chatbot_widget.html` to add a non-intrusive, floating chatbot on your frontend:

* Compatible with any static site or server-rendered page
* Built using Vanilla JS + CSS (no frameworks required)
* Handles EventSource streams and response rendering
* Dynamically rewrites product links and prevents URL duplication or nested anchors

Widget automatically generates a session ID and maintains UI state across toggles.

---

## Prompt Engineering & Safety Guidelines

* Structured system prompt uses multilingual redirection logic to block off-topic queries
* Filters unsupported topics and avoids replying to personal or non-product-related prompts
* Example response:

  ```markdown
  [Gold Plated Kundan Set](https://yourdomain.com/products/gold-plated-kundan-set): traditional design with pearls
  ```
* Respects constraints like:

  * Never ask for user contact info
  * Never hallucinate URLs
  * Never repeat previously shown product info

---

## Deployment Recommendations

* **Backend:** Deploy API to Render, Fly.io, or Railway. Ensure `/chat` SSE endpoint supports long-lived connections.
* **Frontend:** Embed widget directly inside HTML templates or inject via tag manager.
* **Environment Settings:**

  * Set CORS policies to allow domain from which the frontend is served
  * Limit OpenAI/Pinecone usage with quotas where applicable

---

## Advanced Configuration

### Retrieval Filtering

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

Adjust top-k retrieval, use metadata filters, or combine keyword search with vector distance.

### Redis Memory (Optional)

To enable Redis-based long-term memory:

* Install Redis locally or via a service like Upstash
* Add `REDIS_URL=redis://localhost:6379` to `.env`
* Update `chat_history.py` to use RedisMemory class from LangChain

---

## Notes

* This project aligns with modern LangChain standards (0.2.x)
* Token streaming is fully supported via GPT-4o and OpenAI client v1.30+
* Prompt control and response formatting can be extended via `RunnableMap` and `create_stuff_documents_chain`
* Frontend widget includes built-in sanitization and cross-domain compatibility