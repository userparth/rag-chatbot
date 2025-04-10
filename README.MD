# 🛍️ RAG Product Chatbot

An intelligent product discovery chatbot built using LangChain, GPT-4o, Pinecone, and FastAPI.
It supports natural language queries in Hinglish/Hindi/English and helps users explore real products from the Beatrrangi Shopify catalog.

---

## ✨ Features

- 💬 Conversational RAG-based product assistant
- 🌐 Hinglish/Hindi support with tone mirroring
- 🔗 Real Shopify product links (no hallucinated URLs)
- 🧠 Vector search via Pinecone
- 🔁 Full session-based context memory
- ⚡ Live streaming responses using FastAPI + SSE
- 📦 Floating chatbot widget embeddable on Shopify

---

## 📁 Project Structure

```
├── chatbot.py           # RAG core logic and chat history
├── api.py               # FastAPI server with streaming endpoint
├── main.py              # CLI interface (for dev/debug)
├── prepare_data.py      # Loads and upserts product CSV to Pinecone
├── chatbot_widget.html  # Embeddable UI for Shopify
├── run.py               # Entrypoint for launching API or CLI
├── .env                 # API keys (excluded from Git)
├── .gitignore           # Ignores .env, CSVs, venv, etc.
└── products_export.csv  # Input product data (optional)
```

---

## ⚙️ Setup Instructions

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/rag-chatbot
cd rag-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add `.env`
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=products
```

### 3. Prepare Pinecone Index
```bash
python prepare_data.py  # One-time run to upload embeddings from CSV
```

### 4. Run Chatbot (CLI)
```bash
python main.py
```

### 5. Run Chatbot (API + Streaming UI)
```bash
uvicorn api:app --reload
```
Then open `chatbot_widget.html` in your browser or embed in Shopify.

---

## 🧠 Prompt Guidelines

The system prompt ensures:
- Avoids hallucinated product names or links
- Never asks for Instagram/email contact
- Mirrors user's language and tone
- Responds clearly with formatting:
  ```
  👉 [Blue Floral Pendant](https://example.com/products/blue-floral-pendant)
  ```

---

## 💡 Tips

- Make sure only `Published=True` products are embedded.
- You can control vector search filters with:
  ```python
  vectorstore.as_retriever(search_kwargs={"k": 3})
  ```

---

## 🚀 Deploy
You can deploy this chatbot backend to:
- Render / Fly.io / Railway (for API)
- Embed the widget into Shopify theme code

Need help? Open an issue or ask! 😄

