# chatbot.py
# This script implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain, OpenAI, and Pinecone.
# It provides both a streaming and sync response system for handling user queries about Beatrrangi products.

import os
import json
import asyncio
import traceback

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableMap
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from fastapi import Request
from pinecone import Pinecone

# Load environment variables from a .env file
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Set up the OpenAI embeddings model for vector search
# Define embedding model
embeddings = OpenAIEmbeddings()

# Predefined list of topics considered unrelated to product discovery
# Define unrelated topics to block
UNRELATED_TOPICS = [
    "elon musk", "chatgpt", "news", "nasa", "weather", "india", "prime minister", "salman", "sharukh",
    "modi", "politics", "bjp", "congress", "stock market", "sports", "cricket", "football", "celebrities", "movies"
]

# Set to track and prevent duplicate product suggestions
# Memory to prevent showing the same slug again
shown_slugs = set()


# Check if a query contains unrelated topics
def is_unrelated_query(query: str) -> bool:
    query = query.lower()
    return any(topic in query for topic in UNRELATED_TOPICS)


# Prompt template defining chatbot behavior and tone
# Prompt with rich instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful, friendly shopping assistant for Beatrrangi — a brand offering handcrafted 
    jewelry and accessories. Be warm, helpful, and product-focused. Mirror the user's tone and language (Hinglish, Hindi, 
    or English). 

    🎯 Your only job is to assist with product discovery from the Beatrrangi catalog.

    If the user is inspired by celebrities (e.g., “Disha Patni ki jewelry” or “Rekha wala pendant”), assume they’re describing a style and guide them to similar products.

    Only block and politely redirect if the query is fully unrelated to jewelry/product discovery — like:
    - politics, gossip, science, geography
    - personal questions, random fun, unrelated advice

    ❗ In such cases:
    → DO NOT comment on the topic
    → DO NOT say things like “Govinda is great” or “I’m not sure”
    → If the query is completely unrelated, do not comment on it. Instead, **detect the user's language (Hindi, Hinglish, English or any other)** and dynamically respond with a polite redirection, such as:
        - Hindi: "Main sirf Beatrrangi ke products mein madad karta hoon. Aap kis tarah ka jewelry dekh rahe ho?"
        - Hinglish: "Beatrrangi ke products mein hi help kar sakta hoon. Koi specific pendant ya earring dekhna hai?"
        - English: "I'm here to help with Beatrrangi products only. Let me know what kind of jewelry, color, or style you’d like to explore."
        - Any Other: Response according to the user's language
    But DO NOT use these exact lines word for word — generate a natural, polite redirect that fits the user's tone.

    Format valid product matches like:
    👉 [Product Title](https://beattrangi.com/products/slug): hand-painted charm.


    ⚠️ Do not ask for email or Instagram.
    ⚠️ Do not say “India made” — all products are Indian by default.
    ⚠️ Do not repeat product info already shared. Just reference the product title.
    ⚠️ Do not hallucinate product names or links — only use retrieved results.
    ⚠️ Do not say “context” or technical phrases.

    ✅ If no match is found, politely let the user know and offer help with other styles, colors, or categories — using a friendly and natural tone.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}\n\nContext:\n{context}")
])

# Global variable to store chat history
# Global history
chat_history = []


# Stream GPT responses via SSE for real-time chat UI
# Streamed version for FastAPI (SSE)
async def stream_response(query: str, chat_history: list, request: Request):
    if is_unrelated_query(query):
        redir = {
            "hi": "Main sirf Beatrrangi ke products mein madad karta hoon. Aap kis tarah ka jewelry dekh rahe ho?",
            "en": "I'm here to help with Beatrrangi products only. Let me know your style or color preference!",
            "default": "Beatrrangi ke products mein hi help kar sakta hoon. Koi pendant ya earring dekhna hai?"
        }
        token = redir["default"]
        yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Initialize ChatOpenAI with streaming and callback support
    callback = AsyncIteratorCallbackHandler()
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        streaming=True,
        callbacks=[callback],
    )

    try:
        # Create retriever from Pinecone index using OpenAI embeddings
        # Initialize retriever
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=None,
            text_key="text"
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Create document-answering chain using LangChain components
        # RAG chain
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        # Wrap retriever and RAG chain for combined context-answer output
        wrapped_chain = RunnableMap({
            "context": lambda x: retriever.ainvoke(x["input"]),
            "answer": rag_chain
        })

        # Start chain call
        task = asyncio.create_task(wrapped_chain.ainvoke({
            "input": query,
            "chat_history": chat_history
        }))

        # Stream tokens back to client as they are generated
        # Start streaming tokens
        async for token in callback.aiter():
            if await request.is_disconnected():
                print("❌ Client disconnected — cancelling stream")
                task.cancel()
                return
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Now wait for full result (after tokens done)
        await task

    except Exception as e:
        print("❌ Error in stream:", str(e))
        traceback.print_exc()
        yield f"data: {json.dumps({'token': '[Error occurred]'})}\n\n"

    finally:
        if hasattr(llm, "aclose"):
            await llm.aclose()

    yield "data: [DONE]\n\n"

# Synchronous version of the chatbot for CLI or debugging
# CLI-style sync version
def chat_sync(query: str, chat_history: list):
    if is_unrelated_query(query):
        return "I can help you with Beatrrangi products only. Please ask about jewelry, colors, or styles you like."

    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=None,
        text_key="text"
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    wrapper = RunnableMap({
        "context": retriever,
        "answer": rag_chain
    })

    result = wrapper.invoke({
        "input": query,
        "chat_history": chat_history
    })

    return result.get("answer", "")


# Getter for chat history
# Expose history
def get_history():
    return chat_history


# Convert retrieved products to HTML card layout for UI rendering
# Render product cards as HTML blocks
def generate_product_cards_html(products: list[dict]) -> str:
    print("🔍 Raw input to card generator:", products)

    if not products:
        print("⚠️ No products passed for HTML rendering")
        return ""

    if not products:
        return ""

    cards_html = ""
    for p in products:
        slug = p.get("slug")
        if not slug or slug in shown_slugs:
            continue

        shown_slugs.add(slug)
        title = p.get("title", "Jewelry")
        desc = p.get("description", "")[:120]
        cards_html += f"""
        <div class="product-card" onclick="window.open('https://beattrangi.com/products/{slug}', '_blank')">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """

    return f"<div class='product-scroll-container'>{cards_html}</div>"
