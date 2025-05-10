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
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig, RunnableMap
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from fastapi import Request

from chat_history import get_history, append_to_history, trim_history

# Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Set up the OpenAI embeddings model for vector search
embeddings = OpenAIEmbeddings()

# Predefined list of topics considered unrelated to product discovery
UNRELATED_TOPICS = [
    "elon musk", "chatgpt", "news", "nasa", "weather", "india", "prime minister", "salman", "sharukh",
    "modi", "politics", "bjp", "congress", "stock market", "sports", "cricket", "football", "celebrities", "movies"
]

# Set to track and prevent duplicate product suggestions
shown_slugs = set()


def is_unrelated_query(query: str) -> bool:
    query = query.lower()
    return any(topic in query for topic in UNRELATED_TOPICS)


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
    → If the query is completely unrelated, do not comment on it. Instead, **detect the user's language (Hindi, Hinglish, English or any other)** and dynamically respond with a polite redirection.

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


async def stream_response(query: str, chat_history: list, request: Request):
    session_id = request.query_params.get("session_id")
    if not session_id:
        yield f"data: {json.dumps({'token': 'Missing session ID'})}\n\n"
        yield "data: [DONE]\n\n"
        return

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

    chat_history = get_history(session_id)
    append_to_history(session_id, "human", query)

    callback = AsyncIteratorCallbackHandler()
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        streaming=True,
        callbacks=[callback],
    )

    try:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=None,
            text_key="text"
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        wrapped_chain = RunnableMap({
            "context": lambda x: retriever.ainvoke(x["input"]),
            "answer": rag_chain
        })

        task = asyncio.create_task(wrapped_chain.ainvoke({
            "input": query,
            "chat_history": chat_history
        }))

        full_response = ""
        async for token in callback.aiter():
            if await request.is_disconnected():
                print("❌ Client disconnected — cancelling stream")
                task.cancel()
                return
            full_response += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        await task
        append_to_history(session_id, "ai", full_response)
        trim_history(session_id)

    except Exception as e:
        print("❌ Error in stream:", str(e))
        traceback.print_exc()
        yield f"data: {json.dumps({'token': '[Error occurred]'})}\n\n"

    finally:
        if hasattr(llm, "aclose"):
            await llm.aclose()

    yield "data: [DONE]\n\n"


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
