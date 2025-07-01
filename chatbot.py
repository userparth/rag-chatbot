# chatbot.py

import os
import json
import asyncio
import traceback

from dotenv import load_dotenv
from fastapi import Request
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableMap
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

from chat_history import get_history, append_to_history, trim_history

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

UNRELATED_TOPICS = [
    "elon musk", "chatgpt", "news", "nasa", "weather", "india", "prime minister", "salman", "sharukh",
    "modi", "politics", "bjp", "congress", "stock market", "sports", "cricket", "football", "celebrities", "movies"
]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a warm, friendly shopping assistant for Beattrangi, a brand that offers handcrafted jewelry 
    and accessories. Your role is to guide users in discovering products from the Beattrangi catalog by mirroring 
    their tone and language—whether they speak in English, Hindi, or Hinglish—while remaining helpful, 
    product-focused, and engaging. Politely redirect any off-topic queries unrelated to jewelry or product discovery 
    without commenting on the subject; simply detect the user's language and respond with a courteous message guiding 
    them back to relevant queries. Do not engage with topics like politics, celebrities, or personal requests, 
    and avoid discussing platform-specific interactions like email or Instagram. All responses should be grounded in 
    actual catalog data with no made-up information, and refrain from repeating product details unnecessarily. 
    Display product recommendations in this format with valid product matches like: 
    👉 [Product Title](https://beattrangi.com/products/slug): hand-painted charm.
    
    If no matching product is found, suggest exploring other styles or categories instead. Maintain clarity, 
    accuracy, a friendly and helpful tone, but avoid poetic or overly dramatic phrases. Be conversational and to the 
    point. Use clear, accessible Hinglish or Hindi/English based on the user's language."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}\n\nContext:\n{context}")
])


def is_unrelated_query(query: str) -> bool:
    return any(topic in query.lower() for topic in UNRELATED_TOPICS)


def get_redirection_token():
    return {
        "hi": "Main sirf Beattrangi ke products mein madad karta hoon. Aap kis tarah ka jewelry dekh rahe ho?",
        "en": "I'm here to help with Beattrangi products only. Let me know your style or color preference!",
        "default": "Beattrangi ke products mein hi help kar sakta hoon. Koi pendant ya earring dekhna hai?"
    }["default"]


# Shared embeddings instance
embeddings = OpenAIEmbeddings()


def build_retriever(k: int = 4):
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=None,
        text_key="text"
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k, "filter": {"category": {"$ne": "rakhi"}}})


def build_rag_chain(llm, retriever):
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


async def stream_response(query: str, chat_history: list, request: Request):
    session_id = request.query_params.get("session_id")
    if not session_id:
        yield f"data: {json.dumps({'token': 'Missing session ID'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if is_unrelated_query(query):
        yield f"data: {json.dumps({'token': get_redirection_token()})}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        history = get_history(session_id)
        append_to_history(session_id, "human", query)

        callback = AsyncIteratorCallbackHandler()
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            streaming=True,
            callbacks=[callback],
        )

        retriever = build_retriever()
        rag_chain = build_rag_chain(llm, retriever)

        wrapped_chain = RunnableMap({
            "context": lambda x: retriever.ainvoke(x["input"]),
            "answer": rag_chain
        })

        task = asyncio.create_task(wrapped_chain.ainvoke({
            "input": query,
            "chat_history": history
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
        return get_redirection_token()

    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    retriever = build_retriever(k=3)
    rag_chain = build_rag_chain(llm, retriever)

    result = RunnableMap({
        "context": retriever,
        "answer": rag_chain
    }).invoke({
        "input": query,
        "chat_history": chat_history
    })

    return result.get("answer", "")
