import os
import json
import asyncio
import traceback

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableMap
from langchain.chains import create_retrieval_chain
from langchain_core.callbacks import CallbackManager
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from fastapi import Request
from pinecone import Pinecone

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Define embedding model
embeddings = OpenAIEmbeddings()

# Define unrelated topics to block
UNRELATED_TOPICS = [
    "elon musk", "chatgpt", "news", "nasa", "weather", "india", "prime minister", "salman", "sharukh"
                                                                                            "modi", "politics", "bjp",
    "congress", "stock market", "sports", "cricket", "football", "celebrities", "movies"
]


def is_unrelated_query(query: str) -> bool:
    query = query.lower()
    return any(topic in query for topic in UNRELATED_TOPICS)


# Define prompt template with shopping assistant behavior
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful, friendly shopping assistant for Beatrrangi — a brand offering handcrafted 
    jewelry and accessories. Be warm, helpful, and product-focused. Mirror the user's tone and language (Hinglish, Hindi, 
    or English). 

    🎯 Your only job is to assist with product discovery from the Beatrrangi catalog.

    If the user is inspired by celebrities (e.g., “Disha Patni ki jewelry” or “Rekha wala pendant”), assume they’re describing a **style** and guide them to similar products.

    Only block and politely redirect if the query is **fully unrelated to jewelry or product discovery** — like questions about:
    - politics, celebrities, entertainment news, gossip
    - weather, geography, science, history, personal advice, random chit-chat

    ❗ In such cases:
    → DO NOT comment on the topic
    → DO NOT say things like “Govinda is great” or “I’m not sure”
    → If the query is completely unrelated, do not comment on it. Instead, **detect the user's language (Hindi, Hinglish, English or any other)** and dynamically respond with a polite redirection, such as:
        - Hindi: "Main sirf Beatrrangi ke products mein madad karta hoon. Aap kis tarah ka jewelry dekh rahe ho?"
        - Hinglish: "Beatrrangi ke products mein hi help kar sakta hoon. Koi specific pendant ya earring dekhna hai?"
        - English: "I'm here to help with Beatrrangi products only. Let me know what kind of jewelry, color, or style you’d like to explore."
        - Any Other: Response according to the user's language
    But DO NOT use these exact lines word for word — generate a natural, polite redirect that fits the user's tone.

    Format real product suggestions like this:
    👉 [Product Title](https://beattrangi.com/products/slug)

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

# Maintain chat history globally
chat_history = []


# Async streaming version for FastAPI
async def stream_response(query: str, chat_history: list, request: Request):
    if is_unrelated_query(query):
        yield f"data: {json.dumps({'token': 'I can help you with Beatrrangi products only. Please ask about styles, colors, or jewelry you’re interested in!'})}\n\n"
        yield "data: [DONE]\n\n"
        return

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
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        qa_chain = create_stuff_documents_chain(llm, prompt)
        inner_chain = create_retrieval_chain(retriever, qa_chain)

        # Wrap to emit both answer and context
        rag_chain = RunnableMap({
            "context": lambda x: retriever.ainvoke(x["input"]),
            "answer": inner_chain
        })

        task = asyncio.create_task(rag_chain.ainvoke({
            "input": query,
            "chat_history": chat_history
        }))

        await asyncio.sleep(0.5)

        if task.done():
            response = task.result()
            documents = response.get("context", [])
            if documents:
                cards_html = generate_product_cards_html(documents)
                print("📦 Retrieved documents:", len(documents))
                yield f"data: {json.dumps({'type': 'product_cards_html', 'html': cards_html})}\n\n"

        async for token in callback.aiter():
            if await request.is_disconnected():
                print("❌ Client disconnected — cancelling stream")
                task.cancel()
                return
            yield f"data: {json.dumps({'token': token})}\n\n"

        await task

    except Exception as e:
        print("❌ Streaming error:", str(e))
        traceback.print_exc()  # 🔍 will show full error in terminal
        yield f"data: {json.dumps({'token': '[Error occurred]'})}\n\n"

    finally:
        if hasattr(llm, "aclose"):
            await llm.aclose()
            print("✅ LLM session closed")

    yield "data: [DONE]\n\n"


# Sync version for CLI use
def chat_sync(query: str, chat_history: list):
    if is_unrelated_query(query):
        return "I can help you with Beatrrangi products only. Please ask about styles, colors, or jewelry you’re interested in!"

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
    )

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=None,
        text_key="text"
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    qa_chain = create_stuff_documents_chain(llm, prompt)
    inner_chain = create_retrieval_chain(retriever, qa_chain)
    rag_chain = RunnableMap({
        "context": retriever,
        "answer": inner_chain
    })

    response = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })

    return response.get("answer", "")


# Expose chat history for reuse
def get_history():
    return chat_history


def generate_product_cards_html(products: list[dict]) -> str:
    if not products:
        return ""

    cards_html = ""
    for p in products:
        if not all(k in p for k in ["title", "description", "slug"]):
            continue
        cards_html += f"""
        <div class="product-card" onclick="window.open('https://beattrangi.com/products/{p['slug']}', '_blank')">
            <h4>{p['title']}</h4>
            <p>{p['description']}</p>
        </div>
        """

    return f"""
<div class="product-scroll-container">
{cards_html}
</div>
"""
