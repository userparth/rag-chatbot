import os
import json
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain.chains.combine_documents import create_stuff_documents_chain
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

# Define prompt template with shopping assistant behavior
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful, friendly shopping assistant. Speak in a casual and natural way that matches the user's tone and language.

Your job is to:
- Help users find beautiful handcrafted jewelry from the catalog
- Detect the user's tone or language (e.g., Hinglish, Hindi, casual English) and mirror it naturally
- Use retrieved product info to explain size, color, material, and styling
- If the product exists, give a direct link to order using the link already included in the retrieved result

Format the response with:
👉 [Product Title](https://beattrangi.com/products/slug)

⚠️ Do not ask the user to contact via email or Instagram.
⚠️ Do not mention 'India made' — all products are Indian by default.
⚠️ Do not repeat product details that have already been shared earlier in the chat. Only reference them by title.
⚠️ If a product is not found in the retrieved results, do NOT make up or guess a product. Never suggest or create product names or links that are not part of the results.
⚠️ Do not use words like 'context' in your replies. The user should feel like you're responding directly, not reading from a technical source.
✅ If no match is found, politely let the user know and offer to help with other styles, colors, or product types.

Always keep the tone warm, helpful, and respectful.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}\n\nContext:\n{context}")
])

# Maintain chat history globally
chat_history = []


# Sync version for CLI use or testing
def chat_sync(query: str) -> str:
    global chat_history

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        temperature=0.7
    )

    # Load vector store fresh per call
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=None,
        text_key="text"
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    result = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })

    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=result["answer"]))

    return result["answer"]


# Async streaming version for FastAPI
async def stream_response(query: str, chat_history: list, request: Request):
    callback = AsyncIteratorCallbackHandler()
    manager = CallbackManager([callback])

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        streaming=True,
        callback_manager=manager,
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
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        task = asyncio.create_task(rag_chain.ainvoke({
            "input": query,
            "chat_history": chat_history
        }))

        async for token in callback.aiter():
            if await request.is_disconnected():
                print("❌ Client disconnected — cancelling stream")
                task.cancel()
                return
            yield f"data: {json.dumps({'token': token})}\n\n"

        await task

    except Exception as e:
        print("❌ Streaming error:", str(e))
        yield f"data: {json.dumps({'token': '[Error occurred]'})}\n\n"

    finally:
        if hasattr(llm, "aclose"):
            await llm.aclose()
            print("✅ LLM session closed")

    yield "data: [DONE]\n\n"


def get_history():
    return chat_history
