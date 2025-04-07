import asyncio
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain_core.callbacks.manager import CallbackManager

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "products"

# Initialize vector store from Pinecone index
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    text_key="text"
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, streaming=True,
                 callbacks=[])

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
    MessagesPlaceholder("chat_history"),
    ("human", "{input}\n\nContext:\n{context}")
])

# Create QA chain and retrieval pipeline
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


def chat(query, chat_history=None):
    if chat_history is None:
        chat_history = []
    response = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    return response["answer"], chat_history + [
        HumanMessage(content=query),
        SystemMessage(content=response["answer"])
    ]


def get_streaming_chain():
    callback = AsyncIteratorCallbackHandler()
    manager = CallbackManager([callback])

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
        callback_manager=manager
    )

    # Use your existing prompt here (prompt already defined in chatbot.py)
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain), callback


async def stream_response(query: str, chat_history: list):
    rag_chain, callback = get_streaming_chain()
    inputs = {"input": query, "chat_history": chat_history}
    task = asyncio.create_task(rag_chain.ainvoke(inputs))

    async for token in callback.aiter():
        yield f"data: {json.dumps({'token': token})}\n\n"

    await task
    yield f"data: [DONE]\n\n"
