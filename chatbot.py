from langchain_core.runnables import RunnableMap
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from typing import List
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY environment variable not set")

# LLM setup for reasoning over document relevance
gpt4 = ChatOpenAI(model="gpt-4o", temperature=0)

# Embedder and VectorStore
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=os.environ.get("PINECONE_INDEX_NAME", "products"),
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# Prompt template (basic RAG style)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer conversationally using only the information provided."),
    ("user", "{input}")
])


# Construct LangChain chain
def get_chain():
    return RunnableMap({
        "docs": retriever,
        "output": lambda inputs: prompt | gpt4 | StrOutputParser()
    })


# History management (simplified as list)
history_store = {}


def get_history(session_id: str) -> List:
    if session_id not in history_store:
        history_store[session_id] = []
    return history_store[session_id]


def filter_docs_by_context(docs: List[Document], query: str) -> List[Document]:
    filtered = []
    for doc in docs:
        prompt = f"""
User asked: {query}
Product: {doc.page_content}
Is this relevant to the user's request? Answer YES or NO.
"""
        result = gpt4.invoke(prompt)
        if "YES" in result.content.upper():
            filtered.append(doc)
    return filtered


def generate_product_cards_html(docs: List[Document]) -> str:
    if not docs:
        return ""
    html = "<div class='product-scroll-container'>"
    for doc in docs:
        metadata = doc.metadata
        title = metadata.get("title", "No Title")
        url = f"https://beattrangi.com/products/{metadata.get('slug', '')}"
        description = metadata.get("description", "")
        html += f"""
            <div class='product-card'>
                <h4>{title}</h4>
                <p style='font-size: 12px;'>{description}</p>
                <a href='{url}' target='_blank'>View Product</a>
            </div>
        """
    html += "</div>"
    return html


def stream_response(query: str, session_id: str):
    history = get_history(session_id)

    chain = get_chain().with_config({"configurable": {"session_id": session_id}})
    response = chain.invoke({"input": query, "chat_history": history})

    if isinstance(response, dict) and "docs" in response:
        docs = filter_docs_by_context(response["docs"], query)
        html = generate_product_cards_html(docs)
        yield json.dumps({"type": "product_cards_html", "html": html})
        yield "[DONE]"
    elif isinstance(response, AIMessage):
        for token in response.content.split():
            yield json.dumps({"token": token + " "})
        yield "[DONE]"
    else:
        yield json.dumps({"token": str(response)})
        yield "[DONE]"


def chat_sync(query: str, session_id: str) -> str:
    history = get_history(session_id)

    chain = get_chain().with_config({"configurable": {"session_id": session_id}})
    response = chain.invoke({"input": query, "chat_history": history})

    if isinstance(response, dict) and "docs" in response:
        docs = filter_docs_by_context(response["docs"], query)
        return generate_product_cards_html(docs)
    elif isinstance(response, AIMessage):
        return response.content
    return str(response)
