import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

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
retriever = vectorstore.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 3})

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# Define prompt template with shopping assistant behavior
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful, friendly shopping assistant. Detect the user's language or tone (e.g., Hinglish, 
    Hindi, casual English) and reply in the same style and language. Do not switch to formal English unless the user 
    does.

Your job is to:
- Help users find beautiful handcrafted jewelry from the catalog
- Use retrieved product info to explain size, color, material, and styling
- If the product exists, give a direct link to order using the link already present in the context.

Format the response with:
👉 [Product Title](metadata['link'])

⚠️ Do not ask the user to contact via email or Instagram.
⚠️ Do not mention 'India made' — all products are Indian by default.
⚠️ Do not repeat product details already shown earlier in the conversation.
⚠️ If a product has already been shown in the conversation, do not repeat its full details again. Just reference it briefly.
⚠️ If a product is not found in the retrieved context, do NOT make up or guess a product. Never suggest or create product names or links that are not part of the context.
Only refer to products that are explicitly mentioned in the context you are given.
Always keep the tone warm, respectful, and helpful.
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
