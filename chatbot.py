import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
import pinecone

load_dotenv()


def initialize_chatbot(index_name):
    # Define the Pinecone API key and index name
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=os.getenv("PINECONE_ENVIRONMENT"))
    index = pinecone.Index(index_name)

    # Define the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    # Load the Pinecone vector store
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Create a retriever for querying the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    # Create a ChatOpenAI model using gpt-4o
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o")

    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    # Create a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question prompt
    qa_system_prompt = """
        You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
        Don't use information other than the retrieved context. If the answer is not in the retrieved documents, you are also allowed to ask a follow up question
        or if the query doesn't make any sense then just say that "I did not find any relevant information to your query.

        Tips:

        1- Don't use the words like "mentioned in the provided context" in your answer.
        2- If the answer is not in the retrieved documents, you are allowed to ask a follow-up question. (Only if the question is somewhat related to Universities/Students)
        3- If the question is not related Universities/Students, you can say that "I did not find any relevant information to your query." and don't ask any follow up questions for those queries (which are way off).
        4- You are a helpful assistant that helps students with their questions in the most respectful way possible.
        5- You are someone who has a lot of knowledge about universities and students, you want to help students to not have any confusions regarding their careers.

        Do not answer questions other than the provided context:
        {context}

        \n\n===Answer===\n[Provide the normal answer here.]\n"""

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a chain to combine documents for question answering
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain, retriever


def process_query(query, chat_history, rag_chain):
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    return result["answer"]


# Function to simulate a continual chat
def continual_chat(index_name, query, chat_history):
    rag_chain, retriever = initialize_chatbot(index_name)
    while True:
        # Process the user's query through the retrieval chain
        result = process_query(query, chat_history, rag_chain)
        # Display the AI's response
        print(f"AI: {result}")

        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result))

        return result, chat_history

