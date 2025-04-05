from chatbot import continual_chat  # Replace with actual module if separated


if __name__ == "__main__":
    index_name = "your-pinecone-index-name"
    chat_history = []

    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result, chat_history = continual_chat(index_name, user_input, chat_history)
        print("-------------\n")