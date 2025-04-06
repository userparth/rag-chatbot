import os
from chatbot import chat

# Optional: Run prepare_data.py if data needs to be refreshed
RUN_PREPARE = False  # Set to True to refresh data from CSV

if RUN_PREPARE:
    import prepare_data  # Just running it will execute the upsert script


def main():
    print("Welcome to the Product Chatbot! Type 'exit' to quit.\n")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            answer, chat_history = chat(user_input, chat_history)
            print("Bot:", answer)
        except Exception as e:
            print("[Error]", e)


if __name__ == "__main__":
    # print("DONE")
    main()
