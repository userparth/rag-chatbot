import os
from chatbot import chat

# Optional: Run prepare_data.py if data needs to be refreshed
RUN_PREPARE = False  # Set to True to refresh data from CSV

if RUN_PREPARE:
    import prepare_data  # Just running it will execute the upsert script

exit_keywords = {"exit", "quit", "bye", "goodbye", "see you", "talk to you later"}


def run_cli():
    print("Welcome to the Product Chatbot! Type 'exit' to quit.\n")
    chat_history = []

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            response, chat_history = chat(user_input, chat_history)
            print("Bot:", response)

            if any(keyword in user_input.lower() for keyword in exit_keywords):
                print("👋 Exiting chat. Take care!")
                break

    except KeyboardInterrupt:
        print("\n👋 Exiting chat. Take care!")


if __name__ == "__main__":
    # print("DONE")
    run_cli()
