import argparse
import subprocess
import threading
from main import run_cli


def run_api():
    subprocess.run(["uvicorn", "api:app", "--port", "8000", "--reload"])


def main():
    parser = argparse.ArgumentParser(description="Run chatbot in CLI, API, or both modes.")
    parser.add_argument("--cli", action="store_true", help="Run CLI chat interface")
    parser.add_argument("--api", action="store_true", help="Run FastAPI server for frontend access")

    args = parser.parse_args()

    if args.cli and args.api:
        t1 = threading.Thread(target=run_cli)
        t2 = threading.Thread(target=run_api)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    elif args.cli:
        run_cli()

    elif args.api:
        run_api()

    else:
        print("❗ Please specify at least one of: --cli, --api")


if __name__ == "__main__":
    main()
