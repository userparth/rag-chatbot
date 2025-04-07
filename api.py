from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from chatbot import stream_response, chat_sync, get_history

app = FastAPI()

# Allow requests from your frontend (adjust if deployed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify: ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "RAG Chatbot API is live!"}

@app.get("/chat")
async def chat_stream(query: str, request: Request):
    """
    Endpoint for streaming chat responses to frontend via SSE.
    """
    chat_history = get_history()

    async def streamer():
        async for chunk in stream_response(query, chat_history, request):
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")
