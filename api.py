from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from chatbot import stream_response
from collections import defaultdict

app = FastAPI()

# Allow requests from your frontend (adjust if deployed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify: ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
session_histories = defaultdict(list)

@app.get("/")
def home():
    return {"message": "RAG Chatbot API is live!"}

@app.get("/chat")
async def chat_stream(query: str, session_id: str, request: Request):
    """
    Endpoint for streaming chat responses to frontend via SSE.
    """
    history = session_histories[session_id]
    stream = stream_response(query=query, chat_history=history, request=request)
    # chat_history = get_history()
    #
    # async def streamer():
    #     async for chunk in stream_response(query, chat_history, request):
    #         yield chunk

    return StreamingResponse(stream, media_type="text/event-stream")
