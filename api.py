from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chatbot import stream_response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared chat history
chat_history = []


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def stream_chat(req: ChatRequest):
    return StreamingResponse(
        stream_response(req.query, chat_history),
        media_type="text/event-stream"
    )
