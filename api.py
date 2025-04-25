from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional # Import Optional
# Make sure chat_sync is imported if you plan to use it elsewhere,
# but it's not used in the chat_stream endpoint itself.
from chatbot import stream_response, get_history # Removed chat_sync if unused here

import uuid # Import uuid for session ID generation if needed

app = FastAPI()

# Allow requests from your frontend (adjust if deployed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be more specific in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "RAG Chatbot API is live!"}

@app.get("/chat")
async def chat_stream(query: str, request: Request, session_id: Optional[str] =  None): # Added optional session_id query param
    """
    Endpoint for streaming chat responses to frontend via SSE.
    Uses client host as a simple session identifier if no session_id is provided.
    """
    # Option 1: Use client host as session ID (Simple, less reliable for multiple tabs/users behind NAT)
    # effective_session_id = request.client.host if request.client else "default_session"

    # Option 2: Use a provided session_id query parameter, or generate one (More robust)
    if not session_id:
        # If no session_id provided, you might generate one, but it would be new for each request.
        # A better approach is for the client (HTML/JS) to generate/retrieve and send a stable session_id.
        # For now, let's fall back to client host or a default if client info isn't available
        effective_session_id = request.client.host if request.client else "default_session"
        print(f"Warning: No session_id provided by client. Using derived ID: {effective_session_id}")
    else:
        effective_session_id = session_id
        print(f"Using provided session_id: {effective_session_id}")


    # The get_history call is removed from here, as it's handled inside stream_response

    async def streamer():
        # Pass the determined session_id to stream_response
        async for chunk in stream_response(query, effective_session_id):
             # Ensure chunk is string data for SSE (often JSON strings)
            if isinstance(chunk, bytes):
                 yield f"data: {chunk.decode('utf-8')}\n\n"
            else:
                 yield f"data: {str(chunk)}\n\n" # Ensure it's prefixed with 'data: ' and ends with \n\n


    return StreamingResponse(streamer(), media_type="text/event-stream")

# Example of how chat_sync might be used if you had another endpoint
# @app.get("/chat-sync")
# async def chat_sync_endpoint(query: str, request: Request):
#     session_id = request.client.host if request.client else "default_sync_session"
#     response_content = chat_sync(query, session_id)
#     return {"response": response_content}