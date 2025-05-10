# chat_history.py
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage

# Simple in-memory store
_session_store: Dict[str, List] = {}


def get_history(session_id: str) -> List:
    return _session_store.get(session_id, [])


def append_to_history(session_id: str, role: str, content: str):
    message = HumanMessage(content) if role == "human" else AIMessage(content)
    if session_id not in _session_store:
        _session_store[session_id] = []
    _session_store[session_id].append(message)


def trim_history(session_id: str, limit: int = 40):
    if session_id in _session_store:
        _session_store[session_id] = _session_store[session_id][-limit:]
