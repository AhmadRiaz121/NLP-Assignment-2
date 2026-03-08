"""
Pydantic schemas for all WebSocket and REST messages exchanged
between clients and the Chiron-AI hotel API.

WebSocket message flow:
  CLIENT → SERVER  : ChatRequest  (JSON)
  SERVER → CLIENT  : TokenChunk  (one per token, streaming)
                     DoneMessage (final frame when generation ends)
                     ErrorMessage (if something goes wrong)
  SERVER → CLIENT  : SessionCreated  (immediately after WS connect)
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


#  Inbound (client → server) 

class ChatRequest(BaseModel):
    """A single guest turn sent over the WebSocket."""
    message: str = Field(..., min_length=1, max_length=4096,
                         description="The guest's message text.")
    session_id: Optional[str] = Field(None,
                         description="Omit to let the server use the current session.")


#  Outbound (server → client) 

class SessionCreated(BaseModel):
    """Sent once when a new WebSocket connection is established."""
    type: Literal["session_created"] = "session_created"
    session_id: str
    welcome: str = (
        "Welcome to Grand Stay Hotel. I'm Chiron-AI, your virtual "
        "front-desk assistant. How can I help you today?"
    )


class TokenChunk(BaseModel):
    """One streaming token fragment."""
    type: Literal["token"] = "token"
    content: str


class DoneMessage(BaseModel):
    """Sent after the last token to signal completion."""
    type: Literal["done"] = "done"
    session_id: str
    intent: str
    turn_count: int
    latency_ms: float
    memory_stats: dict


class ErrorMessage(BaseModel):
    """Sent when a recoverable error occurs during a turn."""
    type: Literal["error"] = "error"
    code: str
    message: str


#  REST response schemas ─

class SessionStats(BaseModel):
    session_id: str
    turn_count: int
    last_intent: str
    total_stored: int
    sent_to_llm: int
    filtered_out: int
    age_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    model_loaded: bool
    active_sessions: int
    model_name: str
