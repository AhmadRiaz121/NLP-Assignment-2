"""
Chiron-AI Hotel Assistant — FastAPI Microservice (Phase 4)

Endpoints
---------
WebSocket:
  WS  /ws/chat                 Streaming chat (primary endpoint)

REST:
  GET  /health                 Health check
  POST /sessions               Create a new session (returns session_id)
  GET  /sessions/{id}/stats    Stats for one session
  GET  /sessions               List all active session IDs
  DELETE /sessions/{id}        Close / delete a session

WebSocket protocol (JSON frames):
  On connect   → server sends  SessionCreated
  Client sends → ChatRequest   {"message": "...", "session_id": "..."}
  Server sends → TokenChunk    {"type": "token",  "content": "..."}  (repeated)
                 DoneMessage   {"type": "done",   "session_id": "...", ...}
                 ErrorMessage  {"type": "error",  "code": "...", "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .llm_handler import (
    load_model,
    is_loaded,
    stream_response,
    OFF_TOPIC_REPLY,
    MODEL_NAME,
)
from .models import (
    ChatRequest,
    SessionCreated,
    TokenChunk,
    DoneMessage,
    ErrorMessage,
    SessionStats,
    HealthResponse,
)
from .session_store import store, detect_intent


#  Application lifespan 

@asynccontextmanager
async def lifespan(app: FastAPI):                   # noqa: ARG001
    """
    Start the server immediately, then load the model in the background.
    The /health endpoint returns {"status": "degraded"} until loading finishes.
    WebSocket /ws/chat returns a clear error message if a request arrives
    before the model is ready.
    """
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, load_model)          # fire-and-forget background load
    yield
    # Shutdown: nothing to clean up (OS reclaims memory)


#  App factory

app = FastAPI(
    title="Chiron-AI — Grand Stay Hotel Virtual Assistant",
    description="WebSocket + REST microservice for the hotel front-desk chatbot.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


#  Chat UI 

# Resolve the Phase 5 index.html regardless of where uvicorn is launched from
_UI_FILE=Path(__file__).parent.parent.parent / "Phase 5" / "index.html"


@app.get("/", include_in_schema=False)
@app.get("/chat", include_in_schema=False)
async def serve_ui() -> FileResponse:
    """Serve the web-based chat interface."""
    if not _UI_FILE.exists():
        raise HTTPException(status_code=404,
                            detail=f"UI file not found at {_UI_FILE}")
    return FileResponse(str(_UI_FILE), media_type="text/html")


#  Health 

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if is_loaded() else "degraded",
        model_loaded=is_loaded(),
        active_sessions=store.count,
        model_name=MODEL_NAME,
    )


#  Session REST endpoints 

@app.post("/sessions", status_code=status.HTTP_201_CREATED, tags=["sessions"])
async def create_session() -> dict:
    """Create a new guest session and return its ID."""
    session = store.create()
    return {
        "session_id": session.session_id,
        "message"   : "Session created. Connect to /ws/chat to begin.",
    }


@app.get("/sessions", tags=["sessions"])
async def list_sessions() -> dict:
    return {"session_ids": store.all_ids(), "count": store.count}


@app.get("/sessions/{session_id}/stats",
         response_model=SessionStats, tags=["sessions"])
async def session_stats(session_id: str) -> SessionStats:
    try:
        s = store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return SessionStats(**s.stats())


@app.delete("/sessions/{session_id}",
            status_code=status.HTTP_204_NO_CONTENT, tags=["sessions"])
async def delete_session(session_id: str) -> None:
    if not store.delete(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


#  WebSocket chat 

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """
    Persistent WebSocket endpoint for streaming hotel-assistant chat.

    One connection = one guest session (created automatically on connect).
    The session lives until the connection closes or the client calls
    DELETE /sessions/{id}.

    Message protocol (all JSON):
      receive  ChatRequest  → {"message": "...", "session_id": "optional"}
      send     TokenChunk   → {"type": "token",  "content": "..."}
      send     DoneMessage  → {"type": "done",   ...}
      send     ErrorMessage → {"type": "error",  ...}
    """
    await websocket.accept()

    # Create a session for this connection
    session = store.create()
    welcome = SessionCreated(session_id=session.session_id)
    await websocket.send_text(welcome.model_dump_json())

    try:
        while True:
            #  Receive guest message 
            raw = await websocket.receive_text()

            try:
                data    = json.loads(raw)
                request = ChatRequest.model_validate(data)
            except Exception:
                err = ErrorMessage(
                    code    = "invalid_request",
                    message = 'Expected JSON with field "message": "<text>".',
                )
                await websocket.send_text(err.model_dump_json())
                continue

            user_msg = request.message.strip()
            if not user_msg:
                err = ErrorMessage(
                    code    = "empty_message",
                    message = "Message must not be empty.",
                )
                await websocket.send_text(err.model_dump_json())
                continue

            #  Handle warmup message (preload model silently) 
            if user_msg == "__WARMUP__":
                if not is_loaded():
                    # Model still loading, send info but don't error
                    await websocket.send_json({"type": "warmup_complete", "loaded": False})
                else:
                    # Trigger a quick inference to warm up the pipeline
                    try:
                        _ = []
                        async for _ in stream_response([{"role": "user", "content": "Hi"}]):
                            pass  # Discard warmup tokens
                    except Exception:  # noqa: BLE001
                        pass
                    await websocket.send_json({"type": "warmup_complete", "loaded": True})
                continue

            #  Check if model is loaded ─
            if not is_loaded():
                err = ErrorMessage(
                    code    = "model_not_ready",
                    message = "The AI model is still loading. Please wait a moment and try again.",
                )
                await websocket.send_text(err.model_dump_json())
                continue

            #  Intent detection + off-topic guard ─
            intent           = detect_intent(user_msg)
            session.last_intent = intent
            session.turn_count += 1
            t_start          = time.perf_counter()

            if intent == "off_topic":
                # Return canned reply without calling the LLM
                session.memory.add("user",      user_msg)
                session.memory.add("assistant", OFF_TOPIC_REPLY)

                chunk = TokenChunk(content=OFF_TOPIC_REPLY)
                await websocket.send_text(chunk.model_dump_json())

                done = DoneMessage(
                    session_id   = session.session_id,
                    intent       = intent,
                    turn_count   = session.turn_count,
                    latency_ms   = (time.perf_counter() - t_start) * 1000,
                    memory_stats = session.memory.stats(),
                )
                await websocket.send_text(done.model_dump_json())
                continue

            #  Stream LLM tokens ─
            session.memory.add("user", user_msg)
            collected: list[str] = []

            try:
                async for token in stream_response(session.memory.get_context()):
                    collected.append(token)
                    chunk = TokenChunk(content=token)
                    await websocket.send_text(chunk.model_dump_json())

            except Exception as exc:
                err = ErrorMessage(
                    code    = "llm_error",
                    message = f"Inference error: {exc}",
                )
                await websocket.send_text(err.model_dump_json())
                continue

            full_reply = "".join(collected)
            session.memory.add("assistant", full_reply)

            done = DoneMessage(
                session_id   = session.session_id,
                intent       = intent,
                turn_count   = session.turn_count,
                latency_ms   = (time.perf_counter() - t_start) * 1000,
                memory_stats = session.memory.stats(),
            )
            await websocket.send_text(done.model_dump_json())

    except WebSocketDisconnect:
        store.delete(session.session_id)

    except Exception as exc:                        # noqa: BLE001
        try:
            err = ErrorMessage(code="server_error", message=str(exc))
            await websocket.send_text(err.model_dump_json())
            await websocket.close(code=1011)
        except Exception:                           # noqa: BLE001
            pass
        finally:
            store.delete(session.session_id)
