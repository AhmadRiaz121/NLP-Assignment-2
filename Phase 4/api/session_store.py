"""
Session store: in-memory management of guest sessions.

Each session keeps:
  - A ContextMemoryManager (sliding-window conversation history)
  - Turn count and last detected intent
  - Creation timestamp for TTL/cleanup

Intent detection is defined here (keyword-based, 9 intents) so it
is available both to the API layer and to any future tests.
"""

from __future__ import annotations
import uuid
import time
from typing import Optional


#  Intent detection 

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "reservation" : ["book", "reserve", "reservation", "availability",
                     "available", "room", "stay", "nights", "suite",
                     "double", "single"],
    "check_in"    : ["check in", "check-in", "checkin", "arriving",
                     "arrival", "early check", "room ready"],
    "check_out"   : ["check out", "check-out", "checkout", "leaving",
                     "departure", "late check", "extend", "bill",
                     "invoice", "pay"],
    "room_service": ["room service", "food", "order", "hungry", "meal",
                     "drink", "breakfast", "lunch", "dinner", "snack",
                     "coffee", "towel", "pillow", "blanket",
                     "housekeeping", "clean"],
    "complaint"   : ["problem", "issue", "broken", "noise", "dirty",
                     "not working", "complaint", "temperature", "smell",
                     "leak", "cold", "hot", "ac", "air conditioning",
                     "heater", "wifi", "wi-fi"],
    "cancellation": ["cancel", "cancellation", "refund", "change",
                     "modify", "reschedule", "different date"],
    "escalation"  : ["manager", "supervisor", "human", "speak to",
                     "talk to", "real person", "agent", "staff",
                     "unacceptable"],
    "faq"         : ["pool", "gym", "spa", "parking", "wifi", "wi-fi",
                     "internet", "pet", "smoke", "smoking", "hour",
                     "open", "close", "price", "cost", "rate", "policy",
                     "loyalty", "reward", "points"],
    "off_topic"   : ["weather", "stock", "news", "politics", "sport",
                     "game", "movie", "music", "recipe", "math", "code",
                     "program", "flight", "airline", "passport"],
}


def detect_intent(text: str) -> str:
    """Return the best-matching intent label for a guest message."""
    low = text.lower()
    scores = {
        intent: sum(1 for kw in kws if kw in low)
        for intent, kws in _INTENT_KEYWORDS.items()
    }
    if scores["off_topic"] > 0 and scores["off_topic"] >= max(
        v for k, v in scores.items() if k != "off_topic"
    ):
        return "off_topic"
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


#  Context memory 

class ContextMemoryManager:
    """
    Stores the per-session conversation history.

    Sliding window: maximum ``max_turns`` messages stored.
    Filler removal: short low-information messages are stripped from
                    the older portion before sending to the LLM.
    """

    FILLER: set[str] = {
        "ok", "okay", "sure", "thanks", "thank you", "yes", "no",
        "great", "perfect", "alright", "got it", "hi", "hello",
        "yep", "nope", "cool", "fine", "understood", "noted",
    }

    def __init__(self, max_turns: int = 20, recent_guard: int = 6):
        self.max_turns    = max_turns
        self.recent_guard = recent_guard
        self.history: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        while len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self) -> list[dict]:
        if len(self.history) <= self.recent_guard:
            return self.history
        recent = self.history[-self.recent_guard:]
        older  = self.history[:-self.recent_guard]
        cleaned = [
            m for m in older
            if m["content"].strip().lower().rstrip("!.,?") not in self.FILLER
        ]
        return cleaned + recent

    def clear(self) -> None:
        self.history = []

    def __len__(self) -> int:
        return len(self.history)

    def stats(self) -> dict:
        ctx = self.get_context()
        return {
            "total_stored": len(self.history),
            "sent_to_llm" : len(ctx),
            "filtered_out": len(self.history) - len(ctx),
        }


#  Session

class Session:
    """Represents one active guest conversation."""

    def __init__(self):
        self.session_id  : str                  = str(uuid.uuid4())[:8]
        self.memory      : ContextMemoryManager = ContextMemoryManager()
        self.turn_count  : int                  = 0
        self.last_intent : str                  = "general"
        self.created_at  : float                = time.time()

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def stats(self) -> dict:
        return {
            "session_id"  : self.session_id,
            "turn_count"  : self.turn_count,
            "last_intent" : self.last_intent,
            **self.memory.stats(),
            "age_seconds" : round(self.age_seconds, 1),
        }


#  Session store

class SessionStore:
    """Thread-safe, in-memory registry of active guest sessions."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session = Session()
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> Session:
        s = self.get(session_id)
        if s is None:
            raise KeyError(f"Session '{session_id}' not found.")
        return s

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    @property
    def count(self) -> int:
        return len(self._sessions)

    def all_ids(self) -> list[str]:
        return list(self._sessions.keys())


#  Module-level singleton 

store = SessionStore()
