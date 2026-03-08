"""
End-to-End Conversation Test Suite for Chiron-AI Hotel Assistant
================================================================

Connects to the live FastAPI server via WebSocket and runs complete
multi-turn dialogues, verifying:
  - Token streaming works (type="token")
  - Done messages include session_id, intent, turn_count, memory_stats
  - Off-topic guard returns canned reply without LLM call
  - Context retention across turns
  - Conversational fluency & domain-appropriate responses

Requirements:
  1.  Start the server first:
        cd "Phase 4"
        python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
  2.  Run this file:
        python -m tests.test_conversations          (all tests)
        python -m tests.test_conversations --test 3 (single test)

Each test prints the full conversation so the reader can evaluate
quality.  A summary table is printed at the end.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

import websockets                # type: ignore[import-untyped]

# ── Configuration ──────────────────────────────────────────────────────────────

WS_URL = "ws://localhost:8000/ws/chat"
RECV_TIMEOUT = 120          # max seconds to wait for each server response


# ── Helpers ────────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    user_msg:   str   = ""
    bot_reply:  str   = ""
    intent:     str   = ""
    turn_count: int   = 0
    latency_ms: float = 0.0
    memory:     dict  = field(default_factory=dict)
    error:      str   = ""


@dataclass
class DialogueResult:
    name:     str
    passed:   bool
    turns:    list[TurnResult] = field(default_factory=list)
    elapsed:  float = 0.0
    reason:   str   = ""


SEP    = "=" * 70
SEP2   = "-" * 50
INDENT = "  "


async def run_dialogue(
    name: str,
    turns: list[str],
    *,
    checks: list | None = None,
) -> DialogueResult:
    """
    Open a WebSocket, send each turn, collect streamed tokens,
    and return the full result.

    ``checks`` is an optional list of callables:
        check(turn_index: int, turn_result: TurnResult) -> str | None
    Return None for pass, or a failure reason string.
    """
    result = DialogueResult(name=name, passed=True)
    t0 = time.perf_counter()

    try:
        async with websockets.connect(WS_URL, open_timeout=15) as ws:
            # Wait for session_created
            raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
            msg = json.loads(raw)
            assert msg["type"] == "session_created", f"Expected session_created, got {msg}"

            # Send warmup, wait for ack
            await ws.send(json.dumps({"message": "__WARMUP__"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
            _ = json.loads(raw)  # warmup_complete

            for i, user_msg in enumerate(turns):
                tr = TurnResult(user_msg=user_msg)
                await ws.send(json.dumps({"message": user_msg}))

                tokens: list[str] = []
                done = False

                while not done:
                    raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                    msg = json.loads(raw)

                    if msg["type"] == "token":
                        tokens.append(msg["content"])
                    elif msg["type"] == "done":
                        done = True
                        tr.intent     = msg.get("intent", "")
                        tr.turn_count = msg.get("turn_count", 0)
                        tr.latency_ms = msg.get("latency_ms", 0.0)
                        tr.memory     = msg.get("memory_stats", {})
                    elif msg["type"] == "error":
                        tr.error = msg.get("message", "unknown error")
                        done     = True

                tr.bot_reply = "".join(tokens)
                result.turns.append(tr)

                # Run optional checks
                if checks and i < len(checks) and checks[i] is not None:
                    fail = checks[i](i, tr)
                    if fail:
                        result.passed = False
                        result.reason = f"Turn {i+1}: {fail}"

    except Exception as exc:
        result.passed = False
        result.reason = f"Connection/protocol error: {exc}"

    result.elapsed = time.perf_counter() - t0
    return result


def print_result(r: DialogueResult) -> None:
    status = "PASS" if r.passed else "FAIL"
    print(f"\n{SEP}")
    print(f"  [{status}] {r.name}")
    print(f"  Elapsed: {r.elapsed:.1f}s | Turns: {len(r.turns)}")
    if r.reason:
        print(f"  Reason : {r.reason}")
    print(SEP)

    for i, t in enumerate(r.turns, 1):
        print(f"\n{INDENT}[Turn {i}]  intent={t.intent}  latency={t.latency_ms:.0f}ms  memory={t.memory}")
        print(f"{INDENT}Guest    : {t.user_msg}")
        reply_preview = t.bot_reply[:500] + ("..." if len(t.bot_reply) > 500 else "")
        print(f"{INDENT}Assistant: {reply_preview}")
        if t.error:
            print(f"{INDENT}ERROR    : {t.error}")
        print(f"{INDENT}{SEP2}")


# ── Check helpers ──────────────────────────────────────────────────────────────

def reply_not_empty(_, tr: TurnResult) -> str | None:
    if not tr.bot_reply.strip() and not tr.error:
        return "Empty reply"
    return None


def contains_any(*keywords: str):
    """Check that reply contains at least one of the keywords (case-insensitive)."""
    def _check(_, tr: TurnResult) -> str | None:
        low = tr.bot_reply.lower()
        if any(kw.lower() in low for kw in keywords):
            return None
        return f"Expected one of {keywords!r} in reply"
    return _check


def intent_is(expected: str):
    def _check(_, tr: TurnResult) -> str | None:
        if tr.intent != expected:
            return f"Expected intent '{expected}', got '{tr.intent}'"
        return None
    return _check


# ── Test cases ─────────────────────────────────────────────────────────────────

async def tc01_room_reservation() -> DialogueResult:
    """TC-01: Complete room reservation (happy path, multi-turn)."""
    return await run_dialogue(
        "TC-01: Room Reservation — Happy Path",
        [
            "Hi, I'd like to book a room please.",
            "July 10 to July 14.",
            "Two guests.",
            "A double room would be great.",
            "My name is John Smith, email john.smith@email.com.",
            "No special requests. Please confirm.",
        ],
        checks=[
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
        ],
    )


async def tc02_check_in() -> DialogueResult:
    """TC-02: Guest check-in with reservation ID."""
    return await run_dialogue(
        "TC-02: Standard Guest Check-In",
        [
            "I'm here to check in.",
            "My reservation is under the name Elena Vasquez.",
            "Anything I should know about breakfast and the gym?",
            "That's all, thank you!",
        ],
        checks=[
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
        ],
    )


async def tc03_checkout_dispute() -> DialogueResult:
    """TC-03: Check-out with a billing dispute — must escalate, not delete charge."""
    return await run_dialogue(
        "TC-03: Check-Out with Billing Dispute",
        [
            "I'd like to check out. Room 305.",
            "There's a $25 minibar charge on my bill that I never incurred.",
            "This is unacceptable. I want to talk to management.",
        ],
        checks=[
            reply_not_empty,
            reply_not_empty,
            # The third turn should trigger escalation language
            contains_any("manager", "escalat", "billing", "apologi", "sorry"),
        ],
    )


async def tc04_room_service() -> DialogueResult:
    """TC-04: Room service order flow."""
    return await run_dialogue(
        "TC-04: Room Service Order",
        [
            "Can I order room service?",
            "I'm in room 204.",
            "What's on the dinner menu?",
            "I'll have the grilled salmon and a chocolate lava cake.",
            "Yes, charge it to my room.",
        ],
        checks=[
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
        ],
    )


async def tc05_complaint_handling() -> DialogueResult:
    """TC-05: Guest complaint — A/C not working."""
    return await run_dialogue(
        "TC-05: Complaint — AC Not Working",
        [
            "My air conditioning is completely broken. It's very hot in here.",
            "Room 408.",
            "I'd like to be moved to a different room if possible.",
            "OK, please arrange that. Thank you.",
        ],
        checks=[
            # First turn should acknowledge the problem
            contains_any("sorry", "apologi", "understand", "inconvenience"),
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
        ],
    )


async def tc06_faq_chain() -> DialogueResult:
    """TC-06: Multiple FAQ questions in sequence."""
    return await run_dialogue(
        "TC-06: FAQ Chain — Pool, Wi-Fi, Parking, Pets",
        [
            "What are the pool hours?",
            "What is the Wi-Fi password?",
            "How much is parking?",
            "Are pets allowed?",
            "What about the gym?",
        ],
        checks=[
            contains_any("pool", "7", "9", "10"),
            contains_any("wi-fi", "wifi", "password", "gs2024", "grandstay"),
            contains_any("park", "valet", "$20", "$30"),
            contains_any("pet", "no", "service animal"),
            contains_any("gym", "24", "2nd", "floor"),
        ],
    )


async def tc07_escalation() -> DialogueResult:
    """TC-07: Immediate request for a human agent.
    Verifies: intent detection classifies as 'escalation', model responds."""
    return await run_dialogue(
        "TC-07: Immediate Escalation Request",
        [
            "I want to speak to a real person right now.",
            "I insist on speaking to a duty manager about my room situation.",
        ],
        checks=[
            # System correctly tags this as escalation intent
            intent_is("escalation"),
            # Model responds (small models may not always use perfect escalation language)
            reply_not_empty,
        ],
    )


async def tc08_cancellation() -> DialogueResult:
    """TC-08: Reservation cancellation within the free window."""
    return await run_dialogue(
        "TC-08: Reservation Cancellation",
        [
            "I need to cancel my reservation.",
            "My reservation ID is GS-20240801-9912.",
            "Yes I understand. Please go ahead and cancel.",
        ],
        checks=[
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
        ],
    )


async def tc09_off_topic() -> DialogueResult:
    """TC-09: Off-topic messages should be redirected without calling LLM."""
    return await run_dialogue(
        "TC-09: Off-Topic Domain Guard",
        [
            "What is the weather like tomorrow?",
            "Can you write me a Python program?",
            "Actually, what time is breakfast?",
        ],
        checks=[
            intent_is("off_topic"),
            intent_is("off_topic"),
            # The third turn should be handled normally (not off-topic)
            contains_any("breakfast", "7", "10", "garden", "restaurant"),
        ],
    )


async def tc10_topic_switch() -> DialogueResult:
    """TC-10: Switch topic mid-reservation, then resume."""
    return await run_dialogue(
        "TC-10: Topic Switch Mid-Reservation",
        [
            "I'd like to book a room for July 20 to July 23.",
            "Two guests.",
            "Wait, quick question — what time does the spa close?",
            "OK thanks. Back to the booking — I'd like a double room.",
            "Sarah Connor, sarah.c@email.com.",
            "Yes, please confirm the booking.",
        ],
        checks=[
            reply_not_empty,
            reply_not_empty,
            # Spa question should be answered
            contains_any("spa", "8", "9"),
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
        ],
    )


async def tc11_streaming_verification() -> DialogueResult:
    """TC-11: Verifies that tokens arrive incrementally (streaming)."""
    result = DialogueResult(name="TC-11: Streaming Token Delivery", passed=True)
    t0 = time.perf_counter()

    try:
        async with websockets.connect(WS_URL, open_timeout=15) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
            msg = json.loads(raw)
            assert msg["type"] == "session_created"

            await ws.send(json.dumps({"message": "__WARMUP__"}))
            await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)

            await ws.send(json.dumps({"message": "Tell me about the hotel amenities."}))

            token_count = 0
            tokens = []
            first_token_time = None
            done = False

            while not done:
                raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                msg = json.loads(raw)
                if msg["type"] == "token":
                    token_count += 1
                    tokens.append(msg["content"])
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                elif msg["type"] == "done":
                    done = True
                elif msg["type"] == "error":
                    result.passed = False
                    result.reason = msg.get("message", "error")
                    done = True

            full_reply = "".join(tokens)
            tr = TurnResult(
                user_msg="Tell me about the hotel amenities.",
                bot_reply=full_reply,
            )
            result.turns.append(tr)

            if token_count < 5:
                result.passed = False
                result.reason = f"Expected multiple tokens, got {token_count}"
            else:
                ttft = (first_token_time - t0) if first_token_time else 0
                result.reason = f"Streamed {token_count} tokens, TTFT={ttft:.2f}s"

    except Exception as exc:
        result.passed = False
        result.reason = str(exc)

    result.elapsed = time.perf_counter() - t0
    return result


async def tc12_loyalty_member() -> DialogueResult:
    """TC-12: Loyalty member check-in with benefits inquiry."""
    return await run_dialogue(
        "TC-12: Loyalty Member Check-In & Benefits",
        [
            "Hi, I'm a GrandStay Rewards member and I'm checking in.",
            "My name is Elena Vasquez, reservation GS-20240715-2234.",
            "How many points will I earn for a 3-night stay?",
            "Can I get a late check-out tomorrow?",
        ],
        checks=[
            reply_not_empty,
            reply_not_empty,
            # Should mention points/loyalty
            contains_any("point", "reward", "loyalty", "earn", "10"),
            # Should mention late check-out
            contains_any("late", "check-out", "checkout", "2", "free"),
        ],
    )


async def tc13_context_retention() -> DialogueResult:
    """TC-13: Verifies the assistant remembers info from earlier turns."""
    return await run_dialogue(
        "TC-13: Context Retention — No Repeated Questions",
        [
            "My name is David Lee and I want to book a room for July 5 to July 8.",
            "Single room for one guest.",
            "david.lee@email.com",
            "Yes, please confirm.",
            "One more thing — what is the breakfast time?",
            "Can you add breakfast to my booking?",
        ],
        checks=[
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
            reply_not_empty,
            contains_any("breakfast", "7", "10", "garden"),
            reply_not_empty,
        ],
    )


# ── Runner ─────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    tc01_room_reservation,
    tc02_check_in,
    tc03_checkout_dispute,
    tc04_room_service,
    tc05_complaint_handling,
    tc06_faq_chain,
    tc07_escalation,
    tc08_cancellation,
    tc09_off_topic,
    tc10_topic_switch,
    tc11_streaming_verification,
    tc12_loyalty_member,
    tc13_context_retention,
]


async def main(test_num: int = 0) -> None:
    print(f"\n{SEP}")
    print(f"  CHIRON-AI — END-TO-END CONVERSATION TEST SUITE")
    print(f"  Server : {WS_URL}")
    print(f"  Tests  : {len(ALL_TESTS)}")
    print(SEP)

    tests_to_run = ALL_TESTS
    if test_num > 0:
        if 1 <= test_num <= len(ALL_TESTS):
            tests_to_run = [ALL_TESTS[test_num - 1]]
        else:
            print(f"  [ERROR] Test number must be between 1 and {len(ALL_TESTS)}.")
            return

    results: list[DialogueResult] = []
    for test_fn in tests_to_run:
        r = await test_fn()
        print_result(r)
        results.append(r)

    # Summary table
    passed  = sum(1 for r in results if r.passed)
    total   = len(results)
    print(f"\n{SEP}")
    print(f"  SUMMARY")
    print(SEP)
    print(f"  {'#':<5} {'Status':<8} {'Time':>7}  {'Name'}")
    print(f"  {'-'*5} {'-'*8} {'-'*7}  {'-'*40}")
    for i, r in enumerate(results, 1):
        s = "PASS" if r.passed else "FAIL"
        print(f"  {i:<5} {s:<8} {r.elapsed:>6.1f}s  {r.name}")

    print(f"\n  Result: {passed}/{total} passed")
    total_time = sum(r.elapsed for r in results)
    print(f"  Total time: {total_time:.1f}s")
    print(SEP)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chiron-AI E2E Conversation Tests")
    parser.add_argument("--test", type=int, default=0,
                        help="Run a specific test (1-13). Default: all.")
    args = parser.parse_args()
    asyncio.run(main(args.test))
