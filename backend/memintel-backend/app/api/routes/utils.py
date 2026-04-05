"""
app/api/routes/utils.py
──────────────────────────────────────────────────────────────────────────────
SSE formatting helpers for streaming endpoints.

POST /tasks and POST /concepts/compile support opt-in SSE streaming via
the stream=True request field. This module provides the formatting helpers
used by both route handlers.

SSE wire format (per RFC 8895)
──────────────────────────────
Each event is exactly two lines followed by a blank line:
  event: <event_type>\n
  data: <json_payload>\n
  \n

The blank line after every event is mandatory — required for correct client
parsing. Content-Type must be text/event-stream.

Event types: cor_step, cor_complete, cor_error.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator


def sse_event(event_type: str, data: dict) -> str:
    """
    Format one SSE event as a string ready to be sent over the wire.

    Returns exactly:
      event: <event_type>\n
      data: <json_payload>\n
      \n
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def sse_generator(
    stream: AsyncGenerator[dict, None],
) -> AsyncGenerator[str, None]:
    """
    Convert a stream of ``{"event_type": str, "data": dict}`` dicts to
    SSE-formatted strings.

    Catches any unexpected exception from the upstream generator and
    emits a cor_error terminal event so the client always receives a
    well-formed closure.
    """
    try:
        async for event in stream:
            yield sse_event(event["event_type"], event["data"])
    except Exception as exc:
        yield sse_event("cor_error", {"failure_reason": str(exc)})
