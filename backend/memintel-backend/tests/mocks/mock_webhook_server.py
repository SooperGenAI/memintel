"""
tests/mocks/mock_webhook_server.py
──────────────────────────────────────────────────────────────────────────────
MockWebhookServer — intercepts and records outbound httpx calls from
ActionTrigger without making real network requests.

Implementation approach
───────────────────────
ActionTrigger._invoke_webhook() creates a new httpx.AsyncClient on every
webhook delivery call.  MockWebhookServer patches
``app.runtime.action_trigger.httpx.AsyncClient`` with a drop-in replacement
that records every request it receives and returns a configurable response.

No real HTTP server is started.  No network calls are made.

Usage
─────
    server = MockWebhookServer()
    with server:
        # All httpx.AsyncClient calls in action_trigger are intercepted.
        ...  # run code that fires actions
        assert server.call_count == 1
        assert server.last_request.method == "POST"
        body = server.last_request.json()

Configuring responses
─────────────────────
    server.configure_response(status_code=500)          # simulate server failure
    server.configure_response(status_code=200, body={"ok": True})
    server.reset()                                       # clear recorded requests

Inspection
──────────
    server.requests      — list[RecordedRequest], all intercepted requests
    server.last_request  — most recent RecordedRequest (None if no calls)
    server.call_count    — total number of intercepted calls
"""
from __future__ import annotations

import json as json_module
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import httpx


@dataclass
class RecordedRequest:
    """
    A single intercepted outbound HTTP request captured by MockWebhookServer.

    Attributes
    ----------
    method:      HTTP method (always uppercase, e.g. 'POST').
    url:         Full URL string the request was addressed to.
    headers:     Request headers dict, as seen after ${ENV_VAR} resolution
                 performed by ActionTrigger._resolve_env_vars().
    _body_bytes: Raw request body bytes (JSON-encoded payload); None when
                 no body was sent (e.g. GET requests).
    """

    method: str
    url: str
    headers: dict[str, str]
    _body_bytes: bytes | None = field(default=None, repr=False)

    def json(self) -> Any:
        """
        Parse and return the request body as a Python object.

        Returns None when no body was sent.
        """
        if not self._body_bytes:
            return None
        return json_module.loads(self._body_bytes)


class MockWebhookServer:
    """
    Test double for the outbound HTTP layer of ActionTrigger.

    Patches ``app.runtime.action_trigger.httpx.AsyncClient`` so all HTTP
    calls made by ActionTrigger (webhook, notification, workflow) are
    intercepted, recorded, and answered with a configurable response —
    no real network calls are made.

    Use as a context manager or call start()/stop() explicitly.

    Parameters
    ----------
    url:           Nominal endpoint URL to register on the action definition.
                   Since all calls are intercepted regardless of URL, this only
                   needs to be a valid string — it never resolves to a real server.
    status_code:   HTTP status code returned for every intercepted request
                   (default 200).
    response_body: JSON-serialisable response body returned for every
                   intercepted request (default: {"status": "received"}).
    """

    DEFAULT_URL: str = "http://mock-webhook.test/deliver"

    def __init__(
        self,
        url: str = DEFAULT_URL,
        status_code: int = 200,
        response_body: dict[str, Any] | None = None,
    ) -> None:
        self._url = url
        self._status_code = status_code
        self._response_body: dict[str, Any] = (
            response_body if response_body is not None else {"status": "received"}
        )
        self._requests: list[RecordedRequest] = []
        self._patcher: Any = None

    # ── Inspection ────────────────────────────────────────────────────────────

    @property
    def url(self) -> str:
        """Nominal endpoint URL — register this on webhook action definitions."""
        return self._url

    @property
    def requests(self) -> list[RecordedRequest]:
        """All intercepted requests in chronological order (copy)."""
        return list(self._requests)

    @property
    def call_count(self) -> int:
        """Total number of intercepted HTTP calls."""
        return len(self._requests)

    @property
    def last_request(self) -> RecordedRequest | None:
        """Most recent intercepted request, or None if no calls were made."""
        return self._requests[-1] if self._requests else None

    # ── Configuration ──────────────────────────────────────────────────────────

    def configure_response(
        self,
        status_code: int = 200,
        body: dict[str, Any] | None = None,
    ) -> None:
        """
        Change the response returned for subsequent intercepted requests.

        Effective immediately — does not affect already-recorded requests.
        """
        self._status_code = status_code
        if body is not None:
            self._response_body = body

    def reset(self) -> None:
        """Clear all recorded requests without affecting response configuration."""
        self._requests.clear()

    # ── Mock client factory ────────────────────────────────────────────────────

    def _build_mock_client_class(self) -> type:
        """
        Return an httpx.AsyncClient drop-in that records requests and returns
        configurable httpx.Response objects without making network calls.

        The returned class must be compatible with::

            async with httpx.AsyncClient(timeout=...) as client:
                resp = await client.request(method, url, json=..., headers=...)
                resp.raise_for_status()  # works correctly for 4xx/5xx

        We return a real httpx.Response so that raise_for_status() behaves
        exactly as in production — HTTP 4xx/5xx raise httpx.HTTPStatusError.
        """
        server = self

        class _MockAsyncClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass  # ignore timeout and other real-client constructor kwargs

            async def __aenter__(self) -> "_MockAsyncClient":
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

            async def request(
                self,
                method: str,
                url: Any,
                *,
                json: Any = None,
                headers: dict[str, str] | None = None,
                **kwargs: Any,
            ) -> httpx.Response:
                # Record this outbound call.
                body_bytes = (
                    json_module.dumps(json).encode("utf-8")
                    if json is not None
                    else None
                )
                server._requests.append(
                    RecordedRequest(
                        method=method.upper(),
                        url=str(url),
                        headers=dict(headers or {}),
                        _body_bytes=body_bytes,
                    )
                )

                # Return a real httpx.Response so raise_for_status() behaves
                # correctly for both success (2xx) and failure (4xx/5xx) scenarios.
                return httpx.Response(
                    status_code=server._status_code,
                    content=json_module.dumps(server._response_body).encode("utf-8"),
                    headers={"content-type": "application/json"},
                    request=httpx.Request(method, str(url)),
                )

            # Support .post() used by notification/workflow backends (belt-and-braces).
            async def post(
                self,
                url: Any,
                *,
                json: Any = None,
                headers: dict[str, str] | None = None,
                **kwargs: Any,
            ) -> httpx.Response:
                return await self.request(
                    "POST", url, json=json, headers=headers, **kwargs
                )

        return _MockAsyncClient

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "MockWebhookServer":
        """Activate the patch and begin intercepting httpx calls. Returns self."""
        mock_class = self._build_mock_client_class()
        self._patcher = patch(
            "app.runtime.action_trigger.httpx.AsyncClient",
            mock_class,
        )
        self._patcher.start()
        return self

    def stop(self) -> None:
        """Deactivate the patch and restore real httpx.AsyncClient."""
        if self._patcher is not None:
            self._patcher.stop()
            self._patcher = None

    def __enter__(self) -> "MockWebhookServer":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()
