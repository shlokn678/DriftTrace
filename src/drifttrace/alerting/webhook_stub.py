"""Local webhook stub receiver (FR-11.5).

A tiny FastAPI app that accepts webhook POSTs and records them in memory so a demo or
test can prove an event was received. No third-party notification provider is involved
(FR-11.5, NFR-7). FastAPI is imported lazily so this module can be imported for its
in-memory store logic without the serving extra installed.

Endpoints:
    GET  /health   -> liveness
    POST /alert    -> record a received payload
    GET  /received -> list recorded payloads (proof of receipt)
    GET  /count    -> number of received payloads
    POST /reset    -> clear the store (test/demo convenience)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReceivedStore:
    """In-memory record of received webhook payloads (observable state)."""

    items: list[dict[str, Any]] = field(default_factory=list)

    def add(self, payload: dict[str, Any]) -> None:
        self.items.append(payload)

    def clear(self) -> None:
        self.items.clear()

    @property
    def count(self) -> int:
        return len(self.items)


# Module-level store so the running process accumulates receipts across requests.
STORE = ReceivedStore()


def create_app() -> Any:
    """Build the webhook-stub FastAPI app. Imports FastAPI lazily."""
    from fastapi import Body, FastAPI

    app = FastAPI(title="DriftTrace Webhook Stub", version="1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/alert")
    def alert(payload: dict = Body(...)) -> dict:  # noqa: B008 - FastAPI dependency
        STORE.add(payload)
        return {"received": True, "count": STORE.count}

    @app.get("/received")
    def received() -> dict:
        return {"count": STORE.count, "items": STORE.items}

    @app.get("/count")
    def count() -> dict:
        return {"count": STORE.count}

    @app.post("/reset")
    def reset() -> dict:
        STORE.clear()
        return {"count": STORE.count}

    return app
