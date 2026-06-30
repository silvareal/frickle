"""Route dependencies — pull the shared AppState off the FastAPI app."""

from __future__ import annotations

from fastapi import Request

from .state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.appstate
