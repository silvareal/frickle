"""Shared application state: pooled DB, one gRPC client, settings, and the
lazily-(re)loaded artifacts. Constructed once in the FastAPI lifespan."""

from __future__ import annotations

from .ahnlich import AhnlichClient
from .artifacts import Artifacts, load_artifacts, load_manifest
from .config import Settings
from .db import Database


class AppState:
    def __init__(self, settings: Settings, db: Database, ahnlich: AhnlichClient) -> None:
        self.settings = settings
        self.db = db
        self.ahnlich = ahnlich
        self._artifacts: Artifacts | None = None

    def refresh_artifacts(self) -> Artifacts | None:
        self._artifacts = load_artifacts(self.settings.artifact_dir)
        return self._artifacts

    @property
    def artifacts(self) -> Artifacts | None:
        # The retrain worker runs out-of-process and flips the active-store pointer
        # (then drops the old store). Re-read the lightweight manifest and reload the
        # cached artifacts whenever the pointer has moved, so the service never keeps
        # querying a store that retrain has since dropped.
        manifest = load_manifest(self.settings.artifact_dir)
        if manifest is None:
            return None
        stale = (
            self._artifacts is None
            or self._artifacts.manifest.get("active_store") != manifest.get("active_store")
        )
        if stale:
            self.refresh_artifacts()
        return self._artifacts

    @property
    def pipeline_fitted(self) -> bool:
        return self.artifacts is not None
