"""Persistence for fitted pipeline + Isolation Forest + manifest. The manifest is
the active-store pointer: the worker builds a new store in the inactive slot and
flips this pointer only after verification, so a partial run leaves the previous
store (and pointer) intact. Losing the Ahnlich node is a non-event — everything
rebuilds from Postgres."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import joblib
from feature_pipeline import UnifiedFeaturePipeline

_PIPELINE_FILE = "pipeline.joblib"
_MODEL_FILE = "model.joblib"
_MANIFEST_FILE = "manifest.json"


@dataclass
class Artifacts:
    pipeline: UnifiedFeaturePipeline
    model: Any
    manifest: dict[str, Any]


def manifest_path(artifact_dir: str) -> str:
    return os.path.join(artifact_dir, _MANIFEST_FILE)


def load_manifest(artifact_dir: str) -> dict[str, Any] | None:
    path = manifest_path(artifact_dir)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def next_store_slot(manifest: dict[str, Any] | None, base_name: str) -> tuple[str, str | None]:
    """Return (new_store_name, old_store_name). Two slots alternate so the build
    never touches the live store in place."""
    current = manifest.get("active_store") if manifest else None
    current_slot = manifest.get("slot", 1) if manifest else 1
    new_slot = 0 if current_slot == 1 else 1
    return f"{base_name}_{new_slot}", current


def save_artifacts(
    artifact_dir: str,
    pipeline: UnifiedFeaturePipeline,
    model: Any,
    active_store: str,
    slot: int,
) -> None:
    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(artifact_dir, _PIPELINE_FILE))
    joblib.dump(model, os.path.join(artifact_dir, _MODEL_FILE))
    manifest = dict(pipeline.manifest(), active_store=active_store, slot=slot)
    # Write to a temp file then rename — the pointer flip is atomic.
    tmp = manifest_path(artifact_dir) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, manifest_path(artifact_dir))


def load_artifacts(artifact_dir: str) -> Artifacts | None:
    manifest = load_manifest(artifact_dir)
    pipeline_file = os.path.join(artifact_dir, _PIPELINE_FILE)
    if manifest is None or not os.path.exists(pipeline_file):
        return None
    pipeline = joblib.load(pipeline_file)
    model = joblib.load(os.path.join(artifact_dir, _MODEL_FILE))
    return Artifacts(pipeline=pipeline, model=model, manifest=manifest)
