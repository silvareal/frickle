"""Settings from environment, one source of truth. No os.getenv elsewhere."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://demo:demo@postgres:5432/anomaly"
    ahnlich_host: str = "ahnlich"
    ahnlich_port: int = 1369
    # Text embedder for the pipeline's text track: "sentence_transformers" (in-process)
    # or "ahnlich_ai" (offloaded to the ahnlich-ai proxy). Worker and service must
    # agree so stored and online vectors match.
    embedder: str = "sentence_transformers"
    ahnlich_ai_host: str = "ahnlich-ai"
    ahnlich_ai_port: int = 1370
    artifact_dir: str = "/artifacts"
    store_name: str = "transactions"
    knn_k: int = 5
    # Expected anomaly prevalence for Isolation Forest. A business threshold, set
    # near the synthetic outlier rate so the store has a meaningful flagged
    # population to vote on. "auto" leaves it to the model (sklearn default).
    if_contamination: float = 0.10
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
