"""Shared fixtures. Container-backed fixtures use real Ahnlich / Postgres images
(testcontainers) — the SDK shape and the SQL are the things most likely to be
wrong, so they are exercised against the real thing, not mocks."""

from __future__ import annotations

import os
import socket
import time

import pytest

AHNLICH_IMAGE = "ghcr.io/deven96/ahnlich-db:latest"
AHNLICH_PORT = 1369


def _wait_port(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{host}:{port} not ready")


@pytest.fixture(scope="session")
def ahnlich_endpoint():
    # If an endpoint is supplied (e.g. tests run on the compose network against a
    # dedicated Ahnlich), use it directly; otherwise spin one up with testcontainers.
    host_env, port_env = os.getenv("TEST_AHNLICH_HOST"), os.getenv("TEST_AHNLICH_PORT")
    if host_env and port_env:
        _wait_port(host_env, int(port_env))
        yield host_env, int(port_env)
        return

    from testcontainers.core.container import DockerContainer

    container = (
        DockerContainer(AHNLICH_IMAGE)
        .with_exposed_ports(AHNLICH_PORT)
        .with_command(f"ahnlich-db run --host 0.0.0.0 --port {AHNLICH_PORT}")
    )
    container.start()
    try:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(AHNLICH_PORT))
        _wait_port(host, port)
        yield host, port
    finally:
        container.stop()


@pytest.fixture(scope="session")
def postgres_dsn():
    dsn_env = os.getenv("TEST_DATABASE_URL")
    if dsn_env:
        yield dsn_env
        return

    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        dsn = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
        yield dsn
