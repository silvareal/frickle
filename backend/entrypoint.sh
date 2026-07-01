#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import socket, time, urllib.parse, os
def wait(host, port, name):
    for _ in range(60):
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"{name} reachable at {host}:{port}")
                return
        except OSError:
            time.sleep(1)
    raise SystemExit(f"timed out waiting for {name} ({host}:{port})")

dsn = os.environ.get("DATABASE_URL", "postgresql://demo:demo@postgres:5432/anomaly")
u = urllib.parse.urlparse(dsn)
wait(u.hostname, u.port or 5432, "postgres")
wait(os.environ.get("AHNLICH_HOST", "ahnlich"), int(os.environ.get("AHNLICH_PORT", "1369")), "ahnlich")
if os.environ.get("EMBEDDER") == "ahnlich_ai":
    wait(os.environ.get("AHNLICH_AI_HOST", "ahnlich-ai"), int(os.environ.get("AHNLICH_AI_PORT", "1370")), "ahnlich-ai")
PY

echo "Running database migrations..."
alembic upgrade head

exec "$@"
