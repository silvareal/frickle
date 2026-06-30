FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10

WORKDIR /app

# CPU-only torch first — avoids pulling the multi-GB CUDA build that
# sentence-transformers would otherwise drag in.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Shared pipeline package (changes least often).
COPY pipeline /pipeline
RUN pip install /pipeline

# Install backend dependencies from pyproject alone, so editing source code does
# not re-resolve/re-download the dependency set on every rebuild.
COPY backend/pyproject.toml /app/pyproject.toml
RUN mkdir -p app && touch app/__init__.py && pip install ".[dev]"

# Then the source; register the package without re-resolving dependencies.
COPY backend /app
RUN pip install -e . --no-deps

RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
