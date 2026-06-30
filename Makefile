.PHONY: up down logs seed sync test lint loadtest

up:
	docker compose up --build -d

down:
	docker compose down -v

logs:
	docker compose logs -f

# Seed ≥1500 synthetic rows (truncate-then-load), then build the store + labels.
seed:
	docker compose exec -T backend python -m data.seed --count 2200 --fraud-rate 0.08 --seed 42
	docker compose exec -T backend python -m worker.retrain

sync:
	docker compose exec backend python -m worker.retrain

# Tests run against the compose Ahnlich and a dedicated test database, so they
# never touch the demo data. The skew + classify tests need no services.
test:
	-docker compose exec -T postgres createdb -U demo anomaly_test 2>/dev/null || true
	docker compose exec -T \
		-e TEST_AHNLICH_HOST=ahnlich -e TEST_AHNLICH_PORT=1369 \
		-e TEST_DATABASE_URL=postgresql://demo:demo@postgres:5432/anomaly_test \
		backend pytest -q

lint:
	docker compose exec -T backend ruff check .
	docker compose exec -T backend mypy app worker data
	cd frontend && npm run lint

loadtest:
	docker compose exec backend python -m scripts.loadtest --url http://localhost:8000 --n 200
