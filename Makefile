PY=python
PIP=pip

.PHONY: install test run-api run-mock-tiles bench bench-local

install:
	$(PIP) install -r requirements.txt

test:
	pytest -q

run-api:
	uvicorn src.day03_api.app:app --reload

run-mock-tiles:
	uvicorn src.day01_concurrency.mock_tile_server:app --reload --port 8001

# Run the Day 1 benchmark against OSM (default)
bench:
	$(PY) -m src.day01_concurrency.tile_fetcher --tile-count 20 --max-concurrency 10

# Run the Day 1 benchmark against local mock server
bench-local:
	TILE_BASE_URL=http://127.0.0.1:8001 $(PY) -m src.day01_concurrency.tile_fetcher --tile-count 50 --max-concurrency 20 --timeout 2 --retries 2

