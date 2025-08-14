# Python Geospatial Engineering Practices

Hands-on modules covering async I/O, providers, APIs, testing, data, and performance for real-world geospatial systems. Each module is runnable with focused drills and tests.

## Quick Start
- Python 3.11+
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt`
- Makefile (optional): `make install`, `make test`, `make run-api`, `make bench`, `make bench-local`

## Project Structure

```text
src/
├─ day01_concurrency/      # Async I/O and bounded concurrency
│  ├─ tile_fetcher.py      # CLI benchmark with retries/timeouts
│  └─ mock_tile_server.py  # Local FastAPI for safe latency/error simulation
├─ day02_oop/              # Providers and clean interfaces
│  └─ providers/
│     ├─ base.py | factory.py | csv_provider.py | geojson_provider.py | mvt_provider.py
├─ day03_api/              # FastAPI drills (tiles + bbox streaming, metrics)
│  └─ app.py
├─ day04_testing/          # Tests and examples
│  └─ tests/
│     ├─ test_smoke.py | test_day03_api_validation.py | test_concurrency_day01.py
├─ day05_data/             # Data + geospatial (Protobuf, R-tree)
│  └─ protobuf/road_segment.proto
├─ day06_perf/             # Performance and observability
└─ day07_mock/             # End-to-end mock project
   └─ mock_test/
      ├─ api.py | road_network.py | roads.csv
```

## Run Examples
- Day 1 benchmark: `python -m src.day01_concurrency.tile_fetcher --tile-count 50 --max-concurrency 20`
- Local tiles (recommended): `make run-mock-tiles` then `make bench-local`
- Day 3 API (dev): `uvicorn src.day03_api.app:app --reload` (docs at `/docs`, metrics at `/metrics`)
- Day 7 mock API: `uvicorn src.day07_mock.mock_test.api:app --reload`

## Testing
- Run all: `pytest -q`
- Targeted runs:
  - `pytest -q src/day04_testing/tests/test_smoke.py`
  - `pytest -q src/day04_testing/tests/test_day03_api_validation.py`
  - `pytest -q src/day04_testing/tests/test_concurrency_day01.py`

## Docs Site (optional)
Generate a MkDocs + Material site from this repo:
- `python convert_to_mkdocs.py`
- `cd docs && pip install -r requirements.txt && mkdocs serve`

See `PLAN.md` for goals per day, `AGENTS.md` for contributor guidelines, and `MOCK_TEST.md` for the final mock brief.