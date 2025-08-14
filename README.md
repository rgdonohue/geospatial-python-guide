# Python Geospatial Engineering Practices

A practical, hands-on curriculum for software engineers working with geospatial data. Build real skills by implementing concurrency, clean provider abstractions, FastAPI services, testing strategies, data formats, and performance techniques that show up in production geospatial systems.

Live docs: `https://rgdonohue.github.io/geospatial-python-guide`

Who this is for:
- Engineers moving into geospatial systems who want practical patterns (not toy examples)
- Senior Python/Backend engineers who want fast drills with runnable code and tests
- Teams standardizing on clear interfaces for providers, APIs, and data pipelines

## Quick Start
- Python 3.11+
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt`
- Makefile (optional): `make install`, `make test`, `make run-api`, `make bench`, `make bench-local`

Docs locally (optional):
- `cd docs && pip install -r requirements.txt && mkdocs serve`

## Project Structure (geospatial-focused modules)

```text
src/
├─ day01_concurrency/      # Async I/O and bounded concurrency
│  ├─ tile_fetcher.py      # CLI benchmark with retries/timeouts
│  └─ mock_tile_server.py  # Local FastAPI for safe latency/error simulation
├─ day02_oop/              # Providers and clean interfaces
│  └─ providers/
│     ├─ base.py | factory.py | csv_provider.py | geojson_provider.py | mvt_provider.py
├─ day03_api/              # FastAPI drills (tiles + bbox streaming, metrics)
│  └─ app.py               # Tile + bbox streaming endpoints, metrics
├─ day04_testing/          # Tests and examples
│  └─ tests/
│     ├─ test_smoke.py | test_day03_api_validation.py | test_concurrency_day01.py
├─ day05_data/             # Data + geospatial (Protobuf, R-tree)
│  └─ protobuf/road_segment.proto  # Example schema for road segments
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

## Documentation
- Live site: `https://rgdonohue.github.io/geospatial-python-guide`
- Build locally:
  - `python convert_to_mkdocs.py` (optional content refresh)
  - `cd docs && pip install -r requirements.txt && mkdocs serve`
  - Source markdown lives in `docs/content/`

## Contributing
- Issues and PRs welcome. Focus on clarity, runnable drills, and test coverage.
- See `AGENTS.md` for contributor guidelines.

See `PLAN.md` for goals per day and `MOCK_TEST.md` for the final mock brief.