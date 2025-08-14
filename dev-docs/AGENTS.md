# Repository Guidelines

## Project Structure & Modules
- `src/`: package root.
- `src/day01_concurrency/`: async tile fetching drill (`tile_fetcher.py`).
- `src/day02_oop/`: provider abstractions under `providers/`.
- `src/day03_api/`: FastAPI app in `app.py` (`/tiles`, `/stream-features`).
- `src/day04_testing/`: `tests/` with pytest smoke tests.
- `src/day05_data/`: protobuf spec in `protobuf/road_segment.proto`.
- `src/day06_perf/`: performance/observability drills.
- `src/day07_mock/`: mock screening implementation.

## Setup, Build, and Run
- Create venv: `python -m venv .venv && source .venv/bin/activate`.
- Install deps: `pip install -r requirements.txt`.
- Run API (dev): `uvicorn src.day03_api.app:app --reload`.
- Run mock API: `uvicorn src.day07_mock.mock_test.api:app --reload`.
- Day 1 benchmark: `python -m src.day01_concurrency.tile_fetcher`.

## Testing Guidelines
- Framework: pytest (+ pytest-asyncio, hypothesis available).
- Run all tests: `pytest -q`.
- Run a specific test: `pytest -q src/day04_testing/tests/test_smoke.py::test_tiles_smoke`.
- Conventions: tests live in `src/day04_testing/tests/`, files `test_*.py`, functions `test_*`. Target meaningful coverage for new code.

## Coding Style & Naming
- Python 3.11+, PEP 8, 4-space indent, type hints encouraged.
- Names: `snake_case` funcs/vars, `PascalCase` classes, `CONSTANT_CASE` constants.
- Imports: prefer absolute from `src...`; keep modules small and cohesive.
- Formatting: keep lines ≤ 100 chars; if using a formatter, prefer Black defaults. No linter is enforced in-repo.

## Commit & Pull Request Guidelines
- Commits: concise, imperative; prefer Conventional Commits (e.g., `feat(api): add tile bounds check`).
- PRs: describe what/why, link issues, list affected paths, and include curl/examples for API changes.
- Requirements: tests added/updated, `pytest` passes locally, and related READMEs updated.

## Security & Configuration Tips (Optional)
- Do not commit secrets. Add config via env vars if introduced.
- External requests in Day 1 hit OSM; use reasonable timeouts and concurrency.

