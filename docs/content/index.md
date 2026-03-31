# Python Geospatial Engineering Guide

Production-oriented geospatial curriculum for Python and backend engineers moving into spatial systems.

This guide is built around code, services, testing, query behavior, and debugging. It is not a GIS desktop course, and it is not a notebook-only tour.

[Start with the Study Plan](study_plan.md){ .md-button .md-button--primary }
[Open the API Module](day03_api/index.md){ .md-button }
[Jump to the Capstone](day07_mock/index.md){ .md-button }

## Why This Guide Exists

Most geospatial learning material either:

- assumes a GIS classroom workflow
- stays too close to notebooks and one-off analysis
- or jumps straight into architecture talk without enough runnable code

This repo aims for a narrower path:

- geospatial correctness first
- clear provider and data boundaries
- realistic storage and query thinking
- FastAPI services and tile-style endpoints
- testing, metrics, and performance work around actual code

## Start Here

<div class="grid cards" markdown>

- **Study Plan**

  Read the [study plan](study_plan.md) first. It defines the core path, what the repo supports now, and what is still planned expansion.

- **Day 03: APIs and Tile Delivery**

  The strongest current service-oriented module is [Day 03](day03_api/index.md). It is the best entry point if you want concrete FastAPI work.

- **Day 07: Capstone**

  The [capstone](day07_mock/index.md) is the portfolio-oriented end state: a small road-network query service with validation, tests, and performance notes.

- **Day 04: Testing**

  Use [Day 04](day04_testing/index.md) to see how API validation, content types, and geospatial edge cases should be tested.

</div>

## Who This Is For

- Python and backend engineers entering geospatial systems
- data engineers who need to work with spatial data, APIs, and query services
- teams that want more disciplined geospatial service patterns than ad hoc scripts

## What You Can Do In This Repo Today

- run a bounded-concurrency tile-fetch benchmark
- inspect provider patterns across multiple data shapes
- run a small FastAPI app with tile-style and streamed-feature endpoints
- expose Prometheus metrics from that app
- run focused tests around API validation and concurrency behavior
- inspect a simple road-network mock project and capstone brief

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Useful commands:

```bash
# Run the Day 3 API
uvicorn src.day03_api.app:app --reload

# Run focused tests
.venv/bin/python -m pytest -q src/day04_testing/tests/test_smoke.py
.venv/bin/python -m pytest -q src/day04_testing/tests/test_day03_api_validation.py

# Run the Day 1 benchmark
python -m src.day01_concurrency.tile_fetcher --tile-count 50 --max-concurrency 20
```

Docs locally:

```bash
cd docs
pip install -r requirements.txt
mkdocs serve
```

## Suggested Route

1. Read the [Study Plan](study_plan.md).
2. Skim [Day 03](day03_api/index.md) and [Day 04](day04_testing/index.md) to see the current strongest code-backed path.
3. Use [Day 01](day01_concurrency/index.md) as a boundary/integration drill, not as the conceptual start of geospatial learning.
4. Finish by narrowing the [capstone](day07_mock/index.md) into something portfolio-worthy.

## Current Status

### Implemented now

- concurrency drill with retries and bounded concurrency
- provider examples for multiple data shapes
- small FastAPI API with metrics
- initial API and concurrency tests
- road-network mock project

### Planned next

- stronger spatial correctness drills
- more realistic storage and query work, with PostGIS as a core direction
- better tile validation and tile delivery behavior
- stronger capstone realism

## Project Structure

```text
src/
├─ day01_concurrency/      # Async I/O and bounded concurrency drills
├─ day02_oop/              # Provider boundaries and interface patterns
├─ day03_api/              # FastAPI API drills (tiles, bbox streaming, metrics)
├─ day04_testing/          # Focused tests for API and concurrency behavior
├─ day05_data/             # Data and indexing examples
├─ day06_perf/             # Performance and observability exercises
└─ day07_mock/             # Capstone mock project
```

## Live Docs

- Site: [rgdonohue.github.io/geospatial-python-guide](https://rgdonohue.github.io/geospatial-python-guide/)
