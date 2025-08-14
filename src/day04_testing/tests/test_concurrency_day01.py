import asyncio
from typing import List, Tuple

import httpx
import pytest

from src.day01_concurrency.tile_fetcher import (
    fetch_tile,
    fetch_tiles_concurrently,
)


@pytest.mark.asyncio
async def test_retry_on_500_then_success():
    calls = {"count": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        # First call fails with 500, second succeeds
        if calls["count"] == 1:
            return httpx.Response(500, request=request)
        return httpx.Response(200, content=b"ok", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        content = await fetch_tile(client, 0, 0, 0, timeout_seconds=1.0, retries=2)
        assert content == b"ok"
        assert calls["count"] == 2


@pytest.mark.asyncio
async def test_bounded_concurrency_is_respected():
    max_concurrency = 5
    inflight = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal inflight, max_seen
        async with lock:
            inflight += 1
            max_seen = max(max_seen, inflight)
        # Hold the request briefly to amplify concurrency
        await asyncio.sleep(0.05)
        async with lock:
            inflight -= 1
        return httpx.Response(200, content=b"ok", request=request)

    tiles: List[Tuple[int, int, int]] = [(0, i, i) for i in range(20)]
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        results = await fetch_tiles_concurrently(
            tiles, max_concurrency=max_concurrency, client=client
        )
    assert len(results) == len(tiles)
    assert max_seen <= max_concurrency

