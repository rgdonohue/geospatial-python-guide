"""
Tile Fetcher - Asynchronous Concurrency Example

This module demonstrates key concepts in asynchronous programming and concurrent HTTP requests:
- Async/await syntax and event loop management
- Bounded concurrency with semaphores
- Retry logic with exponential backoff
- Performance benchmarking and comparison
- Error handling for network operations

The script fetches map tiles from OpenStreetMap servers and compares sequential vs concurrent performance.
"""

import asyncio
import os
import random
import time
from statistics import median
from typing import Iterable, List, Tuple, Optional

import httpx


# Base URL can be overridden via env var or CLI
TILE_BASE_URL = os.getenv("TILE_BASE_URL", "https://tile.openstreetmap.org")
TILE_URL_TEMPLATE = f"{TILE_BASE_URL}/{{z}}/{{x}}/{{y}}.png"


def generate_sample_tiles(zoom: int = 5, start_x: int = 5, start_y: int = 12, count: int = 20) -> List[Tuple[int, int, int]]:
    """
    Generate a list of sample tile coordinates for testing.
    
    In web mapping, tiles are identified by three coordinates:
    - z: zoom level (higher = more detailed)
    - x: horizontal position within the zoom level
    - y: vertical position within the zoom level
    
    Args:
        zoom: Zoom level for the tiles
        start_x: Starting x coordinate
        start_y: Starting y coordinate  
        count: Number of tiles to generate
        
    Returns:
        List of (z, x, y) tuples representing tile coordinates
    """
    tiles: List[Tuple[int, int, int]] = []
    x, y = start_x, start_y
    for _ in range(count):
        tiles.append((zoom, x, y))
        x += 1
        y += 1
    return tiles


async def fetch_tile(
    client: httpx.AsyncClient,
    z: int,
    x: int,
    y: int,
    timeout_seconds: float = 10.0,
    retries: int = 2,
    backoff_base: float = 0.2,
    backoff_cap: float = 2.0,
) -> bytes:
    """
    Fetch a single tile with retry and backoff.
    
    This function demonstrates several important async programming concepts:
    1. Async HTTP requests with httpx
    2. Retry logic for transient failures
    3. Exponential backoff to avoid overwhelming servers
    4. Proper error handling and classification
    
    Args:
        client: HTTP client for making requests
        z, x, y: Tile coordinates
        timeout_seconds: Per-request timeout
        retries: Maximum number of retry attempts
        backoff_base: Base delay for exponential backoff
        backoff_cap: Maximum delay cap to prevent excessive waits
        
    Returns:
        Tile image data as bytes
        
    Raises:
        httpx.RequestError: For network-level errors
        httpx.HTTPStatusError: For HTTP error responses
    """
    url = TILE_URL_TEMPLATE.format(z=z, x=x, y=y)
    attempt = 0
    
    while True:
        try:
            # Make the HTTP request - this is async and won't block other coroutines
            response = await client.get(url, timeout=timeout_seconds)
            response.raise_for_status()  # Raises exception for 4xx/5xx status codes
            return response.content
            
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            # Determine if this error is retryable
            # Network errors (RequestError) are always retryable
            # HTTP 5xx errors indicate server issues and are retryable
            # HTTP 4xx errors are client errors and should not be retried
            retryable = isinstance(e, httpx.RequestError) or (
                isinstance(e, httpx.HTTPStatusError)
                and 500 <= e.response.status_code < 600
            )
            
            if attempt >= retries or not retryable:
                # Either we've exhausted retries or this is a non-retryable error
                raise
            
            # Calculate backoff delay with jitter to prevent thundering herd
            # Exponential backoff: delay = base * 2^attempt
            # Jitter adds randomness to prevent synchronized retries
            sleep_s = min(backoff_cap, backoff_base * (2 ** attempt)) + random.uniform(0, 0.1)
            await asyncio.sleep(sleep_s)  # Non-blocking sleep
            attempt += 1


async def fetch_tiles_concurrently(
    tiles: Iterable[Tuple[int, int, int]],
    max_concurrency: int = 10,
    timeout_seconds: float = 10.0,
    retries: int = 2,
    client: Optional[httpx.AsyncClient] = None,
) -> List[bytes]:
    """
    Fetch tiles concurrently with bounded concurrency.
    
    This function demonstrates bounded concurrency using asyncio.Semaphore:
    - Limits the number of simultaneous requests to avoid overwhelming servers
    - Creates multiple tasks that run concurrently
    - Uses asyncio.gather to wait for all tasks to complete
    
    Key benefits of bounded concurrency:
    1. Prevents resource exhaustion (too many open connections)
    2. Respects server rate limits
    3. Maintains good performance without being overly aggressive
    
    Args:
        tiles: Iterable of (z, x, y) tile coordinates
        max_concurrency: Maximum number of simultaneous requests
        timeout_seconds: Per-request timeout
        retries: Number of retries per request
        client: Optional HTTP client (if None, creates a new one)
        
    Returns:
        List of tile data bytes in the same order as input tiles
    """
    # Semaphore acts as a "pool" of available slots for concurrent operations
    # Only max_concurrency coroutines can acquire the semaphore simultaneously
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded_fetch(z: int, x: int, y: int) -> bytes:
        """Inner function that respects the concurrency limit."""
        # Acquire semaphore before making request, release when done
        # This ensures we never exceed max_concurrency simultaneous requests
        async with semaphore:
            return await fetch_tile(client_obj, z, x, y, timeout_seconds=timeout_seconds, retries=retries)

    if client is None:
        # Create a new client context manager if none provided
        # This ensures proper cleanup of resources
        async with httpx.AsyncClient() as client_obj:
            # Create a task for each tile - these will run concurrently
            # asyncio.create_task schedules the coroutine to run
            tasks = [asyncio.create_task(_bounded_fetch(z, x, y)) for (z, x, y) in tiles]
            # Wait for all tasks to complete and return results
            return await asyncio.gather(*tasks)
    else:
        # Use provided client (caller is responsible for cleanup)
        client_obj = client
        tasks = [asyncio.create_task(_bounded_fetch(z, x, y)) for (z, x, y) in tiles]
        return await asyncio.gather(*tasks)


async def fetch_tiles_sequential(
    tiles: Iterable[Tuple[int, int, int]],
    timeout_seconds: float = 10.0,
    retries: int = 2,
    client: Optional[httpx.AsyncClient] = None,
) -> List[bytes]:
    """
    Fetch tiles sequentially (one at a time).
    
    This function serves as a baseline for performance comparison.
    While it's simpler than concurrent fetching, it's much slower
    because each request waits for the previous one to complete.
    
    Args:
        tiles: Iterable of (z, x, y) tile coordinates
        timeout_seconds: Per-request timeout
        retries: Number of retries per request
        client: Optional HTTP client (if None, creates a new one)
        
    Returns:
        List of tile data bytes in the same order as input tiles
    """
    results: List[bytes] = []
    
    if client is None:
        # Create new client for this operation
        async with httpx.AsyncClient() as client_obj:
            # Process tiles one by one - this is the key difference from concurrent
            for z, x, y in tiles:
                results.append(
                    await fetch_tile(client_obj, z, x, y, timeout_seconds=timeout_seconds, retries=retries)
                )
    else:
        # Use provided client
        for z, x, y in tiles:
            results.append(
                await fetch_tile(client, z, x, y, timeout_seconds=timeout_seconds, retries=retries)
            )
    return results


def _percentile(values: List[float], pct: float) -> float:
    """
    Calculate the nth percentile of a list of values.
    
    This is a simple implementation that sorts the values and
    finds the value at the specified percentile position.
    
    Args:
        values: List of numeric values
        pct: Percentile (0-100)
        
    Returns:
        The value at the specified percentile
    """
    if not values:
        return 0.0
    # Calculate index for the percentile
    # Round to nearest integer for proper indexing
    idx = int(round((pct / 100.0) * (len(values) - 1)))
    return sorted(values)[idx]


def benchmark(
    tile_count: int = 20,
    max_concurrency: int = 10,
    timeout_seconds: float = 10.0,
    retries: int = 2,
    global_timeout: Optional[float] = None,
) -> None:
    """
    Benchmark sequential vs concurrent tile fetching.
    
    This function demonstrates how to measure and compare performance
    between different approaches. It measures:
    - Total execution time
    - Individual request latencies
    - Throughput (tiles per second)
    - Percentile latencies (p50, p95, p99)
    
    Args:
        tile_count: Number of tiles to fetch for the benchmark
        max_concurrency: Maximum concurrent requests for concurrent mode
        timeout_seconds: Per-request timeout
        retries: Number of retries per request
        global_timeout: Overall timeout for the entire benchmark
    """
    tiles = generate_sample_tiles(count=tile_count)

    async def collect_latencies_seq() -> List[float]:
        """Collect latencies for sequential fetching."""
        lats: List[float] = []
        async with httpx.AsyncClient() as client:
            for z, x, y in tiles:
                start = time.perf_counter()  # High-precision timer
                await fetch_tile(client, z, x, y, timeout_seconds=timeout_seconds, retries=retries)
                lats.append(time.perf_counter() - start)
        return lats

    async def collect_latencies_con() -> List[float]:
        """Collect latencies for concurrent fetching."""
        lats: List[float] = []
        sem = asyncio.Semaphore(max_concurrency)
        async with httpx.AsyncClient() as client:
            async def one(z: int, x: int, y: int) -> None:
                """Fetch one tile and record its latency."""
                async with sem:
                    start = time.perf_counter()
                    await fetch_tile(client, z, x, y, timeout_seconds=timeout_seconds, retries=retries)
                    lats.append(time.perf_counter() - start)

            # Create and run all tasks concurrently
            await asyncio.gather(*(one(z, x, y) for z, x, y in tiles))
        return lats

    def run(coro):
        """Run a coroutine with optional global timeout."""
        if global_timeout is None:
            return asyncio.run(coro)
        try:
            # Apply global timeout to prevent hanging benchmarks
            return asyncio.run(asyncio.wait_for(coro, timeout=global_timeout))
        except asyncio.TimeoutError:
            print("Global timeout reached; partial results shown.")
            return []

    # Benchmark sequential approach
    print(f"Benchmarking {tile_count} tiles...")
    seq_start = time.perf_counter()
    seq_lats = run(collect_latencies_seq())
    seq_time = time.perf_counter() - seq_start

    # Benchmark concurrent approach
    con_start = time.perf_counter()
    con_lats = run(collect_latencies_con())
    con_time = time.perf_counter() - con_start

    def stats(name: str, lats: List[float], total: float) -> None:
        """Print performance statistics for a benchmark run."""
        if not lats:
            print(f"{name}: no results")
            return
        
        # Calculate throughput (tiles per second)
        thr = len(lats) / total if total > 0 else 0
        
        # Print comprehensive performance metrics
        print(
            f"{name}: {len(lats)} tiles in {total:.2f}s | throughput={thr:.2f} tps | "
            f"p50={_percentile(lats,50)*1000:.0f}ms p95={_percentile(lats,95)*1000:.0f}ms "
            f"p99={_percentile(lats,99)*1000:.0f}ms median={median(lats)*1000:.0f}ms"
        )

    # Display results
    print("\n" + "="*60)
    stats("Sequential", seq_lats, seq_time)
    stats("Concurrent", con_lats, con_time)
    
    # Show performance improvement
    if seq_time > 0 and con_time > 0:
        speedup = seq_time / con_time
        print(f"\nConcurrent fetching is {speedup:.1f}x faster than sequential")


if __name__ == "__main__":
    """
    Command-line interface for the tile fetcher benchmark.
    
    This demonstrates how to create a user-friendly CLI for async scripts.
    The script can be run with various parameters to test different scenarios.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Async tile fetch benchmark")
    parser.add_argument("--tile-count", type=int, default=20, 
                       help="Number of tiles to fetch")
    parser.add_argument("--max-concurrency", type=int, default=10,
                       help="Maximum concurrent requests")
    parser.add_argument("--timeout", type=float, default=10.0,
                       help="Per-request timeout seconds")
    parser.add_argument("--retries", type=int, default=2,
                       help="Number of retries per request")
    parser.add_argument("--base-url", type=str, default=None,
                       help="Override tile base URL")
    parser.add_argument("--global-timeout", type=float, default=None,
                       help="Overall timeout seconds")
    args = parser.parse_args()

    # Override base URL if specified
    if args.base_url:
        TILE_BASE_URL = args.base_url
        TILE_URL_TEMPLATE = f"{TILE_BASE_URL}/{{z}}/{{x}}/{{y}}.png"

    # Run the benchmark with provided arguments
    benchmark(
        tile_count=args.tile_count,
        max_concurrency=args.max_concurrency,
        timeout_seconds=args.timeout,
        retries=args.retries,
        global_timeout=args.global_timeout,
    )

