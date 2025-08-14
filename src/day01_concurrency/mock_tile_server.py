from fastapi import FastAPI, Response
import asyncio

app = FastAPI(title="Mock Tile Server")


@app.get("/{z}/{x}/{y}.png")
async def tile(z: int, x: int, y: int, delay_ms: int = 50, status: int = 200) -> Response:
    """Serve a tiny PNG header with optional delay and status.

    Query params:
    - delay_ms: artificial latency to simulate network jitter.
    - status: status code to force error responses (e.g., 500) for retry tests.
    """
    await asyncio.sleep(max(0, delay_ms) / 1000.0)
    if status != 200:
        return Response(status_code=status)
    # Minimal PNG signature; enough for a byte payload but not a valid image
    content = b"\x89PNG\r\n\x1a\n"
    return Response(content=content, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)

