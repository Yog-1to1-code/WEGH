"""
WEGH Visualization Dashboard (FastAPI)
Serves the real-time CPU design dashboard at port 7861.
Proxies /api/* to the Go environment server at port 7860.
"""

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="WEGH Dashboard", version="1.0.0")

ENV_SERVER = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
STATIC_DIR = Path(__file__).parent / "static"

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the dashboard HTML."""
    with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/{path:path}")
async def proxy_get(path: str, request: Request):
    """Proxy GET requests to the Go environment server."""
    url = f"{ENV_SERVER}/{path}"
    params = dict(request.query_params)
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(url, params=params)
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)


@app.post("/api/{path:path}")
async def proxy_post(path: str, request: Request):
    """Proxy POST requests to the Go environment server."""
    url = f"{ENV_SERVER}/{path}"
    body = await request.body()
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(url, content=body,
                                   headers={"Content-Type": "application/json"})
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)


@app.get("/health")
async def health():
    return {"status": "ok", "dashboard": "wegh"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DASHBOARD_PORT", "7861"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
