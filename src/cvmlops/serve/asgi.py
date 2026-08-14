"""Single ASGI app for deployment: the API plus the control panel at /ui.

One process, one port (7860) — exactly what a Hugging Face Spaces Docker SDK
container expects. Run: uvicorn cvmlops.serve.asgi:app --host 0.0.0.0 --port 7860
  API:   /health, /predict, /predict/sample, /monitor/summary, /docs
  Panel: /ui

The panel is three static files with no build step, served by this same process,
so there is no second origin and no CORS.
"""

from __future__ import annotations

from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from cvmlops.serve.app import WEB_DIR, app

if WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def _root() -> RedirectResponse:
        return RedirectResponse("/ui/")
