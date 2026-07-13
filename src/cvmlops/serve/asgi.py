"""Single ASGI app for deployment: FastAPI (API) with Gradio mounted at /ui.

One process, one port (7860) — exactly what a Hugging Face Spaces Docker SDK
container expects. Run: uvicorn cvmlops.serve.asgi:app --host 0.0.0.0 --port 7860
  API:  /health, /predict, /monitor/summary, /docs
  UI:   /ui
"""

from __future__ import annotations

import gradio as gr

from cvmlops.serve.app import app
from cvmlops.ui.gradio_app import build

app = gr.mount_gradio_app(app, build(), path="/ui")
