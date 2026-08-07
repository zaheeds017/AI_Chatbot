"""
api/index.py - FastAPI backend for Omni AI (Vercel deployment).

Reuses the pure-Python chatbot engine (chatbot_engine.py) and the streaming
AI connectors (ai_providers.py) that power the original Streamlit app, but
serves them as a stateless HTTP + SSE API so the app runs natively on Vercel.

Endpoints:
  GET  /            -> the dark chat UI (static/index.html)
  GET  /api/health  -> status + provider list
  POST /api/chat    -> SSE stream of the assistant's reply
"""

import json
import os
import sys

# Make the project root importable (Vercel imports this file from api/).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ai_providers as ap
import chatbot_engine as engine

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Omni AI", version="1.0.0")

INDEX_HTML = os.path.join(_ROOT, "static", "index.html")

PROVIDERS = {k: v for k, v in ap.PROVIDERS.items() if k != "gemini_oauth"}


# ---- Brain (built once, reused across requests) -----------------------------
def _get_brain():
    kb = engine.load_kb()
    retriever = engine.Retriever(engine.flatten_faqs(kb))
    return kb, retriever


def _strip_greeting(messages):
    msgs = [dict(m) for m in messages]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


# ---- Request model ----------------------------------------------------------
class ChatRequest(BaseModel):
    messages: list[dict] = Field(..., description="Full conversation history")
    provider: str = "offline"
    model: str | None = None
    api_key: str = ""
    openai_base_url: str = ""
    max_tokens: int = 1024


# ---- Routes -----------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/api/health")
def health():
    kb, _ = _get_brain()
    n_faqs = len(engine.flatten_faqs(kb))
    return {
        "status": "ok",
        "app": "Omni AI",
        "faqs": n_faqs,
        "providers": [k for k in PROVIDERS if k != "offline"],
    }


def _sse(payload: dict):
    return "data: %s\n\n" % json.dumps(payload, ensure_ascii=False)


def _stream_reply(req: ChatRequest):
    kb, retriever = _get_brain()
    history = _strip_greeting(req.messages)
    text = history[-1]["content"] if history else ""

    if not text:
        yield _sse({"type": "error", "message": "Empty message."})
        return

    if req.provider != "offline" and req.provider in PROVIDERS:
        key = ap._get_key(req.provider, req.api_key)
        if key:
            try:
                system = engine.build_system_prompt(kb)
                gen = ap.stream_response(
                    req.provider, req.model,
                    history,
                    api_key=key,
                    system=system,
                    max_tokens=req.max_tokens,
                    openai_base_url=req.openai_base_url or None,
                )
                full = []
                for chunk in gen:
                    full.append(chunk)
                    yield _sse({"type": "delta", "text": chunk})
                yield _sse({"type": "done", "text": "".join(full)})
                return
            except Exception as ex:  # surface provider errors in the UI
                yield _sse({"type": "error", "message": str(ex)})
                return
        # no key -> fall through to offline mode

    reply = engine.get_response(text, history, kb, retriever)
    yield _sse({"type": "done", "text": reply})


@app.post("/api/chat")
def chat(req: ChatRequest):
    return StreamingResponse(
        _stream_reply(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
