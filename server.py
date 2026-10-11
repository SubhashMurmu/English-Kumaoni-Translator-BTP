"""Web server: serves the translator page and a JSON API from one process (same origin, no CORS).

  uvicorn server:app --port 8000          # real model
  KMT_DEMO=1 uvicorn server:app --port 8000   # UI development without the model (echo only)

Environment:
  KMT_MODEL   model directory           (default runs/base/final)
  KMT_PRESET  mbart | nllb               (default mbart)
  KMT_SCRIPT  devanagari | roman         (default devanagari: Roman model output is converted;
                                          Devanagari output passes through unchanged)
  KMT_ALPHA   Hindi-leak rerank weight   (default 0 = off)
"""
import os
import threading

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

DEMO = os.environ.get("KMT_DEMO") == "1"
MODEL_DIR = os.environ.get("KMT_MODEL", "runs/base/final")
PRESET = os.environ.get("KMT_PRESET", "mbart")
SCRIPT = os.environ.get("KMT_SCRIPT", "devanagari")
ALPHA = float(os.environ.get("KMT_ALPHA", "0"))

translator = None
overrides = {}
if not DEMO:
    from kmt.infer import Translator, load_hindi_only
    from kmt.romanize import load_overrides
    translator = Translator(MODEL_DIR, PRESET, load_hindi_only("synth/hindi_only.txt"))  # fails loudly
    overrides = load_overrides("synth/overrides.tsv")

lock = threading.Lock()   # one generation at a time; the model is not thread-safe
app = FastAPI(title="English to Kumaoni")


class Req(BaseModel):
    text: str = Field(max_length=500)


@app.get("/api/health")
def health():
    return {"ok": True, "demo": DEMO, "script": SCRIPT}


@app.post("/api/translate")
def translate(r: Req):
    text = r.text.strip()
    if not text:
        return {"translation": "", "demo": DEMO}
    if DEMO:
        return {"translation": "(demo mode, no model loaded) " + text, "demo": True}
    with lock:
        out = translator.translate([text], num_beams=4, n_best=4, alpha=ALPHA)[0]
    if SCRIPT == "devanagari":
        from kmt.romanize import to_devanagari
        out = to_devanagari(out, overrides)
    return {"translation": out, "demo": False}


@app.get("/")
def index():
    return FileResponse("web/index.html")


app.mount("/", StaticFiles(directory="web"), name="web")
