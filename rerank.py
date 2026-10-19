"""
rerank.py - a second, closer look at the best passages (a "reranker" / cross-encoder model).

Word and meaning search find ~20 candidate passages fast. The reranker then reads the QUESTION and
EACH passage TOGETHER and scores how well that passage answers it - much closer to how a person
judges, and the biggest single accuracy gain for document chatbots. It works with AI on or off.

  RERANKER = "auto"   (default) on when the fastembed library is installed; the model is downloaded
                      ONCE from the official Qdrant / Hugging Face pages into data/models/
  RERANKER = "off"    rules-only ranking (as before v5.1)
  RERANK_MODEL        default Xenova/ms-marco-MiniLM-L-12-v2  (120 MB, English, Apache-2.0)
                      company server: BAAI/bge-reranker-v2-m3-int8 (570 MB, multilingual incl. Hinglish,
                      Apache-2.0). Do NOT use jina-reranker-v2 (non-commercial licence).

The model is loaded in the BACKGROUND when the app starts, so no question ever waits for it. Until it
is ready - or if it cannot load (download blocked, too little memory) - Vidya ranks with its rules,
exactly as before. Nothing here sends data anywhere: the model runs inside Vidya.
"""
import importlib.util
import math
import os
import threading
import time

import config

_state = {"model": None, "status": "off", "detail": "", "started": False, "seconds": None}
_lock = threading.Lock()


def mode():
    return (getattr(config, "RERANKER", "auto") or "auto").lower().strip()


def installed():
    try:
        return importlib.util.find_spec("fastembed") is not None
    except Exception:                                   # noqa: BLE001
        return False


def _load():
    t0 = time.time()
    try:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        cache = config.DATA_DIR / "models"
        cache.mkdir(parents=True, exist_ok=True)
        model = TextCrossEncoder(model_name=config.RERANK_MODEL, cache_dir=str(cache))
        list(model.rerank("test", ["test passage"]))       # make sure it really runs
        _state.update(model=model, status="ready", detail=config.RERANK_MODEL,
                      seconds=round(time.time() - t0, 1))
    except Exception as e:                              # noqa: BLE001
        _state.update(model=None, status="failed",
                      detail=f"{type(e).__name__}: {str(e)[:160]} - ranking with rules meanwhile")


def start():
    """Start loading the model in the background (once per server). Never blocks."""
    if mode() == "off":
        _state.update(status="off", detail="RERANKER = off")
        return
    if not installed():
        _state.update(status="not installed", detail="fastembed library missing (requirements.txt)")
        return
    with _lock:
        if _state["started"]:
            return
        _state.update(started=True, status="loading", detail=config.RERANK_MODEL)
    threading.Thread(target=_load, name="vidya-reranker", daemon=True).start()


def ready():
    return _state["model"] is not None and mode() != "off"


def status():
    """(status, detail) for the Settings page."""
    return _state["status"], _state["detail"]


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def scores(question, texts):
    """0..1 relevance for each passage, or None when the reranker is not ready / fails."""
    model = _state["model"]
    if model is None or not texts:
        return None
    try:
        raw = list(model.rerank(question, [t[:2000] for t in texts]))
        return [_sigmoid(float(s)) for s in raw]
    except Exception as e:                              # noqa: BLE001
        _state.update(status="failed", detail=f"{type(e).__name__}: {str(e)[:160]}")
        _state["model"] = None
        return None


def apply(question, hits, top=None, weight=None, cover=None):
    """Re-order the best passages by the reranker. Each score is multiplied by a factor between
    (1 - weight) and (1 + weight): the rules (LOB, SME answers, FAQ match, 👍/👎) still count, the
    reranker decides between close candidates.
    cover(hit) -> 0..1 share of the question's key words in the passage. Only passages close to the
    best word match are re-ordered; the others get the lowest factor, so the reranker can never lift a
    passage that does not contain what was asked. Returns (hits, used)."""
    top = top or config.RERANK_TOP
    weight = config.RERANK_WEIGHT if weight is None else weight
    head = hits[:top]
    if len(head) < 2 or not ready():
        return hits, False
    covers = [cover(h) for h in head] if cover else [1.0] * len(head)
    best_cover = max(covers)
    pick = [i for i, c in enumerate(covers) if c >= best_cover - 0.25]
    if len(pick) < 2:
        return hits, False
    texts = [f"{head[i].get('title', '')} - {head[i].get('location', '')}\n{head[i].get('question', '') or ''}\n"
             f"{head[i]['text']}" for i in pick]
    sc = scores(question, texts)
    if not sc:
        return hits, False
    for i, h in enumerate(head):
        h["score"] = h["score"] * (1 - weight)                 # outside the close group: lowest factor
    for i, p in zip(pick, sc):
        head[i]["rerank"] = round(p, 3)
        head[i]["score"] = head[i]["score"] / (1 - weight) * (1 - weight + 2 * weight * p)
    return sorted(head, key=lambda h: h["score"], reverse=True) + hits[top:], True
