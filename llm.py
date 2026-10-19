"""
llm.py - One connection point for the AI models Vidya can use.

  Gemini (Google)    - free tier available; also provides EMBEDDINGS (semantic search)
  Claude (Anthropic) - paid
  Local (Ollama)     - free, NO key, runs on your own laptop/server; nothing leaves the network
  Built-in           - small meaning-search model inside Vidya (fastembed), no key
  none               - Vidya still works offline (word matching + spelling fix)

Local AI uses only Python's own built-in web module (urllib) - no extra library needed.

Every function here fails safely: if the AI is unreachable, out of quota or the
key is wrong, it raises AIError and Vidya falls back to its offline answer.
"""
import base64
import importlib.util
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

import config


class AIError(Exception):
    pass


# ===========================================================================
# WHICH PROVIDER?
# ===========================================================================

_ORDER = {"gemini": ["gemini", "claude", "local"], "claude": ["claude", "gemini", "local"],
          "local": ["local", "gemini", "claude"], "auto": ["gemini", "claude", "local"]}
_probe = {"at": 0.0, "url": "", "ok": False}


def local_alive():
    """Is Ollama answering at OLLAMA_URL? Quick check (1.5 s), remembered for 60 seconds.
    Only talks to OLLAMA_URL."""
    url = (config.OLLAMA_URL or "").strip()
    if not url.startswith(("http://", "https://")):
        return False
    if _probe["url"] == url and time.time() - _probe["at"] < 60:
        return _probe["ok"]
    ok = False
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        handlers = [urllib.request.ProxyHandler({})] if host in ("localhost", "127.0.0.1", "::1") else []
        with urllib.request.build_opener(*handlers).open(url.rstrip("/") + "/api/tags", timeout=1.5) as r:
            ok = r.status == 200
    except Exception:                                    # noqa: BLE001
        ok = False
    _probe.update(at=time.time(), url=url, ok=ok)
    return ok


def providers():
    """Every AI Vidya may use, in the order it tries them.
    The first is AI_PROVIDER; the others are BACKUPS used when it fails (quota used up, key
    rejected, Ollama closed). AI_BACKUP: auto | on | off
      auto - cloud first: backups are the other cloud key and Ollama (if it is running).
             local first: NO cloud backup, so SOP text never leaves your network.
      on   - every AI that is set up, also cloud backup for local.
      off  - only AI_PROVIDER.
    When every AI fails, Vidya answers from the documents (word + meaning search)."""
    p = (config.AI_PROVIDER or "auto").lower().strip()
    if p == "off":
        return []
    p = p if p in _ORDER else "auto"
    backup = (getattr(config, "AI_BACKUP", "auto") or "auto").lower().strip()
    out = []
    for x in _ORDER[p]:
        primary = (x == p) or (p == "auto" and not out)
        if not primary and (backup == "off" or (p == "local" and backup != "on" and x != "local")):
            continue
        if x == "gemini" and config.GEMINI_API_KEY:
            out.append(x)
        elif x == "claude" and config.ANTHROPIC_API_KEY:
            out.append(x)
        elif x == "local" and (p == "local" or local_alive()):
            out.append(x)
    return out


def provider():
    """The AI that will answer the next question: the first one in the order that is not
    paused (paused = it failed a moment ago)."""
    chain = providers()
    for x in chain:
        if not paused(f"generate:{x}"):
            return x
    return chain[0] if chain else None


def ai_on():
    return provider() is not None


def _can_see(p):
    return p in ("gemini", "claude") or (p == "local" and bool(config.LOCAL_VISION_MODEL))


def vision_on():
    """Can the AI read pictures (screenshots, images inside documents)?"""
    return any(_can_see(p) for p in providers())


def builtin_available():
    """Is the built-in meaning-search library installed? (checked without loading it)"""
    return importlib.util.find_spec("fastembed") is not None


def embedder():
    """Which meaning-search model to use: 'gemini' | 'local' | 'builtin' | None.
    When AI_PROVIDER is local, auto NEVER sends documents to Google."""
    m = (getattr(config, "MEANING_SEARCH", "auto") or "auto").lower().strip()
    ai = (config.AI_PROVIDER or "auto").lower().strip()
    if m == "off":
        return None
    if m == "builtin":
        return "builtin" if builtin_available() else None
    if m == "local":
        return "local"
    if m == "gemini":
        return "gemini" if config.GEMINI_API_KEY else None
    # auto
    if ai == "off":
        return None
    if ai == "local":
        return "local"
    return "gemini" if config.GEMINI_API_KEY else None


def embeddings_on():
    return embedder() is not None


def embedder_id():
    """Name of the meaning-search model. Vectors from different models can't be mixed,
    so knowledge.py rebuilds the index when this changes."""
    e = embedder()
    if e == "gemini":
        return f"gemini:{config.GEMINI_EMBED_MODEL}:{config.EMBED_DIM}"
    if e == "local":
        return f"local:{config.LOCAL_EMBED_MODEL}"
    if e == "builtin":
        return f"builtin:{config.BUILTIN_EMBED_MODEL}"
    return ""


def embed_label():
    e = embedder()
    return {"gemini": f"On (Gemini {config.GEMINI_EMBED_MODEL})",
            "local": f"On (local AI {config.LOCAL_EMBED_MODEL})",
            "builtin": f"On (built-in {config.BUILTIN_EMBED_MODEL})"}.get(e, "Off")


_NAMES = {"gemini": lambda: f"Gemini ({config.GEMINI_MODEL})",
          "claude": lambda: f"Claude ({config.CLAUDE_MODEL})",
          "local": lambda: f"Local AI ({config.LOCAL_MODEL})"}
last_used = {"provider": None}       # which AI wrote the last answer (shown as a small note)


def name_of(p):
    return _NAMES[p]() if p in _NAMES else "Off"


def label():
    p = provider()
    if p is None:
        return "Off"
    chain = providers()
    text = name_of(p)
    if chain and p != chain[0]:
        text += " - backup, " + name_of(chain[0]).split(" (")[0] + " is paused"
    return text


def chain_label():
    """e.g. 'Gemini → Local AI → documents'"""
    return " → ".join([name_of(p).split(" (")[0] for p in providers()] + ["documents"])


# ===========================================================================
# CLIENTS
# ===========================================================================
_gemini = None
_claude = None


def _gemini_client():
    global _gemini
    if _gemini is None:
        try:
            from google import genai
        except ImportError:
            raise AIError("Gemini library missing - run: pip install -r requirements.txt")
        from google.genai import types
        # 60-second limit: a hanging request never blocks the agent - the backup / documents answer
        _gemini = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(timeout=60000))
    return _gemini


def _types():
    try:
        from google.genai import types
        return types
    except ImportError:
        raise AIError("Gemini library missing - run: pip install -r requirements.txt")


def _claude_client():
    global _claude
    if _claude is None:
        import anthropic
        _claude = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=60)
    return _claude


# ---- Local AI (Ollama) - plain HTTP to the Ollama app on this machine / server ----
LOCAL_DOWN_TEXT = "Local AI is not running - answering from documents."


def _local_url(path):
    base = (config.OLLAMA_URL or "http://localhost:11434").strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise AIError("OLLAMA_URL must start with http:// or https://")
    return base + path


def _local_post(path, payload):
    """POST JSON to Ollama and return the JSON reply. Talks ONLY to OLLAMA_URL."""
    url = _local_url(path)
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never via a proxy
    else:
        opener = urllib.request.build_opener()
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with opener.open(req, timeout=config.LOCAL_TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        model = payload.get("model", "")
        if e.code == 404 and "not found" in body.lower():
            raise AIError(f"Local model '{model}' is not installed - run: ollama pull {model}")
        raise AIError(f"Local AI error {e.code}: {body[:120]}")
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        reason = str(getattr(e, "reason", e)).lower()
        if "timed out" in reason:
            raise AIError("Local AI took too long - try a smaller model (e.g. llama3.2:3b).")
        raise AIError(LOCAL_DOWN_TEXT)


_THINK = re.compile(r"<think>.*?</think>", re.S)


def _local_generate(prompt, images, fast, json_mode, max_tokens):
    if images:
        if not config.LOCAL_VISION_MODEL:
            raise AIError("Reading pictures needs a local vision model (set LOCAL_VISION_MODEL)")
        model = config.LOCAL_VISION_MODEL
    else:
        model = (config.LOCAL_FAST_MODEL or config.LOCAL_MODEL) if fast else config.LOCAL_MODEL
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = [base64.b64encode(b).decode() for b in images]
    payload = {"model": model, "messages": [msg], "stream": False, "keep_alive": "30m",
               "options": {"temperature": 0.1, "num_predict": max_tokens,
                           "num_ctx": config.LOCAL_CONTEXT}}
    if json_mode:
        payload["format"] = "json"
    r = _local_post("/api/chat", payload)
    text = ((r.get("message") or {}).get("content") or "")
    return _THINK.sub("", text).strip()


def _local_embed(texts, query=False):
    model = config.LOCAL_EMBED_MODEL
    if "nomic" in model.lower():                       # nomic models expect these prefixes
        texts = [("search_query: " if query else "search_document: ") + t for t in texts]
    out = []
    for i in range(0, len(texts), 32):
        r = _local_post("/api/embed", {"model": model, "input": [t[:6000] for t in texts[i:i + 32]],
                                       "keep_alive": "30m"})
        vecs = r.get("embeddings") or []
        if len(vecs) != len(texts[i:i + 32]):
            raise AIError("Local AI returned no meaning-vectors - check LOCAL_EMBED_MODEL")
        out += vecs
    return out


# ---- Built-in meaning search (fastembed + ONNX, runs inside Vidya) ----------
_builtin_model = None


def _builtin_embed(texts, query=False):
    global _builtin_model
    if _builtin_model is None:
        if not builtin_available():
            raise AIError("Built-in meaning search is not installed - run: pip install -r requirements.txt")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        try:
            from fastembed import TextEmbedding
            cache = config.DATA_DIR / "models"
            cache.mkdir(parents=True, exist_ok=True)
            _builtin_model = TextEmbedding(model_name=config.BUILTIN_EMBED_MODEL, cache_dir=str(cache))
        except Exception as e:                              # noqa: BLE001
            _paused_until["embed"] = time.time() + 600      # don't retry the download on every question
            raise AIError(f"Built-in meaning search could not start ({str(e)[:100]}) - "
                          "word search is used meanwhile.")
    try:
        texts = [t[:2000] for t in texts]
        it = _builtin_model.query_embed(texts) if query else _builtin_model.passage_embed(texts)
        return [list(v) for v in it]
    except Exception as e:                                  # noqa: BLE001
        raise AIError(f"Built-in meaning search failed ({str(e)[:100]})")


def _png(img_bytes):
    """Resize any image to a sensible size and return PNG bytes."""
    from PIL import Image
    im = Image.open(io.BytesIO(img_bytes))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.thumbnail((1568, 1568))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


# After a quota / rate-limit error the AI is skipped for a few minutes, so staff get the
# document answer at once instead of waiting for the AI to fail again on every question.
_paused_until = {"generate": 0.0, "embed": 0.0}
QUOTA_TEXT = "The AI's free quota is used up for now - answering from documents."


def paused(kind="generate"):
    """kind 'generate' = EVERY AI in the order is paused; 'generate:gemini' = only that one."""
    now = time.time()
    if now < _paused_until.get(kind, 0.0):
        return True
    if kind == "generate":
        chain = providers()
        return bool(chain) and all(now < _paused_until.get(f"generate:{p}", 0.0) for p in chain)
    return False


def _check_pause(kind):
    if paused(kind):
        if kind == "embed" and embedder() == "builtin":
            raise AIError("Built-in meaning search is not ready - using word search for now.")
        raise AIError(LOCAL_DOWN_TEXT if provider() == "local" else QUOTA_TEXT)


KEY_TEXT = "The AI key was rejected - check the key in Secrets."


def _note_failure(kind, err):
    """Skip an AI that just failed for a while, so the next question goes straight to the backup
    (or the documents) instead of waiting for the same failure again."""
    msg = err.args[0] if err.args else ""
    if msg == QUOTA_TEXT:
        wait = getattr(config, "AI_PAUSE_SECONDS", 300)
    elif msg == KEY_TEXT:
        wait = 1800                                 # wrong key: check again in 30 minutes
    elif msg == LOCAL_DOWN_TEXT:
        wait = 60                                   # Ollama closed: check again in 1 minute
    elif msg.startswith(("AI unavailable", "Local AI took too long")):
        wait = 60                                   # server busy / timeout
    else:
        return
    _paused_until[kind] = time.time() + wait


def _retry(fn):
    """Try twice: free tiers sometimes answer 'busy / rate limit' for a moment."""
    last = None
    for attempt in range(2):
        try:
            return fn()
        except Exception as e:                          # noqa: BLE001
            last = e
            msg = str(e).lower()
            if attempt == 0 and any(k in msg for k in ("429", "503", "rate", "overloaded",
                                                       "unavailable", "resource_exhausted")):
                time.sleep(2)
                continue
            break
    raise AIError(_friendly(last))


def _friendly(e):
    msg = str(e)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "401" in low or "permission" in low or "invalid" in low and "key" in low:
        return KEY_TEXT
    if "429" in low or "quota" in low or "resource_exhausted" in low or "rate" in low:
        return QUOTA_TEXT
    return f"AI unavailable ({msg[:120]})"


# ===========================================================================
# GENERATE TEXT (optionally with images / audio)
# ===========================================================================

def generate(prompt, images=None, audio=None, fast=False, json_mode=False, max_tokens=1200):
    """Ask the first AI that works. If it fails (quota, key, closed, busy) the next one in
    providers() is tried at once; the failed one is skipped for a few minutes."""
    chain = providers()
    if not chain:
        raise AIError("AI is off")
    first = None
    for p in chain:
        if images and not _can_see(p):
            continue
        if audio and p != "gemini":
            continue
        if paused(f"generate:{p}"):
            first = first or AIError(LOCAL_DOWN_TEXT if p == "local" else QUOTA_TEXT)
            continue
        try:
            out = _generate(p, prompt, images, audio, fast, json_mode, max_tokens)
            last_used["provider"] = p
            return out
        except AIError as e:
            _note_failure(f"generate:{p}", e)
            first = first or e
    last_used["provider"] = None
    raise first or AIError("No AI here can read pictures - answering from documents.")


def _generate(p, prompt, images, audio, fast, json_mode, max_tokens):
    images = [_png(b) for b in (images or []) if b]

    if p == "local":
        if audio:
            raise AIError("Voice questions need a Gemini key")
        return _local_generate(prompt, images, fast, json_mode, max_tokens)

    if p == "gemini":
        types = _types()
        model = config.GEMINI_FAST_MODEL if fast else config.GEMINI_MODEL
        parts = [types.Part.from_bytes(data=b, mime_type="image/png") for b in images]
        if audio:
            parts.append(types.Part.from_bytes(data=audio[0], mime_type=audio[1]))
        parts.append(prompt)

        def call(thinking=True):
            cfg = dict(temperature=0.1, max_output_tokens=max_tokens + 1024)
            if json_mode:
                cfg["response_mime_type"] = "application/json"
            if thinking:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_level="low")
            r = _gemini_client().models.generate_content(
                model=model, contents=parts, config=types.GenerateContentConfig(**cfg))
            return (r.text or "").strip()

        def run():
            try:
                return call(True)
            except Exception as e:                      # model without thinking options
                if "thinking" in str(e).lower():
                    return call(False)
                raise
        try:
            return _retry(run)
        except AIError as e:
            # Each Gemini model has its OWN free quota: when the main model is used up or
            # retired ("not found"), try the fast model before moving to the next AI.
            other = config.GEMINI_FAST_MODEL
            if fast or not other or other == model or str(e) == KEY_TEXT:
                raise
            model = other
            return _retry(run)

    # Claude
    model = config.CLAUDE_FAST_MODEL if fast else config.CLAUDE_MODEL
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(b).decode()}} for b in images]
    content.append({"type": "text", "text": prompt})

    def run_claude():
        msg = _claude_client().messages.create(model=model, max_tokens=max_tokens,
                                               messages=[{"role": "user", "content": content}])
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    return _retry(run_claude)


# ===========================================================================
# EMBEDDINGS - turn text into meaning-vectors for semantic search (Gemini, local or built-in)
# ===========================================================================

def embed(texts, query=False):
    """Returns an (n, EMBED_DIM) float32 array of unit vectors, or raises AIError."""
    e = embedder()
    if e is None:
        raise AIError("Meaning search is off")
    _check_pause("embed")
    try:
        if e == "local":
            out = _local_embed(texts, query)
        elif e == "builtin":
            out = _builtin_embed(texts, query)
        else:
            return _embed(texts, query)
        arr = np.array(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return arr / norms
    except AIError as e:
        _note_failure("embed", e)
        raise


def _embed(texts, query=False):
    types = _types()
    out = []
    for i in range(0, len(texts), 50):
        batch = [t[:6000] for t in texts[i:i + 50]]

        def run(batch=batch):
            r = _gemini_client().models.embed_content(
                model=config.GEMINI_EMBED_MODEL, contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
                    output_dimensionality=config.EMBED_DIM))
            return [e.values for e in r.embeddings]
        out += _retry(run)
    arr = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms


# ===========================================================================
# VOICE (Gemini) and CONNECTION TEST
# ===========================================================================

def transcribe(audio_bytes, mime="audio/wav"):
    import prompts
    if "gemini" not in providers():
        raise AIError("Voice questions need a Gemini key")
    return generate(prompts.TRANSCRIBE_PROMPT, audio=(audio_bytes, mime), fast=True, max_tokens=300)


def test_connection():
    """Used by the Settings page: tests EVERY AI in the order and returns (ok, message).
    A working AI is un-paused at once."""
    chain = providers()
    if not chain and not embeddings_on():
        return False, ("No AI set - Vidya answers from the documents only (free, offline). "
                       + ("Ollama is not answering at " + config.OLLAMA_URL + "." if not local_alive() else ""))
    msg, ok_any = [], False
    for p in chain:
        try:
            reply = _generate(p, "Reply with the single word: ready", None, None, True, False, 20)
            _paused_until.pop(f"generate:{p}", None)
            msg.append(f"✅ {name_of(p)}: {reply[:30]}")
            ok_any = True
        except AIError as e:
            _note_failure(f"generate:{p}", e)
            msg.append(f"❌ {name_of(p)}: {e}")
    if embeddings_on():
        try:
            _paused_until.pop("embed", None)
            embed(["test"], query=True)
            msg.append(f"✅ meaning search ({embed_label()})")
            ok_any = ok_any or not chain
        except AIError as e:
            msg.append(f"❌ meaning search: {e}")
    msg.append("documents ✅ always on")
    return ok_any, " · ".join(msg)
