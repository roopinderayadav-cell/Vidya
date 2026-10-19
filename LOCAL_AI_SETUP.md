# Vidya - AI without any key (local AI + built-in meaning search)

Two separate switches. Use one, both, or neither.

| | Local AI (`AI_PROVIDER = "local"`) | Built-in meaning search (`MEANING_SEARCH = "builtin"`) |
|---|---|---|
| What it does | writes the answer from the SOP passages (like Gemini/Claude) | finds passages by meaning, not only exact words |
| Key needed | no | no |
| Data leaves the machine | no | no (the model is downloaded once, then runs inside Vidya) |
| Streamlit Community Cloud | **no** (needs 8 GB+ memory) | yes |
| Laptop / company server | yes | yes |

---------------------------------------------------------------------------------------------

## A. Built-in meaning search (easiest, works on Streamlit Cloud too)

1. `requirements.txt` already contains `fastembed==0.9.0` (official package of Qdrant,
   Apache-2.0 licence, from pypi.org).
2. Add one line to Secrets (each entry on ONE line):
   ```
   MEANING_SEARCH = "builtin"
   ```
3. Reboot the app. On the first question Vidya downloads the model **BAAI/bge-small-en-v1.5**
   (about 70 MB, MIT licence) from Qdrant's official page on huggingface.co
   (fallback: Qdrant's own Google Cloud storage). It is saved in `data/models/`.
4. Admin → Settings → **Build semantic index now** (once; again after a reboot on Cloud).

If the download is blocked (company firewall), Vidya just keeps using word search and shows a
note in Settings - nothing breaks. Ask IT to allow `huggingface.co` once, or copy the
`data/models/` folder from a machine that has it.

---------------------------------------------------------------------------------------------

## B. Local AI with Ollama (laptop or company server)

### 1. Install Ollama - ONLY from the official site
* Go to **https://ollama.com/download** (type the address yourself; do not use links from emails,
  ads or "mirror" sites). Windows: `OllamaSetup.exe`. Linux server: the install command shown on
  that page. Source code: https://github.com/ollama/ollama (MIT licence).
* Company laptop: ask IT before installing - they may want to install it for you.
* After install Ollama runs in the background and listens only on this computer
  (`http://localhost:11434`).

### 2. Download the models (from Ollama's official library, https://ollama.com/library)
Open Command Prompt / terminal:
```
ollama pull qwen2.5:7b          # answers   (~4.7 GB, Apache-2.0 licence, needs ~8 GB RAM)
ollama pull nomic-embed-text    # meaning search (~270 MB, Apache-2.0)
```
Smaller laptop (8 GB RAM or less):  `ollama pull llama3.2:3b`  (~2 GB, Llama 3.2 Community licence)
then set `LOCAL_MODEL = "llama3.2:3b"`.
Optional - read screenshots: `ollama pull gemma3:4b` (~3.3 GB, Gemma licence) and set
`LOCAL_VISION_MODEL = "gemma3:4b"`.

**Licence check:** before using a model at work, ask your InfoSec / legal team to approve the
licence (Apache-2.0 / MIT are the simplest). The licence is shown on each model's page at
ollama.com/library.

Quick test: `ollama run qwen2.5:7b "say ready"` - it should answer.

### 3. Tell Vidya to use it
Laptop (Command Prompt, once, then close and reopen):
```
setx AI_PROVIDER "local"
```
or in `.streamlit/secrets.toml` / Secrets:
```
AI_PROVIDER = "local"
LOCAL_MODEL = "qwen2.5:7b"
LOCAL_EMBED_MODEL = "nomic-embed-text"
```
Ollama on another company server:  `OLLAMA_URL = "http://<server-name>:11434"`
(by default Ollama only listens on its own machine; the server admin must open it to the
Vidya server only, never to the internet).

### 4. Start and check
```
python -m streamlit run app.py
```
Admin → Settings → **Test AI connection** → should show `Local AI (qwen2.5:7b) answered: ready
· meaning search ✅`. Then **Build semantic index now** (once - the old Gemini index can't be
mixed with the local one, Vidya clears it automatically when the model changes).

---------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------

## C. Backup AI - what Vidya does when a key stops working

Vidya tries the AIs in this order and moves to the next one at once when one fails:

| Step | Who answers | When it is skipped |
|---|---|---|
| 1 | Gemini main model (`GEMINI_MODEL`) | quota used up, key rejected, Google busy |
| 2 | Gemini second model (`GEMINI_FAST_MODEL`) - has its own free quota | same |
| 3 | Claude (only if `ANTHROPIC_API_KEY` is set) | same |
| 4 | Ollama on this computer (only if it is running) | Ollama closed |
| 5 | **Documents** - word search + meaning search + exact-line look-up | never - always works |

* A failed AI is skipped for a few minutes (quota 5 min, wrong key 30 min, Ollama closed 1 min),
  so agents never wait for the same error twice.
* Admin → Settings shows "Order it tries" and which AI answers right now. **Test AI connection**
  tests every AI in the order separately.
* Secrets (each on ONE line): `AI_BACKUP = "auto"` (default) · `"on"` · `"off"`
  - `auto`: Gemini first → Claude → Ollama. If `AI_PROVIDER = "local"`, NO cloud backup is used,
    so your SOP text never leaves the laptop.
  - `on`: also use Gemini/Claude as backup for local AI.
  - `off`: only `AI_PROVIDER`, then documents.

### Your laptop with Ollama (the easy way)
1. Ollama running (you see the llama icon near the clock) and `ollama pull qwen2.5:7b` done.
2. Start Vidya with **START_APP.bat** on the same laptop.
3. Nothing else to set: with `AI_PROVIDER = "auto"` Vidya finds Ollama by itself and uses it
   when there is no Gemini key or when Gemini is out of quota.
   Want Ollama FIRST (nothing to Google)? add `AI_PROVIDER = "local"`.

**Important:** the Streamlit Cloud app (the link your team opens) runs on Streamlit's computer, not
yours, so it **cannot** use the Ollama on your laptop. There it uses Gemini → Claude → documents.
Do not open your laptop's Ollama to the internet to make this work.

### Best free set-up for Streamlit Cloud (no quota worries)
```
GEMINI_API_KEY = "your-new-key"
MEANING_SEARCH = "builtin"
```
`MEANING_SEARCH = "builtin"` makes meaning search run inside Vidya (no key, no quota), so even when
Gemini's quota is used up the search still understands meaning, not only exact words.
After adding it: Reboot → Admin → Settings → **Build semantic index now** (once).

## What happens when something is wrong

| You see | Reason | Fix |
|---|---|---|
| "Local AI is not running - answering from documents." | Ollama app closed / PC restarted | start Ollama (Windows: Start menu → Ollama). Vidya checks again after 1 minute |
| "Local model 'x' is not installed - run: ollama pull x" | model name not downloaded | run the command shown |
| "Local AI took too long" | model too big for this PC | use `llama3.2:3b`, or a server with a GPU |
| Screenshots not read | no vision model | set `LOCAL_VISION_MODEL` |
| Voice button missing | voice needs Gemini | type the question |
| "Built-in meaning search could not start" | model download blocked | see section A, step 3 |

In every case Vidya still answers from the documents (word search), exactly like AI-off mode.

## Safety notes
* No new Python library is needed for local AI - Vidya talks to Ollama with Python's own
  built-in `urllib`, only to the address in `OLLAMA_URL`, never through a proxy for localhost.
* Only download Ollama from ollama.com and models with `ollama pull` (official library).
  Never run `.exe` files or "model packs" from other websites, shared drives or chats.
* Answer quality of a small local model is lower than Gemini/Claude. Test with your own SOPs
  (20-30 real questions) before using it on the floor. The document-answer safety rules
  (supervisor reply, conflict check, confidence %) work the same.

## What was tested
* Tested: the Vidya code that talks to Ollama (with a fake Ollama server on the same
  machine), Ollama closed / model missing messages, switching meaning-search models, built-in
  search with a stand-in model, a clean install of requirements.txt (Python 3.13), and all
  earlier answer checks (21 automated tests).
* NOT tested by us: a real model download and real answer quality - do this on your laptop or
  server (step 4 above).


---------------------------------------------------------------------------------------------

## D. Best models for a company server (v5.1)

| Job | Streamlit Cloud (small) | Company server (best) | Licence |
|---|---|---|---|
| Answers (local AI) | - (not possible on Cloud) | `LOCAL_MODEL = "qwen2.5:7b"` (GPU: `qwen2.5:14b`) | Apache-2.0 |
| Meaning search | `MEANING_SEARCH = "builtin"` (bge-small, English) | `MEANING_SEARCH = "local"` + `LOCAL_EMBED_MODEL = "bge-m3"` (`ollama pull bge-m3`, multilingual incl. Hinglish) | MIT |
| Reranker | default `Xenova/ms-marco-MiniLM-L-12-v2` (120 MB, English) | `RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8"` (570 MB, multilingual) | Apache-2.0 |

* Do **not** use `jinaai/jina-reranker-v2-base-multilingual` - its licence does not allow commercial use.
* After changing the meaning-search model: Admin -> Settings -> **Build semantic index now** (once).
* The reranker model is downloaded once from the official Hugging Face page (fallback: Qdrant's own
  storage) into `data/models/`. If the company firewall blocks it, ask IT to allow `huggingface.co`
  once, or copy `data/models/` from a machine that has it. Until then Vidya ranks with its rules.
* Not tested by us with the real reranker download (blocked in our test environment); the ranking logic
  was tested with a stand-in model. Compare 20-30 real questions with `RERANKER = "on"` and `"off"`.
