"""
config.py - All Vidya settings in one place.
Change values here (or in Streamlit Secrets / environment variables).
"""
import os
from pathlib import Path


# What happened when Vidya read Streamlit Secrets - shown to admins so a format mistake is obvious.
# state: "ok" | "missing" (no Secrets saved) | "error" (Secrets saved but could not be read)
SECRETS_STATUS = {"state": "ok", "detail": ""}
SECRET_SOURCE = {}            # setting name -> "environment" / "secrets" / "default" (never the value)


def _secrets_problem(err):
    """Short, safe description of a Secrets problem (no file paths, no secret values)."""
    import re
    msg = str(err)
    if "No secrets found" in msg:
        return "missing", "No Secrets are saved for this app."
    m = re.search(r"\(line (\d+) column (\d+)", msg)
    reason = msg.split(":")[-1].split("(")[0].strip() if ":" in msg else "format problem"
    where = f" (line {m.group(1)})" if m else ""
    return "error", f"{reason}{where}"


def _secret(name, default=""):
    """Read a setting from an environment variable, or from Streamlit Secrets."""
    value = os.getenv(name)
    if value:
        SECRET_SOURCE[name] = "environment"
        return value
    try:
        import streamlit as st
        if name in st.secrets:
            SECRET_SOURCE[name] = "secrets"
            return str(st.secrets[name])
    except Exception as e:                  # Secrets could not be read at all
        state, detail = _secrets_problem(e)
        SECRETS_STATUS.update(state=state, detail=detail)
    SECRET_SOURCE[name] = "default"
    return default


# ---- App identity ---------------------------------------------------------
APP_NAME = "Vidya"
TAGLINE = "Your AI Learning & Knowledge Assistant"
WELCOME_TEXT = "Get instant answers from your learning resources — SOPs, FAQs, guides, documents and more."
COMPANY_NAME = "Your Company"

# ---- The two standard replies (edit the wording here) -----------------------
FALLBACK_TEXT = "I do not have answer to this question, please consult with your team supervisor."
CONFLICT_TEXT = ("I found different answers to this question in more than one document, so I cannot "
                 "confirm which one is correct. " + FALLBACK_TEXT)

# ---- Lines of business (LOB) ----------------------------------------------
# Each LOB is a tab above the chat; its sub-LOBs are the chips under the tab.
# Questions search the selected LOB's documents (plus GLOBAL documents) first,
# ranking the selected sub-LOB's documents highest. If nothing is found there,
# Vidya searches the other LOBs and says clearly where the answer came from.
# Sub-LOBs can be changed by admins in Settings (saved in the database).
# Adding a NEW top-level LOB needs a change here (REGIONS and DEFAULT_LOBS).
REGIONS = ["NORAM", "EUROPE", "APAC"]
GLOBAL_REGION = "GLOBAL"            # GLOBAL documents are visible in every LOB
REGION_LABELS = {"NORAM": "🌎 NORAM", "EUROPE": "🌍 EUROPE", "APAC": "🌏 APAC"}
DEFAULT_LOBS = {
    "NORAM": ["OBT Cancellation", "Schedule Change", "Ticketing", "Refund", "Profile Management", "Others"],
    "EUROPE": ["Ticketing", "Refund", "UETTR", "AGM", "BRIM ERM", "BRIM", "Matching", "Conferma", "SAP",
               "OTSMP", "DQS"],
    "APAC": ["AU TED", "AU Queues", "AU Ticketing", "SG Ticketing", "SG Refund", "SG Fares and Commission",
             "SG Support Service", "HK Ticketing", "HK Refund", "General Documents"],
}
SUB_LOB_BOOST = 2.20          # documents of the selected sub-LOB rank first...
OTHER_SUB_FACTOR = 0.80       # ...other sub-LOBs of the same LOB are still searched, slightly lower
GENERAL_DOC_FACTOR = 0.75     # documents with no sub-LOB (and GLOBAL ones) when a sub-LOB is chosen
CROSS_LOB_FALLBACK = True     # nothing found in this LOB -> look in the other LOBs (clearly labelled)

# ---- Topics (home-screen buttons + dashboard grouping) --------------------
# name: words that put a question into that topic
TOPICS = {
    "Ticketing": "ticket issue issuance ttp tst pnr booking book void e-ticket eticket itinerary "
                 "name correction spelling passenger",
    "Refunds": "refund refunds refundable money back cancellation fee refnd",
    "Exchanges": "exchange reissue revalidation date change change fee atc rebook",
    "OBT": "obt online booking tool self booking self-booking concur cytric",
    "Invoicing": "invoice invoicing billing credit note supplier gst vat",
    "Fare Rules": "fare rule rules category cat 16 31 33 penalty fare basis",
    "GDS (Amadeus/Sabre)": "amadeus sabre galileo gds command cryptic entry twd fxp tqt",
    "Profiles": "profile profiles traveller profile frequent flyer ffp passport details",
    "Travel Policy": "policy approval business class economy hotel preferred out of policy",
    "Quality / Errors": "error errors adm quality debit memo mistake issue failed invalid",
}
TOPIC_LABELS = {"GDS (Amadeus/Sabre)": "GDS", "Quality / Errors": "Quality"}   # short button text
TOPIC_ICONS = {"Ticketing": "🎫", "Refunds": "💸", "Exchanges": "🔄", "OBT": "🧳", "Invoicing": "🧾",
               "Fare Rules": "📜", "GDS (Amadeus/Sabre)": "🖥️", "Profiles": "👤",
               "Travel Policy": "📋", "Quality / Errors": "⚠️"}

# ---- Document categories --------------------------------------------------
DOC_TYPES = ["SOP", "FAQ", "Quick Guide", "Process Knowledge", "Policy", "Other"]
# Which document wins when two documents disagree: the NEWER effective date first, then this
# authority order (higher number = stronger). It also nudges ranking a little (x1.00 - x1.06).
DOC_AUTHORITY = {"Policy": 5, "SOP": 4, "Process Knowledge": 3, "FAQ": 2, "Quick Guide": 2, "Other": 1}
ALLOWED_EXTENSIONS = ["pdf", "docx", "pptx", "xlsx", "txt", "md", "csv", "png", "jpg", "jpeg"]

# ---- Admin access ---------------------------------------------------------
# Set ADMIN_PASSWORD in Streamlit Secrets. "admin123" is only a demo fallback and
# the app shows a warning to admins while it is in use.
ADMIN_PASSWORD = _secret("ADMIN_PASSWORD", "admin123")
DEFAULT_ADMIN_PASSWORD_IN_USE = ADMIN_PASSWORD == "admin123"
# SME (subject matter expert) sign-in for the SME Desk. Empty = SMEs use the admin password.
SME_PASSWORD = _secret("SME_PASSWORD", "")
# Answers below this confidence % go to the SME Desk to be confirmed (with 👎, no-answer, conflicts)
SME_REVIEW_BELOW_PCT = 70

# ---- Card numbers (PCI) -------------------------------------------------------
# Card numbers are hidden in questions, chats, comments, SME answers, documents and screenshots
# BEFORE anything is saved, shown or sent to an AI. Screenshots need Tesseract OCR
# (Streamlit Cloud: packages.txt; Windows laptop: see README). CARD_CHECK = True means screenshots
# are refused when Tesseract is missing - keep it True.
CARD_CHECK = True
MASK_KEEP_LAST4 = True                                    # ************1234
# Any run of this many digits or more (up to 19, spaces / hyphens allowed) is hidden, whatever its
# first digits. 13 also hides 13-digit airline TICKET numbers - set 14 to keep ticket numbers readable.
MASK_MIN_DIGITS = 13
TESSERACT_CMD = _secret("TESSERACT_CMD", "")              # Windows e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe


def password_warning():
    """None when the admin password is set properly, else the exact reason and how to fix it."""
    if not DEFAULT_ADMIN_PASSWORD_IN_USE:
        return None
    fix = ("Fix: in Streamlit → ⋮ → Settings → Secrets, put each entry on ONE line, e.g. "
           "ADMIN_PASSWORD = \"your-strong-password\" and GEMINI_API_KEY = \"your-key\" - "
           "then Save and ⋮ → Reboot app.")
    if SECRETS_STATUS["state"] == "error":
        return (f"Streamlit Secrets could not be read - check the format: {SECRETS_STATUS['detail']}. "
                "Vidya is using the demo password admin123 until this is fixed. "
                "A value split over two lines is the usual cause. " + fix)
    if SECRET_SOURCE.get("ADMIN_PASSWORD") in ("secrets", "environment"):
        return "ADMIN_PASSWORD in Secrets is still admin123. Change it to your own password, Save, then ⋮ → Reboot app."
    if SECRETS_STATUS["state"] == "missing":
        return "No Secrets are saved for this app, so the demo password admin123 is in use. " + fix
    return ("ADMIN_PASSWORD is not in Streamlit Secrets, so the demo password admin123 is in use. "
            "If you just added it, ⋮ → Reboot app so Vidya reads it. " + fix)

# ---- Demo mode ------------------------------------------------------------
# Load sample_docs/ automatically when the knowledge base is empty (free hosting
# wipes data on restart). File names start with the region: NORAM_..., GLOBAL_...
AUTO_LOAD_SAMPLES = os.getenv("AUTO_LOAD_SAMPLES", "true").lower() == "true"

# ---- Storage --------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "knowledge.db"

# ---- Search tuning --------------------------------------------------------
CHUNK_SIZE = 700          # characters per knowledge piece
CHUNK_OVERLAP = 150
CANDIDATES = 8            # passages compared for each answer (from all documents)
PER_DOC = 2               # max passages from one document -> answers mix documents
LEARNED_BOOST = 1.25      # verified answers (written by the process team)
AUTO_LEARNED_BOOST = 1.10 # answers Vidya learned by itself from 👍 votes

# Confidence without AI (word-match score of the best passage)
CONF_HIGH = 0.22
CONF_MEDIUM = 0.11        # below this Vidya gives FALLBACK_TEXT instead of guessing
# Confidence with Gemini semantic search (cosine similarity of the best passage)
SEM_HIGH = 0.72
SEM_MEDIUM = 0.60

MAX_ANSWER_WORDS = 60     # "short" answers stay under this many words

# Confidence % shown with every answer. The level (High / Medium) is decided by the
# rules above; the % places the answer inside that level's band using how strongly
# the best passage matches, how much of the question it covers and how many
# documents support it.
CONF_BANDS = {"high": (80, 98), "medium": (60, 79), "low": (20, 59)}

# ---- Self-learning --------------------------------------------------------
AUTO_LEARN = True         # save answers that staff mark 👍 as "auto-learned"
# Vidya learns from 👍 only when BOTH are reached for the same answer (similar questions count together):
LEARN_AGENT_VOTES = 10    # 👍 from this many DIFFERENT agents
LEARN_SME_VOTES = 2       # ...and from this many DIFFERENT SMEs (an admin counts as an SME)
# Until then 👍 does not move answers up. SMEs confirm answers in SME Desk → "👍 Answers to confirm".
LEARN_MIN_POSITIVE_PCT = 80  # ...and at least this % of the people who rated it gave 👍 (each person's last vote
                             # counts). Below 80% nothing is learned either way: no boost, no penalty.
LEARN_APPROVED_BOOST = 1.5  # an answer that reached both counts ranks this much higher for similar questions
AUTO_LEARN_MIN_VOTES = LEARN_AGENT_VOTES   # old name, kept so older code still reads a number
FEEDBACK_SIMILARITY = 0.55  # how similar a new question must be to reuse past 👍/👎

# ---- AI providers ----------------------------------------------------------
# Vidya works with NO key (free, offline). Add ONE key to turn on AI:
#   Gemini (free tier available): GEMINI_API_KEY  -> also gives semantic search
#   Claude (paid):                ANTHROPIC_API_KEY
# Streamlit Cloud: App -> Settings -> Secrets. Laptop: setx GEMINI_API_KEY "..."
GEMINI_API_KEY = _secret("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY", "")
AI_PROVIDER = _secret("AI_PROVIDER", "auto")   # auto | gemini | claude | local | off
# Backup AI when the first one fails (quota used up, key rejected, Ollama closed):
#   auto = cloud first -> other cloud key, then Ollama if it is running; local first -> no cloud
#   on   = use every AI that is set up (also Gemini/Claude as backup for local)
#   off  = only AI_PROVIDER
# Whatever happens, Vidya still answers from the documents when no AI works.
AI_BACKUP = _secret("AI_BACKUP", "auto")

# ---- Local AI (no key, nothing leaves your machine / company network) ---------
# Needs Ollama (free, official installer ONLY from https://ollama.com) running on the
# same laptop/server as Vidya. Then: ollama pull qwen2.5:7b  and  ollama pull nomic-embed-text
# Secrets / environment:  AI_PROVIDER = "local"
OLLAMA_URL = _secret("OLLAMA_URL", "http://localhost:11434")
LOCAL_MODEL = _secret("LOCAL_MODEL", "qwen2.5:7b")            # answers (Apache-2.0 licence)
LOCAL_FAST_MODEL = _secret("LOCAL_FAST_MODEL", "")             # empty = same as LOCAL_MODEL
LOCAL_VISION_MODEL = _secret("LOCAL_VISION_MODEL", "")         # e.g. gemma3:4b ; empty = no screenshots
LOCAL_EMBED_MODEL = _secret("LOCAL_EMBED_MODEL", "nomic-embed-text")   # meaning search
LOCAL_CONTEXT = 8192          # how much text the local model reads at once (Ollama default is small)
LOCAL_TIMEOUT = 180           # seconds - a laptop CPU can be slow on the first question

# ---- Meaning search (semantic search) ------------------------------------------
#   auto    = local AI embeddings when AI_PROVIDER is local, else Gemini when a Gemini key is set
#   builtin = small model INSIDE Vidya (fastembed, ~70 MB, downloaded once from the official
#             Qdrant page on huggingface.co). No key, no quota, works on Streamlit Cloud too.
#   gemini | local | off
MEANING_SEARCH = _secret("MEANING_SEARCH", "auto")
BUILTIN_EMBED_MODEL = _secret("BUILTIN_EMBED_MODEL", "BAAI/bge-small-en-v1.5")   # MIT licence

# ---- Reranker (v5.1): a model that reads the question + each candidate passage together ------
# auto = on when fastembed is installed (model downloaded once into data/models, loaded in the
# background - questions never wait); off = rules-only ranking. See rerank.py.
RERANKER = _secret("RERANKER", "auto")
RERANK_MODEL = _secret("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-12-v2")   # 120 MB, Apache-2.0
#   company server, multilingual (Hinglish): RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8" (570 MB)
RERANK_TOP = 12               # how many of the best passages the reranker reads
RERANK_WEIGHT = 0.5           # 0 = rules only, 1 = reranker decides alone; 0.5 = both count

GEMINI_MODEL = _secret("GEMINI_MODEL", "gemini-3.8-flash")            # answers, screenshots
GEMINI_FAST_MODEL = _secret("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")  # rephrasing
GEMINI_EMBED_MODEL = _secret("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768

CLAUDE_MODEL = _secret("CLAUDE_MODEL", "claude-sonnet-5-5")
CLAUDE_FAST_MODEL = _secret("CLAUDE_FAST_MODEL", "claude-haiku-4-5-20251001")

# ---- Images inside documents ----------------------------------------------
READ_DOC_IMAGES = True    # read screenshots/tables/scans inside uploaded files (AI mode)
MAX_IMAGES_PER_DOC = 25
# When the AI is out of quota, unreachable or says "not found", Vidya still answers from the
# documents if the strict word-match rule finds a strong match (the same rule used with AI off).
DOCUMENT_ANSWER_WHEN_AI_UNSURE = True
DOCUMENT_OVERRIDE_LEVELS = ("high", "medium")
AI_PAUSE_SECONDS = 300        # after a quota error, skip the AI for 5 minutes (no slow retries)
FAQ_MATCH_SIM = 0.65         # question worded like a document's "Q: ..." line (character similarity)
FAQ_MATCH_BOOST = 1.5        # ... ranks that FAQ answer first
DEFINITION_BOOST = 1.25      # 'what is X': sections headed 'X - ...' or 'What is X?' rank first

# A question about several things ("unused ticket, fare changed and invoice wrong") is split, each
# part is searched and answered on its own, and parts with no answer are named (never guessed).
MULTI_PART = True

# Ask a short question first when the answer depends on something the agent did not say
ASK_CLARIFY = True
CLARIFY_GROUPS = {"GDS": ["Amadeus", "Sabre", "Galileo", "Worldspan", "Apollo"]}
CLARIFY_ALIASES = {"Amadeus": ["1A", "ama", "amadus", "amedeus", "amadues", "amedus", "amdeus", "amadeous"],
                   "Sabre": ["1S", "saber", "sabr"], "Galileo": ["1G", "Travelport"],
                   "Worldspan": ["1P"], "Apollo": ["1V"]}
CLARIFY_CLOSE = 0.70          # other variants count when they score at least 70% of the best one
CLARIFY_MIN_COVER = 0.5       # ...and match at least half of the question's key words
# GDS questions: when the agent does not say which GDS, Vidya ALWAYS asks first - even when a learned
# answer, 👍 votes or a document covering both would otherwise answer straight away.
GDS_ALWAYS_ASK = True
GDS_OPTIONS = ["Amadeus", "Sabre"]            # the buttons Vidya shows
# Words that make a question a GDS question (whole words, any case)
GDS_WORDS = ["gds", "command", "commands", "entry", "entries", "cryptic", "format", "pnr", "pnrs", "tst",
             "tsts", "tqt", "pq", "pqs", "queue", "queues", "emd", "emds", "void", "voiding", "reissue",
             "re-issue", "reissuing", "revalidate", "revalidation", "exchange", "exchanges", "ssr", "osi",
             "fop", "itinerary", "segment", "segments", "pricing", "reprice", "repricing", "fare quote",
             "issue ticket", "issue a ticket", "issue the ticket", "ticket issue", "ticket issuance",
             "ticketing error", "error message", "refund mask", "remark", "remarks", "pcc", "oid",
             "office id", "atc", "fxp", "fxb", "fxx", "fxq", "ttp", "wpnc", "wfr", "*pq"]
# Commands that already tell which GDS it is (the agent typed a GDS-specific entry)
GDS_COMMANDS = {"Amadeus": ["TST", "TSTs", "TQT", "TTP", "FXP", "FXB", "FXX", "FXQ", "FXR", "FXA", "TTK", "TRDC", "TWD",
                            "RTSTR", "RTTN", "TRF", "TRFU", "TRFP", "ETK", "FQD", "FQN"],
                "Sabre": ["WPNC", "WPNCB", "WFR", "WETR", "PQ", "*PQ", "*IA", "*T", "W¥", "WP", "QXP", "TAW",
                          "WFRF", "WETR*"]}
# Showing document pictures with answers (works without the AI key)
SHOW_ANSWER_IMAGES = True
MAX_ANSWER_IMAGES = 2         # pictures shown under one answer
IMAGE_SOURCE_PASSAGES = 1     # pictures only from the passage the answer came from
MAX_STORED_IMAGES_PER_DOC = 60
MIN_IMAGE_BYTES = 3000        # smaller files are icons / logos / lines
MIN_IMAGE_SIDE = 80           # pixels
MAX_IMAGE_WIDTH = 1400        # bigger pictures are shrunk to save space
MIN_IMAGE_PX = 120
