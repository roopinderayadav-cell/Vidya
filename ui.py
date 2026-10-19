"""
ui.py - Vidya look and feel: colours, header bar, logo, running announcement line, cards,
and the live card-number guard on every text box.
"""
import base64
import html
from pathlib import Path

import streamlit as st

import config

NAVY, BLUE, PURPLE, TEAL, ORANGE = "#0f1b3d", "#1F5FBF", "#6A3FB5", "#0F8A7E", "#E8743B"
REGION_COLORS = {"NORAM": BLUE, "EUROPE": PURPLE, "APAC": TEAL, "GLOBAL": "#5b6475"}
ASSETS = Path(__file__).parent / "assets"
ICON_FILE = ASSETS / "vidya_icon.png"            # round Vidya robot: browser tab + chat avatar
ICON_WEB, HERO_WEB = ASSETS / "vidya_icon.webp", ASSETS / "vidya_hero.webp"   # header circle, welcome banner


def _b64(path):
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return ""


ICON_B64 = _b64(ICON_WEB)
HERO_B64 = _b64(HERO_WEB)
RESULT_BADGE = {"answered": ("Answered", "#e7f6ee", "#16794c"),
                "not_found": ("Not found", "#fdecec", "#b42318"),
                "conflict": ("Conflict", "#fff4e5", "#b54708")}

CSS = f"""
<style>
  /* The header holds the top menu bar - it must stay visible */
  header[data-testid="stHeader"] {{background: #ffffff; border-bottom: 1px solid #e3e8f2;}}
  .block-container {{padding-top: 5.6rem; padding-bottom: 6rem; max-width: 1400px;}}
  .om-conf {{background:#e7f6ee; color:#16794c; border-radius:999px; padding:1px 10px; font-size:.78rem;
             font-weight:700; margin-left:6px;}}
  [data-testid="stSegmentedControl"] button p {{font-weight:600;}}
  [data-testid="stSidebar"] {{background: #f6f8fc; border-right: 1px solid #e3e8f2;}}
  [data-testid="stSidebar"] .stRadio label p {{font-size: 0.95rem;}}
  .om-top {{display:flex; align-items:center; gap:18px; padding:10px 22px; border-radius:16px;
           background: linear-gradient(100deg, {NAVY} 0%, #1d3a8a 55%, #3b5bd6 100%); color:#fff;
           margin-bottom: 16px;}}
  .om-logo {{display:flex; align-items:center; gap:10px; font-size:1.6rem; font-weight:800;}}
  .om-logo .ring {{width:54px; height:54px; border-radius:50%; background:#fff; color:{NAVY};
                   display:flex; align-items:center; justify-content:center; font-size:1.2rem; overflow:hidden;
                   box-shadow:0 0 0 3px rgba(255,255,255,.25);}}
  .om-logo .ring img {{width:52px; height:52px;}}
  .om-checks {{background:#f2f6ff; border:1px solid #c9d8f5; border-left:6px solid {BLUE}; border-radius:10px;
               padding:10px 14px; margin:10px 0; font-size:.9rem;}}
  .om-checks b.t {{display:block; margin-bottom:4px; color:{NAVY};}}
  .om-checks ol {{margin:0; padding-left:20px;}}
  .om-ticker {{background:#fff1f0; border:1px solid #f4b4b0; border-left:6px solid #d92d20; border-radius:10px;
              overflow:hidden; white-space:nowrap; margin:0 0 14px; display:flex; align-items:center;}}
  .om-ticker .lbl {{background:#d92d20; color:#fff; font-weight:800; padding:8px 14px; flex:none; z-index:1;}}
  .om-ticker .run {{display:inline-block; padding:8px 0 8px 100%; color:#b42318; font-weight:700;
                   animation: om-run linear infinite;}}
  .om-ticker:hover .run {{animation-play-state: paused;}}
  .om-ticker .sep {{color:#f04438; margin:0 22px;}}
  @keyframes om-run {{ from {{transform: translateX(0);}} to {{transform: translateX(-100%);}} }}
  .om-hero-img {{position:absolute; right:10px; bottom:0; height:100%; max-height:150px; width:auto;}}
  .om-tag {{opacity:.85; border-left:1px solid rgba(255,255,255,.35); padding-left:14px;}}
  .om-sp {{flex:1}}
  .om-pill {{background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
             padding:6px 14px; border-radius:999px; font-weight:600; font-size:.9rem;}}
  .om-user {{display:flex; align-items:center; gap:10px; font-size:.85rem; line-height:1.15;}}
  .om-av {{width:34px; height:34px; border-radius:50%; background:#fff; color:{BLUE};
           display:flex; align-items:center; justify-content:center; font-weight:700;}}
  .om-motto {{font-weight:700; letter-spacing:.02em; opacity:.95}}
  .om-hero {{border-radius:16px; padding:22px 300px 22px 28px; min-height:150px; color:#fff; position:relative;
            overflow:hidden; display:flex; flex-direction:column; justify-content:center;
            background: linear-gradient(115deg, #0e1a3c 0%, #23367a 40%, #c9643f 82%, #f6b26b 100%);}}
  .om-hero h1 {{font-size:1.5rem !important; line-height:1.2; margin:0 0 8px; color:#fff !important; padding:0 !important;}}
  .om-hero h1 span {{color:#6cc4ff !important;}}
  .om-hero p {{margin:0; opacity:.92; font-size:.95rem; color:#fff !important;}}
  @media (max-width: 900px) {{ .om-hero {{padding-right:28px;}} .om-hero-img {{display:none;}} }}
  .om-card {{background:#fff; border:1px solid #e3e8f2; border-radius:14px; padding:16px 18px; margin-bottom:12px;}}
  .om-card h4 {{margin:0 0 10px; font-size:1rem;}}
  .om-badge {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:700;}}
  .om-region {{display:inline-block; padding:3px 12px; border-radius:999px; color:#fff; font-weight:700; font-size:.78rem;}}
  .om-kpi {{background:#fff; border:1px solid #e3e8f2; border-radius:14px; padding:14px 16px;}}
  .om-kpi .v {{font-size:1.7rem; font-weight:800; color:{NAVY};}}
  .om-kpi .l {{color:#5b6475; font-size:.85rem;}}
  .om-kpi .i {{font-size:1.1rem; margin-right:6px}}
  .om-answer-head {{display:flex; align-items:center; gap:8px; font-weight:700; color:{BLUE}; margin-bottom:4px;}}
  .om-important {{background:#fff7e8; border:1px solid #f6d58e; border-radius:10px; padding:10px 14px; margin:10px 0;}}
  .om-conflict {{background:#fff4e5; border:1px solid #f5c58a; border-radius:10px; padding:10px 14px; margin:8px 0;}}
  .om-fallback {{background:#fdecec; border:1px solid #f4b4b0; border-radius:10px; padding:10px 14px;}}
  .om-src {{border-left:4px solid #ccd5e6; background:#f7f9fc; border-radius:6px; padding:8px 12px; margin:6px 0; font-size:.88rem;}}
  .om-meta {{color:#5b6475; font-size:.82rem;}}
  .om-help {{background:#eef3fd; border-radius:12px; padding:12px 14px; font-size:.86rem; color:#33415c;}}
  div[data-testid="stButton"] button[kind="secondary"] {{border-radius:10px;}}
  .om-clarify {{background:#eef3fd; border:1px solid #bcd0f5; border-radius:10px; padding:10px 14px; margin:4px 0 8px;
                font-weight:600; color:#1d3a8a;}}
  [data-testid="stChatMessage"] code {{background:#0f2238; color:#7CFC9A; border-radius:5px; padding:1px 6px;
                font-family: Consolas, "DejaVu Sans Mono", monospace; font-size:.88em;}}
  .om-term {{background:#0f2238; border-radius:10px; margin:10px 0; overflow:hidden; max-width:640px;}}
  .om-term .bar {{background:#1e3a5f; color:#e2e8f0; font-size:.78rem; padding:5px 12px; font-weight:600;}}
  .om-term .body {{padding:8px 14px 10px;}}
  .om-term .ln {{color:#7CFC9A !important; font-family: Consolas, "DejaVu Sans Mono", monospace !important;
                 font-size:.9rem; line-height:1.6; white-space:pre-wrap; word-break:break-all;}}
  .om-term .ln span {{color:#94a3b8 !important;}}
  .om-section {{font-weight:800; color:{NAVY}; font-size:1.05rem; margin:6px 0 8px;}}
</style>
"""




def setup():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text):
    return html.escape("" if text is None else str(text))


def initials(name):
    parts = [p for p in str(name).replace("(", " ").split() if p[:1].isalpha()]
    return ("".join(p[0] for p in parts[:2]) or "U").upper()


def topbar(user, role, region):
    st.markdown(
        f"<div class='om-top'><div class='om-logo'><div class='ring'>{_icon()}</div>{config.APP_NAME}</div>"
        f"<div class='om-tag'>{esc(config.TAGLINE)}</div><div class='om-sp'></div>"
        f"<div class='om-pill'>🌐 {esc(region)}</div>"
        f"<div class='om-user'><div class='om-av'>{esc(initials(user))}</div>"
        f"<div><b>{esc(user)}</b><br>{esc(role)}</div></div></div>", unsafe_allow_html=True)


def _icon():
    return f"<img src='data:image/webp;base64,{ICON_B64}' alt='Vidya'>" if ICON_B64 else "V"


def ticker(items):
    """Announcements as a red running line. items: [{'title', 'body'}]. Hover = pause."""
    if not items:
        return
    text = "".join(f"<span>📢 <b>{esc(a['title'])}</b>{(' - ' + esc(a['body'])) if a.get('body') else ''}</span>"
                   f"<span class='sep'>●</span>" for a in items)
    chars = sum(len(a["title"]) + len(a.get("body") or "") for a in items)
    secs = max(18, min(240, chars // 6))            # about 6 characters a second - readable
    st.markdown(f"<div class='om-ticker'><div class='lbl'>ANNOUNCEMENT</div><div style='overflow:hidden;flex:1'>"
                f"<div class='run' style='animation-duration:{secs}s'>{text}</div></div></div>",
                unsafe_allow_html=True)


# Masks a card number INSIDE the text box while it is typed or pasted - before it is sent, so neither
# the agent's screen nor the SME's ever shows it. Same rule as privacy.must_mask (13-19 digits;
# MASK_MIN_DIGITS = 13 also hides ticket numbers). The server masks again on save (privacy.py) - two locks.
_GUARD_JS = r"""
<script>
(function () {
  // Runs in the MAIN page (not this small frame): the frame is replaced when the page changes,
  // and a listener living in it would stop working.
  function guard() {
    if (window.__vidyaCardGuard) return; window.__vidyaCardGuard = true;
    var ticket = /^\d{3}[ -]?\d{10}([ -\/]\d{1,2})?$/, MIN = __MIN__;
    function mask(t) {
      t = t.replace(/\*{6,}\d{5,}/g, function (m) { var n = m.replace(/\D/g, ''), s = m.length - n.length;
        return '*'.repeat(s + n.length - 4) + n.slice(-4); });
      return t.replace(/(^|[^\d])(\d(?:[ -]?\d){11,18})(?!\d)/g, function (all, pre, num) {
        var n = num.replace(/\D/g, '');
        if (n.length < MIN || n.length > 19 || (MIN >= 14 && ticket.test(num))) return all;
        return pre + '*'.repeat(n.length - 4) + n.slice(-4); });
    }
    function fix(el) {
      if (!el || !(el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && (el.type === 'text' || el.type === 'search')))) return;
      var v = el.value, m = mask(v);
      if (m === v) return;
      var proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, m);
      el.dispatchEvent(new Event('input', { bubbles: true }));
      try { el.setSelectionRange(m.length, m.length); } catch (e) {}
    }
    document.addEventListener('input', function (e) { fix(e.target); }, true);
    document.addEventListener('paste', function (e) { setTimeout(function () { fix(e.target); }, 0); }, true);
    window.__vidyaMask = mask;
  }
  var p = window.parent, d = p.document;
  if (p.__vidyaCardGuard) return;
  var s = d.createElement('script');
  s.textContent = '(' + guard.toString() + ')();';
  d.head.appendChild(s);
})();
</script>
"""


def card_guard():
    import streamlit.components.v1 as components
    import privacy
    components.html(_GUARD_JS.replace("__MIN__", str(privacy.min_digits())), height=0)


def hero(first_name):
    """Welcome banner: navy-to-orange, Vidya robot reading on the right."""
    img = (f"<img class='om-hero-img' src='data:image/webp;base64,{HERO_B64}' alt=''>" if HERO_B64 else "")
    st.markdown(
        f"<div class='om-hero'>{img}<h1>Hi {esc(first_name)}, how can <span>{config.APP_NAME}</span> "
        f"help you today?</h1><p>{esc(config.WELCOME_TEXT)}</p></div>", unsafe_allow_html=True)


def region_pill(region):
    return (f"<span class='om-region' style='background:{REGION_COLORS.get(region, '#5b6475')}'>"
            f"{esc(region)}</span>")


def result_badge(result):
    label, bg, fg = RESULT_BADGE.get(result or "answered", RESULT_BADGE["answered"])
    return f"<span class='om-badge' style='background:{bg};color:{fg}'>{label}</span>"


def kpi(col, icon, value, label, sub=""):
    col.markdown(f"<div class='om-kpi'><div class='v'><span class='i'>{icon}</span>{esc(value)}</div>"
                 f"<div class='l'>{esc(label)}</div><div class='om-meta'>{esc(sub)}</div></div>",
                 unsafe_allow_html=True)


def terminal(commands, system="GDS"):
    """GDS-style box listing the commands used in an answer, in order."""
    lines = "".join(f"<div class='ln'><span>&gt;</span> {esc(c)}</div>" for c in commands)
    return (f"<div class='om-term'><div class='bar'>🖥 {esc(system)} entries in this answer</div>"
            f"<div class='body'>{lines}</div></div>")


def card(title, body_html):
    st.markdown(f"<div class='om-card'><h4>{title}</h4>{body_html}</div>", unsafe_allow_html=True)


def checks(items):
    """The three checks shown with every fare-penalty answer."""
    rows = "".join(f"<li><b>{esc(t)}:</b> {esc(d)}</li>" for t, d in items)
    st.markdown(f"<div class='om-checks'><b class='t'>✅ Before you process - check these 3 things</b>"
                f"<ol>{rows}</ol></div>", unsafe_allow_html=True)


def avatar():
    """Vidya's face for her chat messages (falls back to a plane if the logo file is missing)."""
    return str(ICON_FILE) if ICON_FILE.exists() else "📘"
