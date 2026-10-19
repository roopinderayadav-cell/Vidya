"""
admin_pages.py - Vidya admin screens: Dashboard & Analytics, Document Management,
SME Desk, Review Queue, Knowledge Gaps, Learned Answers, Announcements, Settings.
"""
import json
import re
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import config
import knowledge as kb
import llm
import privacy
import ui

PERIODS = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": None}


def route(name, region, user):
    pages = {"📊 Dashboard": dashboard, "🗂️ Document Management": documents,
             "🧠 Review Queue": review, "🕳️ Knowledge Gaps": gaps,
             "✅ Learned Answers": learned, "⚙️ Settings": settings}
    fn = pages.get(name)
    if fn:
        fn(region, user)


def _clean_q(q):
    return re.sub(r"\s*\[screenshot\]$", "", str(q)).strip()


# ---------------------------------------------------------------------------
# DASHBOARD & ANALYTICS
# ---------------------------------------------------------------------------
def dashboard(region, user):
    st.header("Dashboard & Analytics")
    c1, c2, c5, c3, c4 = st.columns([1.1, 1.1, 1.2, 1.3, 1.3])
    period = c1.selectbox("Period", list(PERIODS), index=1)
    reg = c2.selectbox("LOB", ["All"] + config.REGIONS)
    lob = c5.selectbox("Sub-LOB", ["All"] + (kb.get_lobs().get(reg, []) if reg != "All" else []), key=f"dsub_{reg}")
    topic = c3.selectbox("Topic", ["All"] + list(config.TOPICS) + ["Other"])
    df = kb.questions_df(reg, PERIODS[period], topic)
    if lob != "All" and not df.empty:
        df = df[df["sub_lob"].fillna("") == lob]
    since = None
    if PERIODS[period]:
        since = (pd.Timestamp.now() - pd.Timedelta(days=PERIODS[period])).strftime("%Y-%m-%d %H:%M")
    gap_list = kb.knowledge_gaps(None if reg == "All" else reg, since)
    c4.write("")
    c4.download_button("⬇ Export to Excel", kb.export_excel(df, gap_list), type="primary",
                       file_name=f"Vidya_questions_{datetime.now():%Y-%m-%d}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       width="stretch")
    if df.empty:
        st.info("No questions in this period yet.")
        return

    total = len(df)
    answered = int((df["result"] == "answered").sum())
    rated = int(df["feedback"].notna().sum())
    helpful = int((df["feedback"] == 1).sum())
    unhelpful = int((df["feedback"] == -1).sum())
    unanswered = int(df["result"].isin(["not_found", "conflict"]).sum())
    k = st.columns(6)
    ui.kpi(k[0], "💬", f"{total:,}", "Total questions")
    ui.kpi(k[1], "✅", f"{100 * answered / total:.0f}%", "Answered")
    ui.kpi(k[2], "👍", f"{100 * helpful / rated:.0f}%" if rated else "–", "Helpful", f"{rated} rated")
    ui.kpi(k[3], "👎", f"{100 * unhelpful / rated:.0f}%" if rated else "–", "Not helpful")
    ui.kpi(k[4], "🕳️", f"{unanswered:,}", "Unanswered / conflict",
           f"{int((df['result'] == 'conflict').sum())} conflicts")
    pcts = pd.to_numeric(df["confidence_pct"], errors="coerce").dropna() if "confidence_pct" in df else pd.Series(dtype=float)
    ui.kpi(k[5], "🎯", f"{pcts.mean():.0f}%" if len(pcts) else "–", "Avg confidence", "of answered questions")
    st.write("")

    colors = alt.Scale(domain=list(ui.REGION_COLORS), range=list(ui.REGION_COLORS.values()))
    a, b, c = st.columns([1, 1.2, 1.5])
    with a:
        st.markdown("<div class='om-section'>Questions by region</div>", unsafe_allow_html=True)
        by_reg = df.groupby("region").size().reset_index(name="questions")
        st.altair_chart(alt.Chart(by_reg).mark_arc(innerRadius=55).encode(
            theta="questions:Q", color=alt.Color("region:N", scale=colors, legend=alt.Legend(orient="bottom")),
            tooltip=["region", "questions"]).properties(height=240), width="stretch")
    with b:
        st.markdown("<div class='om-section'>Top topics</div>", unsafe_allow_html=True)
        by_topic = df.groupby("topic").size().reset_index(name="questions").sort_values("questions", ascending=False)
        st.altair_chart(alt.Chart(by_topic.head(8)).mark_bar(color=ui.BLUE, cornerRadiusEnd=4).encode(
            x=alt.X("questions:Q", title=None), y=alt.Y("topic:N", sort="-x", title=None),
            tooltip=["topic", "questions"]).properties(height=240), width="stretch")
    with c:
        st.markdown("<div class='om-section'>Question trend</div>", unsafe_allow_html=True)
        trend = df.groupby(["date", "region"]).size().reset_index(name="questions")
        st.altair_chart(alt.Chart(trend).mark_line(point=True).encode(
            x=alt.X("date:T", title=None), y=alt.Y("questions:Q", title=None),
            color=alt.Color("region:N", scale=colors, legend=alt.Legend(orient="bottom")),
            tooltip=["date", "region", "questions"]).properties(height=240), width="stretch")

    d, e, f = st.columns([1.4, 1.4, 1])
    with d:
        st.markdown("<div class='om-section'>Most asked questions</div>", unsafe_allow_html=True)
        q = df.assign(q=df["question"].map(_clean_q).str.lower().str.strip(" ?."))
        top = q.groupby("q").size().reset_index(name="times").sort_values("times", ascending=False).head(8)
        st.dataframe(top.rename(columns={"q": "Question", "times": "Times"}), hide_index=True,
                     width="stretch")
    with e:
        st.markdown("<div class='om-section'>Unanswered / needs review</div>", unsafe_allow_html=True)
        g = pd.DataFrame([{"Question": x["question"], "Type": x["type"], "Times": x["count"]} for x in gap_list[:8]])
        if g.empty:
            st.success("No open gaps 🎉")
        else:
            st.dataframe(g, hide_index=True, width="stretch")
    with f:
        st.markdown("<div class='om-section'>Feedback summary</div>", unsafe_allow_html=True)
        fb = pd.DataFrame({"feedback": ["Helpful", "Not helpful"], "votes": [helpful, unhelpful]})
        if rated:
            st.altair_chart(alt.Chart(fb).mark_arc(innerRadius=50).encode(
                theta="votes:Q", color=alt.Color("feedback:N", scale=alt.Scale(
                    domain=["Helpful", "Not helpful"], range=["#16a34a", "#dc2626"]),
                    legend=alt.Legend(orient="bottom")), tooltip=["feedback", "votes"]).properties(height=200),
                width="stretch")
        else:
            st.caption("No ratings yet.")

    with st.expander("All questions in this period"):
        show = df[["asked_at", "region", "sub_lob", "asked_by", "topic", "question", "result", "confidence",
                   "confidence_pct", "helpful"]]
        st.dataframe(show, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# DOCUMENT MANAGEMENT
# ---------------------------------------------------------------------------
def documents(region, user):
    st.header("Document Management")
    lobs = kb.get_lobs()
    with st.expander("⬆ Upload document", expanded=False):
        l1, l2 = st.columns(2)
        up_region = l1.selectbox("LOB", config.REGIONS + [config.GLOBAL_REGION], key="up_region",
                                 index=config.REGIONS.index(region) if region in config.REGIONS else 0,
                                 format_func=lambda r: config.REGION_LABELS.get(r, "🌐 GLOBAL (every LOB)"),
                                 help="GLOBAL = visible in every LOB")
        up_sub = l2.selectbox("Sub-LOB", ["General (whole LOB)"] + lobs.get(up_region, []), key=f"up_sub_{up_region}",
                              disabled=up_region == config.GLOBAL_REGION,
                              help="Answers for this sub-LOB rank this document first.")
        up_sub = "" if up_sub.startswith("General") else up_sub
        with st.form("upload", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            doc_type = c2.selectbox("Document type", config.DOC_TYPES)
            version = c3.text_input("Version", placeholder="e.g. 2.1")
            title = c1.text_input("Title (optional)")
            owner = c2.text_input("Owner (optional)")
            eff = c3.date_input("Effective from (optional)", value=None,
                                help="When two documents disagree, the one with the newer effective date "
                                     "is used (then Policy > SOP > FAQ). The other goes to the SME Desk.")
            files = st.file_uploader("Files", type=config.ALLOWED_EXTENSIONS, accept_multiple_files=True)
            if st.form_submit_button("Upload & index", type="primary") and files:
                for f in files:
                    try:
                        with st.spinner(f"Reading {f.name}…"):
                            _, n, info = kb.add_document(up_region, doc_type, title if len(files) == 1 else "",
                                                         f.name, f.getvalue(), user, version, owner,
                                                         sub_lob=up_sub,
                                                         effective_date=eff.strftime("%Y-%m-%d") if eff else "")
                        msg = f"✔ {f.name} → {up_region} · {up_sub or 'General'}: {n} knowledge pieces"
                        if info["images_read"]:
                            msg += f", {info['images_read']} image(s) read by AI"
                        if info.get("images_stored"):
                            msg += f", {info['images_stored']} picture(s) kept to show with answers"
                        if info["semantic"] == "on":
                            msg += ", semantic search ✔"
                        st.success(msg)
                        if info["images_skipped"]:
                            st.warning(f"{info['images_skipped']} image(s) in {f.name} were not read - they are "
                                       "shown with answers, but text inside them is searchable only with a "
                                       "Gemini key (add it and upload again).")
                        if info["gaps_rechecked"]:
                            st.info(f"🧠 {info['gaps_rechecked']} open knowledge gap(s) may now be answered "
                                    "by this document - see Knowledge Gaps.")
                    except Exception as e:
                        st.error(f"✖ {f.name}: {e}")
        st.caption("Tip: use the Vidya SOP / FAQ templates - clear headings, numbered steps, real tables "
                   "and 'Q:' / 'A:' FAQs give the most accurate answers.")

    c1, c0, c2, c3, c4 = st.columns([1, 1.2, 1, 1, 2])
    r = c1.selectbox("LOB", ["All"] + config.REGIONS + [config.GLOBAL_REGION])
    lf = c0.selectbox("Sub-LOB", ["All"] + (lobs.get(r, []) if r in lobs else []) + ["General (no sub-LOB)"],
                      key=f"lf_{r}")
    t = c2.selectbox("Type", ["All"] + config.DOC_TYPES)
    s = c3.selectbox("Status", ["All", "Active", "Archived"])
    q = c4.text_input("Search documents…")
    docs = kb.list_documents()
    docs = [d for d in docs if (r == "All" or d["region"] == r) and (t == "All" or d["doc_type"] == t)
            and (lf == "All" or (d.get("sub_lob") or "") == ("" if lf.startswith("General") else lf))
            and (s == "All" or (d.get("status") or "Active") == s)
            and (not q or q.lower() in (d["title"] + d["filename"]).lower())]
    untagged = sum(1 for d in kb.list_documents() if not d.get("sub_lob") and d["region"] != config.GLOBAL_REGION)
    if untagged:
        st.caption(f"ℹ️ {untagged} document(s) have no sub-LOB (General). They are used for the whole LOB; "
                   "tag them with ✏️ so the right sub-LOB's documents rank first.")
    h = st.columns([3.2, 1, 1.1, 0.8, 1.3, 1, 2.2])
    for col, label in zip(h, ["File name", "LOB · sub-LOB", "Type", "Version", "Updated on", "Status", "Actions"]):
        col.markdown(f"**{label}**")
    for d in docs:
        status = d.get("status") or "Active"
        c = st.columns([3.2, 1, 1.1, 0.8, 1.3, 1, 2.2])
        icon = {"pdf": "📕", "docx": "📘", "xlsx": "📗", "pptx": "📙"}.get(d["filename"].rsplit(".", 1)[-1], "📄")
        c[0].markdown(f"{icon} {d['title']}  \n<span class='om-meta'>{d['filename']} · {d['chunks']} piece{'s' if d['chunks'] != 1 else ''}</span>",
                      unsafe_allow_html=True)
        lob_txt = d.get("sub_lob") or "General"
        c[1].markdown(ui.region_pill(d["region"]) + f"<br><span class='om-meta'>{ui.esc(lob_txt)}</span>",
                      unsafe_allow_html=True)
        c[2].write(d["doc_type"])
        c[3].markdown(f"{ui.esc(d['version'] or '–')}" + (f"<br><span class='om-meta'>eff. {d['effective_date']}</span>"
                      if d.get("effective_date") else ""), unsafe_allow_html=True)
        c[4].write(d["uploaded_at"][:10])
        c[5].markdown(ui.result_badge("answered").replace("Answered", "Active") if status == "Active"
                      else "<span class='om-badge' style='background:#eef0f4;color:#5b6475'>Archived</span>",
                      unsafe_allow_html=True)
        a1, a2, a3, a4 = c[6].columns(4)
        with a4.popover("✏️", help="Change sub-LOB / effective date"):
            subs = ["General (whole LOB)"] + lobs.get(d["region"], [])
            cs = d.get("sub_lob") or ""
            ns = st.selectbox(f"Sub-LOB in {d['region']}", subs, index=subs.index(cs) if cs in subs else 0,
                              key=f"ts{d['id']}", disabled=d["region"] == config.GLOBAL_REGION)
            from datetime import date
            try:
                cur = date.fromisoformat(d["effective_date"]) if d.get("effective_date") else None
            except ValueError:
                cur = None
            ne = st.date_input("Effective from", value=cur, key=f"te{d['id']}")
            if st.button("Save", key=f"tsave{d['id']}", type="primary"):
                kb.update_document_tags(d["id"], "" if ns.startswith("General") else ns,
                                        effective_date=ne.strftime("%Y-%m-%d") if ne else "")
                st.rerun()
        a1.download_button("⬇", kb.read_file(d), file_name=d["filename"], key=f"d{d['id']}", help="Download")
        if a2.button("🗄" if status == "Active" else "♻", key=f"s{d['id']}",
                     help="Archive (stop using in answers)" if status == "Active" else "Make active again"):
            kb.set_document_status(d["id"], "Archived" if status == "Active" else "Active")
            st.rerun()
        if a3.button("🗑", key=f"x{d['id']}", help="Delete"):
            kb.delete_document(d["id"])
            st.rerun()
    if not docs:
        st.info("No documents match.")


# ---------------------------------------------------------------------------
# REVIEW QUEUE
# ---------------------------------------------------------------------------
def review(region, user):
    st.header("Review Queue")
    st.caption("Questions Vidya could not answer, answered with low confidence, found conflicting answers "
               "for, or staff marked 👎 (the same list SMEs see on the SME Desk, one row per time asked). "
               "Write the correct answer once - it is used for similar questions from then on.")
    scope = st.radio("Show", ["This region", "All regions"], horizontal=True)
    queue = kb.review_queue(region if scope == "This region" else None)
    if not queue:
        st.success("Nothing waiting. 🎉")
    limit = st.session_state.setdefault("rq_limit", 10)
    st.caption(f"Showing {min(limit, len(queue))} of {len(queue)} - newest first. Tip: Knowledge Gaps groups "
               "repeated questions so you can fix many at once.")
    for q in queue[:limit]:
        with st.container(border=True):
            result = q.get("result") or "not_found"
            st.markdown(f"{ui.region_pill(q['region'])} {ui.result_badge(result)} &nbsp; **{ui.esc(q['question'])}**",
                        unsafe_allow_html=True)
            st.caption(f"Asked by {q['asked_by']} · {q['asked_at']} · topic {q.get('topic') or '–'}"
                       + (" · 👎 rated not helpful" if q["feedback"] == -1 else ""))
            if q["comment"]:
                st.markdown(f"> Staff note: {q['comment']}")
            if q.get("image_path"):
                from pathlib import Path
                if Path(q["image_path"]).exists():
                    st.image(q["image_path"], caption="Screenshot the agent pasted", width=360)
            with st.expander("What Vidya answered"):
                st.markdown(q["answer"] or "")
            with st.form(f"teach{q['id']}"):
                new_q = st.text_input("Question (edit to make it general)", _clean_q(q["question"]))
                ans = st.text_area("Correct answer", height=120,
                                   placeholder="Write the steps / rule exactly as staff should follow them")
                c1, c2, c3 = st.columns(3)
                if c1.form_submit_button("✅ Save & teach", type="primary"):
                    if ans.strip():
                        kb.teach(q["region"], new_q, ans.strip(), user, q["id"])
                        st.rerun()
                    else:
                        st.warning("Please write an answer.")
                if c2.form_submit_button("📝 SOP update needed"):
                    kb.set_gap([q["id"]], status="SOP update needed")
                    st.rerun()
                if c3.form_submit_button("Dismiss"):
                    kb.dismiss(q["id"])
                    st.rerun()
    if len(queue) > limit and st.button("Show 10 more"):
        st.session_state.rq_limit = limit + 10
        st.rerun()


# ---------------------------------------------------------------------------
# SME DESK - an SME answers what Vidya could not answer well. Their answer is saved as a
# verified answer that Vidya searches FIRST next time. Uploaded SOPs are never changed.
# ---------------------------------------------------------------------------
REASON_ICON = {"No answer": "🕳️", "Documents disagree": "⚠️", "Not helpful 👎": "👎"}


def sme_desk(region, user):
    st.header("🧑‍🏫 SME Desk")
    st.caption("Questions Vidya could not answer, answered with less than "
               f"{config.SME_REVIEW_BELOW_PCT}% confidence, found conflicting SOPs for, or staff marked 👎. "
               "Write the correct answer once - Vidya uses it for this and similar questions from then on. "
               "Uploaded SOPs and documents are not changed.")
    n_wait, n_rev = len(kb.waiting_chats()), len(kb.chats_to_review(sme=user))
    t1, tc, t2, th = st.tabs(["⏳ Waiting for an SME", f"💬 Live chat ({n_wait} waiting"
                              + (f", {n_rev} to save" if n_rev else "") + ")",
                              "✅ Answers given by SMEs", "🗂️ Chat history"])
    with tc:
        _sme_live(region, user)
    with th:
        _sme_history(region)
    with t1:
        c1, c2 = st.columns([1, 2.5])
        lob = c1.selectbox("LOB", [region, "All"] + [r for r in config.REGIONS if r != region], key="sme_lob")
        queue = kb.sme_queue(None if lob == "All" else lob)
        reasons = sorted({x.split(" (")[0] for g in queue for x in g["reasons"]})
        pick = c2.multiselect("Why it is here", reasons, default=reasons, key="sme_reasons")
        shown = [g for g in queue if any(x.split(" (")[0] in pick for x in g["reasons"])]
        k = st.columns(4)
        ui.kpi(k[0], "⏳", len(queue), "Questions waiting")
        ui.kpi(k[1], "🔁", sum(g["count"] for g in queue), "Times asked")
        ui.kpi(k[2], "📉", len([g for g in queue if any(x.startswith("Low") for x in g["reasons"])]),
               f"Below {config.SME_REVIEW_BELOW_PCT}%")
        ui.kpi(k[3], "👎", len([g for g in queue if "Not helpful 👎" in g["reasons"]]), "Marked 👎")
        st.write("")
        if not shown:
            st.success("Nothing waiting. 🎉")
        limit = st.session_state.setdefault("sme_limit", 10)
        for n, g in enumerate(shown[:limit]):
            _sme_card(n, g, user)
        if len(shown) > limit and st.button("Show 10 more", key="sme_more"):
            st.session_state.sme_limit = limit + 10
            st.rerun()
    with t2:
        _sme_answers(region, user)


def _sme_card(n, g, user):
    q = g["latest"]
    with st.container(border=True):
        tags = " ".join(f"`{REASON_ICON.get(x, '📉')} {x}`" for x in g["reasons"])
        st.markdown(f"{ui.region_pill(g['region'])} &nbsp; **{ui.esc(g['question'])}**", unsafe_allow_html=True)
        st.markdown(tags)
        sub = f" · sub-LOB {q['sub_lob']}" if q.get("sub_lob") else ""
        st.caption(f"Asked {g['count']}× by {', '.join(g['users'][:4])}{sub} · last {g['last']}")
        for c in g["comments"][:3]:
            st.markdown(f"> Staff note: {ui.esc(c)}")
        if q.get("image_path"):
            from pathlib import Path
            if Path(q["image_path"]).exists():
                st.image(q["image_path"], caption="Screenshot the agent pasted", width=360)
        bot_answered = q.get("result") == "answered"
        with st.expander("What Vidya answered" + (f" ({q['confidence_pct']}% confident)"
                                                   if q.get("confidence_pct") is not None else "")):
            st.markdown(q["answer"] or "")
            try:
                srcs = json.loads(q.get("sources") or "[]")
            except ValueError:
                srcs = []
            for src in srcs[:3]:
                st.caption(f"📄 {src.get('title', '')} · {src.get('location', '')}")
        with st.form(f"sme{n}_{g['ids'][0]}"):
            new_q = st.text_input("Question (you can make it general, e.g. remove names)", _clean_q(g["question"]))
            ans = st.text_area("SME answer", height=140,
                               placeholder="Write the steps / rule exactly as agents should follow them. "
                                           "Numbered steps work best: 1. ... 2. ...")
            b = st.columns(4)
            save = b[0].form_submit_button("✅ Save SME answer", type="primary")
            ok = b[1].form_submit_button("👍 Vidya's answer is right", disabled=not bot_answered)
            sop = b[2].form_submit_button("📝 SOP update needed")
            drop = b[3].form_submit_button("🚫 Not a process question")
            try:
                if save:
                    kb.sme_answer(g["ids"], g["region"], new_q, ans, user)
                    st.toast("Saved - Vidya will use this answer next time.", icon="✅")
                    st.rerun()
                if ok:
                    kb.sme_confirm(g["ids"], user)
                    st.toast("Confirmed - kept as a verified answer.", icon="👍")
                    st.rerun()
                if sop:
                    kb.sme_flag_sop(g["ids"], user, ans.strip())
                    st.toast("Sent to Admin → Knowledge Gaps as 'SOP update needed'.", icon="📝")
                    st.rerun()
                if drop:
                    kb.sme_dismiss(g["ids"], user)
                    st.rerun()
            except ValueError as e:
                st.warning(str(e))


def screenshot_picker(key):
    """Paste or attach a screenshot for the live chat. Card numbers are covered BEFORE it is shown
    or saved; the agent sees exactly what the other person will see. Returns masked PNG or None."""
    ss = st.session_state
    n = ss.setdefault(f"{key}_n", 0)
    c1, c2 = st.columns([1, 2])
    raw = None
    try:
        from streamlit_paste_button import paste_image_button
        with c1:
            res = paste_image_button("📋 Paste screenshot", background_color=ui.BLUE,
                                     hover_background_color="#174a96", key=f"{key}_paste{n}")
        if res.image_data is not None:
            import io
            buf = io.BytesIO()
            res.image_data.save(buf, format="PNG")
            raw = buf.getvalue()
    except ImportError:
        pass
    up = c2.file_uploader("…or attach a screenshot", type=["png", "jpg", "jpeg"], key=f"{key}_up{n}",
                          label_visibility="collapsed")
    if up is not None:
        raw = up.getvalue()
    if raw:
        try:
            with st.spinner("Checking the screenshot for card numbers…"):
                ss[f"{key}_img"], ss[f"{key}_note"] = privacy.check_image(raw)
        except privacy.PrivacyError as e:
            ss[f"{key}_img"] = None
            st.error(str(e))
    img = ss.get(f"{key}_img")
    if img:
        a, b = st.columns([3, 1])
        a.image(img, caption="This is what will be sent (card numbers hidden)", width=320)
        a.caption(ss.get(f"{key}_note", ""))
        if b.button("✕ Remove", key=f"{key}_rm"):
            clear_screenshot(key)
            st.rerun()
    return img


def clear_screenshot(key):
    ss = st.session_state
    ss[f"{key}_img"] = None
    ss[f"{key}_n"] = ss.get(f"{key}_n", 0) + 1


def show_chat_message(m):
    """One chat bubble (text masked again on display, picture already masked when saved)."""
    if m["role"] == "system":
        st.caption(f"— {m['text']} ({m['at'][-5:]})")
        return
    with st.chat_message("user" if m["role"] == "agent" else "assistant",
                         avatar="🧑" if m["role"] == "agent" else "🧑‍🏫"):
        if m["text"] != "📷 Screenshot" or not m.get("image_path"):
            st.markdown(privacy.mask_md(m["text"]))
        if m.get("image_path"):
            from pathlib import Path
            if Path(m["image_path"]).exists():
                st.image(m["image_path"], width=480)
        st.caption(f"{m['sender']} · {m['at'][-5:]}")


def _sme_live(region, user):
    ss = st.session_state
    c1, c2 = st.columns([1.3, 2])
    avail = c1.toggle("🟢 I'm available for live chat", value=True, key="sme_available")
    lobs = c2.multiselect("Chats from LOB", config.REGIONS, default=[region], key="sme_chat_lobs")

    @st.fragment(run_every=4)
    def inbox():
        kb.set_presence(user, region, avail)            # tells agents an SME is online
        waiting = [c for c in kb.waiting_chats() if c["region"] in lobs]
        seen = ss.setdefault("sme_seen_chats", set())
        for c in waiting:
            if c["id"] not in seen:
                seen.add(c["id"])
                st.toast(f"New chat from {c['agent']}: {c['first'][:60]}", icon="💬")
        if not waiting:
            st.caption("No agent is waiting. New chats appear here by themselves (checked every few seconds).")
        for c in waiting:
            with st.container(border=True):
                a, b = st.columns([5, 1])
                a.markdown(f"{ui.region_pill(c['region'])} **{ui.esc(c['agent'])}** · {c['started_at'][-5:]}"
                           + (f" · {ui.esc(c['sub_lob'])}" if c["sub_lob"] else "")
                           + f"<br>{ui.esc(c['first'][:300])}", unsafe_allow_html=True)
                if b.button("Pick up", key=f"pick{c['id']}", type="primary"):
                    if kb.pick_up_chat(c["id"], user):
                        ss.sme_chat = c["id"]
                    else:
                        st.toast("Another SME already took this chat.", icon="ℹ️")
                    st.rerun(scope="app")
    inbox()

    mine = kb.sme_active_chats(user)
    if mine:
        ids = [c["id"] for c in mine]
        if ss.get("sme_chat") not in ids:
            ss.sme_chat = ids[0]
        if len(mine) > 1:
            ss.sme_chat = st.radio("My open chats", ids, horizontal=True, key="sme_chat_pick",
                                   index=ids.index(ss.sme_chat),
                                   format_func=lambda i: next(f"{c['agent']} (#{i})" for c in mine if c["id"] == i))
        cid = ss.sme_chat
        chat = kb.get_chat(cid)
        st.markdown(f"**Chat with {ui.esc(chat['agent'])}** · {chat['region']}"
                    + (f" › {ui.esc(chat['sub_lob'])}" if chat["sub_lob"] else ""), unsafe_allow_html=True)

        @st.fragment(run_every=3)
        def convo():
            with st.container(height=380, border=True):
                for m in kb.chat_messages(cid):
                    show_chat_message(m)
            if kb.get_chat(cid)["status"] == "closed":
                st.rerun(scope="app")
        convo()
        with st.expander("🔎 Look it up in Vidya first"):
            q = st.text_input("Search the SOPs", key=f"sme_lookup{cid}")
            if q.strip():
                hits = kb.search(chat["region"], [q])[:3]
                for h in hits:
                    st.markdown(f"**{ui.esc(h['title'])}** · {ui.esc(h['location'])}")
                    st.caption(h["text"][:500])
        shot = screenshot_picker(f"sme_shot{cid}")
        with st.form(f"sme_send{cid}", clear_on_submit=True):
            t = st.text_area("Your reply", height=90, max_chars=kb.CHAT_MAX_LEN,
                             placeholder="Write the steps clearly - this can become a verified answer.")
            b1, b2 = st.columns(2)
            if b1.form_submit_button("➤ Send", type="primary") and (t.strip() or shot):
                try:
                    kb.send_chat(cid, user, "sme", t, image=shot)
                    clear_screenshot(f"sme_shot{cid}")
                except privacy.PrivacyError as e:
                    st.error(str(e))
                st.rerun()
            if b2.form_submit_button("End chat & save answers"):
                kb.close_chat(cid, user)
                st.rerun()

    review = kb.chats_to_review(sme=user)
    if review:
        st.divider()
        st.subheader("💾 Save answers from ended chats")
        st.caption("Tick the question/answer pairs worth keeping and correct the wording. Saved pairs become "
                   "verified answers that Vidya uses first next time. The full chat is kept either way.")
    for c in review[:5]:
        pairs = kb.chat_pairs(c["id"])
        with st.container(border=True):
            st.markdown(f"Chat #{c['id']} with **{ui.esc(c['agent'])}** · {c['region']} · ended {c['closed_at']}")
            with st.expander("Full chat"):
                for m in kb.chat_messages(c["id"]):
                    show_chat_message(m)
            with st.form(f"save_chat{c['id']}"):
                chosen = []
                if not pairs:
                    st.caption("No question/answer pair found in this chat.")
                for k, p in enumerate(pairs):
                    keep = st.checkbox(f"Save pair {k + 1}", value=True, key=f"keep{c['id']}_{k}")
                    q = st.text_input("Question", p["question"][:500], key=f"pq{c['id']}_{k}")
                    a = st.text_area("Answer", p["answer"], height=100, key=f"pa{c['id']}_{k}")
                    if keep:
                        chosen.append({"question": q, "answer": a, "asked": p["asked"]})
                b1, b2 = st.columns(2)
                if b1.form_submit_button("💾 Save ticked answers", type="primary"):
                    n = kb.save_chat_pairs(c["id"], chosen, user)
                    st.toast(f"Saved {n} answer(s) - Vidya will use them next time.", icon="✅")
                    st.rerun()
                if b2.form_submit_button("Nothing to save"):
                    kb.save_chat_pairs(c["id"], [], user)
                    st.rerun()


def _sme_history(region):
    c1, c2, c3 = st.columns([1, 2, 1.2])
    lob = c1.selectbox("LOB", [region, "All"] + [r for r in config.REGIONS if r != region], key="hist_lob")
    find = c2.text_input("Search in chats", key="hist_find")
    c3.write("")
    c3.download_button("⬇ All chats to Excel", kb.chats_excel(None if lob == "All" else lob),
                       file_name=f"Vidya_SME_chats_{datetime.now():%Y-%m-%d}.xlsx", width="stretch")
    chats = kb.chat_history(None if lob == "All" else lob, find.strip(), limit=50)
    if not chats:
        st.caption("No chats yet.")
    for c in chats:
        title = (f"#{c['id']} · {c['agent']} ↔ {c['sme'] or '(no SME)'} · {c['region']} · {c['started_at']} · "
                 f"{c['status']}" + (f" · {c['saved_count']} answer(s) saved" if c.get("saved_count") else ""))
        with st.expander(title):
            for m in c["messages"]:
                show_chat_message(m)


def _sme_answers(region, user):
    lob = st.selectbox("LOB", [region, config.GLOBAL_REGION] + [r for r in config.REGIONS if r != region],
                       key="sme_ans_lob")
    find = st.text_input("Search answers", key="sme_find", placeholder="word in the question or answer")
    rows = kb.sme_answers(lob)
    if find.strip():
        rows = [r for r in rows if find.lower() in (r["question"] + " " + r["answer"]).lower()]
    st.caption(f"{len(rows)} verified answers in {lob}. Edit one when the process changes; switch it off "
               "to stop Vidya using it.")
    for r in rows[:50]:
        with st.container(border=True):
            state = "🟢 in use" if r.get("active", 1) else "⚪ switched off"
            st.markdown(f"{state} · **Q:** {ui.esc(r['question'])}")
            st.markdown(r["answer"])
            st.caption(f"by {r['added_by']} · {r['added_at']}")
            with st.expander("✏ Edit"):
                with st.form(f"sme_edit{r['id']}"):
                    nq = st.text_input("Question", r["question"])
                    na = st.text_area("Answer", r["answer"], height=120)
                    c1, c2 = st.columns(2)
                    if c1.form_submit_button("Save changes", type="primary"):
                        try:
                            kb.update_learned(r["id"], nq, na, user)
                            st.rerun()
                        except ValueError as e:
                            st.warning(str(e))
                    if c2.form_submit_button("Switch off" if r.get("active", 1) else "Switch on"):
                        kb.set_learned(r["id"], active=not r.get("active", 1))
                        st.rerun()


# ---------------------------------------------------------------------------
# KNOWLEDGE GAPS
# ---------------------------------------------------------------------------
def gaps(region, user):
    st.header("Knowledge Gaps")
    st.caption("Questions Vidya could not answer, where documents disagree, or that staff marked 👎 - "
               "grouped by similar wording, most-asked first. Fix these to improve the process.")
    c1, c2, c3, c4 = st.columns([1, 1.6, 1.2, 1.3])
    reg = c1.selectbox("Region", ["All"] + config.REGIONS)
    statuses = c2.multiselect("Status", kb.GAP_STATUSES,
                              default=["Open", "SOP update needed", "In progress", "Possibly answered"])
    kind = c3.selectbox("Type", ["All", "No answer", "Conflict", "Not helpful"])
    groups = kb.knowledge_gaps(None if reg == "All" else reg, include_closed=True)
    c4.write("")
    c4.download_button("⬇ Export gaps to Excel", kb.export_excel(kb.questions_df(None if reg == "All" else reg),
                                                                [g for g in groups if g["status"] in statuses]),
                       file_name=f"Vidya_gaps_{datetime.now():%Y-%m-%d}.xlsx", width="stretch")
    shown = [g for g in groups if g["status"] in statuses and (kind == "All" or g["type"] == kind)]
    k = st.columns(4)
    ui.kpi(k[0], "🕳️", len([g for g in groups if g["status"] not in ("Fixed", "Closed")]), "Open gaps")
    ui.kpi(k[1], "🔁", sum(g["count"] for g in shown), "Times asked (shown)")
    ui.kpi(k[2], "⚠️", len([g for g in shown if g["type"] == "Conflict"]), "Conflicts")
    ui.kpi(k[3], "🧠", len([g for g in groups if g["status"] == "Possibly answered"]), "Possibly answered")
    st.write("")
    if not shown:
        st.success("No gaps match. 🎉")
    glimit = st.session_state.setdefault("gap_limit", 10)
    for n, g in enumerate(shown[:glimit]):
        with st.container(border=True):
            badge = {"Conflict": "conflict", "No answer": "not_found", "Not helpful": "not_found"}[g["type"]]
            st.markdown(f"{ui.result_badge(badge).replace(ui.RESULT_BADGE[badge][0], g['type'])} "
                        + " ".join(ui.region_pill(r) for r in g["regions"])
                        + f" &nbsp; **{ui.esc(g['question'])}**", unsafe_allow_html=True)
            st.caption(f"Asked {g['count']}× · topic {g['topic']} · first {g['first']} · last {g['last']} · "
                       f"by {', '.join(g['users'][:4])}")
            if len(g["examples"]) > 1:
                st.markdown("Also asked as: " + " · ".join(f"*{ui.esc(x)}*" for x in g["examples"][1:]))
            if g["note"]:
                st.info(g["note"])
            c1, c2, c3, c4 = st.columns([1.3, 1.2, 2.2, 0.8])
            status = c1.selectbox("Status", kb.GAP_STATUSES, index=kb.GAP_STATUSES.index(g["status"])
                                  if g["status"] in kb.GAP_STATUSES else 0, key=f"gs{n}")
            owner = c2.text_input("Owner", g["owner"], key=f"go{n}")
            note = c3.text_input("Note / action", g["note"], key=f"gn{n}")
            c4.write("")
            if c4.button("Save", key=f"gsv{n}"):
                kb.set_gap(g["ids"], status, owner, note)
                st.rerun()
            with st.expander("✍ Write the answer now"):
                with st.form(f"gt{n}"):
                    qtext = st.text_input("Question", g["question"])
                    ans = st.text_area("Correct answer", height=100)
                    if st.form_submit_button("✅ Save & teach (closes this gap)", type="primary") and ans.strip():
                        for r in g["regions"]:
                            kb.teach(r, qtext, ans.strip(), user)
                        kb.set_gap(g["ids"], "Fixed", owner or user, "Answered by process team")
                        st.rerun()
    if len(shown) > glimit and st.button("Show 10 more gaps"):
        st.session_state.gap_limit = glimit + 10
        st.rerun()


# ---------------------------------------------------------------------------
# LEARNED ANSWERS
# ---------------------------------------------------------------------------
def learned(region, user):
    st.header("Learned Answers")
    st.caption("Verified answers come from the process team. Auto-learned answers are saved by Vidya "
               f"when {config.AUTO_LEARN_MIN_VOTES} different people (or one admin) mark the same answer 👍; "
               "a 👎 switches an auto-learned answer off automatically.")
    t1, t2, t3 = st.tabs(["✅ Verified", "🧠 Auto-learned (staff 👍)", "➕ Add a Q&A"])
    with t1:
        rows = kb.list_learned(region, "verified") + kb.list_learned(config.GLOBAL_REGION, "verified")
        if not rows:
            st.info("No verified answers yet for this region.")
        for l in rows:
            with st.container(border=True):
                a, b = st.columns([6, 1])
                a.markdown(ui.region_pill(l["region"]) + f" **Q:** {ui.esc(l['question'])}", unsafe_allow_html=True)
                a.markdown(f"**A:** {l['answer']}")
                a.caption(f"by {l['added_by']} · {l['added_at']}")
                if b.button("Delete", key=f"dv{l['id']}"):
                    kb.set_learned(l["id"], delete=True)
                    st.rerun()
    with t2:
        rows = kb.list_learned(region, "auto")
        if not rows:
            st.info("Nothing auto-learned yet. Answers staff mark 👍 appear here.")
        for l in rows:
            with st.container(border=True):
                a, b = st.columns([6, 1.4])
                state = "🟢 in use" if l["active"] else "⚪ switched off"
                a.markdown(f"{state} · 👍 {l.get('votes') or 1} · **Q:** {ui.esc(l['question'])}")
                a.markdown(l["answer"])
                if b.button("Make verified", key=f"mv{l['id']}"):
                    kb.set_learned(l["id"], make_verified=True, admin=user)
                    st.rerun()
                if b.button("Switch off" if l["active"] else "Switch on", key=f"ta{l['id']}"):
                    kb.set_learned(l["id"], active=not l["active"])
                    st.rerun()
                if b.button("Delete", key=f"da{l['id']}"):
                    kb.set_learned(l["id"], delete=True)
                    st.rerun()
    with t3:
        with st.form("add_qa", clear_on_submit=True):
            r = st.selectbox("Region", config.REGIONS + [config.GLOBAL_REGION], index=config.REGIONS.index(region))
            q = st.text_input("Question")
            a = st.text_area("Answer")
            if st.form_submit_button("Save", type="primary") and q and a:
                kb.teach(r, q, a, user)
                st.success("Saved.")


# ---------------------------------------------------------------------------
# ANNOUNCEMENTS (admin form, shown on the Announcements page)
# ---------------------------------------------------------------------------
def announcements_admin(user):
    """SMEs and admins: post an update. Up to 250 words. 'Running line' = the red line that runs
    across the top of every agent screen."""
    limit = kb.ANNOUNCEMENT_MAX_WORDS
    with st.expander("➕ Post an announcement (red running line on agent screens)", expanded=True):
        n = st.session_state.get("ann_n", 0)                 # new keys after posting = empty form
        c1, c2 = st.columns(2)
        title = c1.text_input("Title", key=f"ann_t{n}", max_chars=120,
                              placeholder="Refund process updated for NORAM from Monday")
        level = c2.selectbox("Type", ["Alert", "New SOP", "Process update", "Maintenance", "Info"], key=f"ann_l{n}")
        body = st.text_area(f"Message (up to {limit} words)", key=f"ann_b{n}", height=140)
        words = len((title + " " + body).split())
        (st.caption if words <= limit else st.error)(f"{words} / {limit} words")
        c3, c4, c5 = st.columns(3)
        reg = c3.selectbox("Show to", ["All"] + config.REGIONS, key=f"ann_r{n}")
        until = c4.date_input("Show until (optional)", value=None, key=f"ann_u{n}",
                              help="Empty = until someone hides it")
        ticker = c5.checkbox("Red running line", value=True, key=f"ann_k{n}",
                             help="Also scroll it across the top of every agent screen")
        if st.button("Post announcement", type="primary", disabled=not title.strip() or words > limit,
                     key=f"ann_go{n}"):
            try:
                kb.add_announcement(title, body, level, reg, user, ticker=ticker,
                                    until=until.strftime("%Y-%m-%d") if until else None)
                st.session_state["ann_n"] = n + 1
                st.toast("Posted - agents see it now.", icon="📢")
                st.rerun()
            except ValueError as e:
                st.error(str(e))


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
def settings(region, user):
    st.header("Settings")
    if config.password_warning():
        st.error(config.password_warning(), icon="🔐")
    src = {"secrets": "✔ from Secrets", "environment": "✔ from environment", "default": "✖ not set"}
    st.caption("Secrets check (values are never shown): "
               + ("Secrets read OK" if config.SECRETS_STATUS["state"] == "ok" else
                  "⚠ Secrets could not be read" if config.SECRETS_STATUS["state"] == "error" else "no Secrets saved")
               + f" · ADMIN_PASSWORD {src[config.SECRET_SOURCE.get('ADMIN_PASSWORD', 'default')]}"
               + f" · GEMINI_API_KEY {src[config.SECRET_SOURCE.get('GEMINI_API_KEY', 'default')]}"
               + f" · AI_PROVIDER {src[config.SECRET_SOURCE.get('AI_PROVIDER', 'default')]}"
               + f" · SME_PASSWORD {src[config.SECRET_SOURCE.get('SME_PASSWORD', 'default')]}")
    if not config.SME_PASSWORD:
        st.info("SMEs sign in with the admin password until you add SME_PASSWORD = \"...\" to Secrets "
                "(one line). Give SMEs their own password so they don't get admin access.", icon="🧑‍🏫")
    ok, ver = privacy.ocr_ready()
    if ok:
        st.success(f"🔒 Card numbers are hidden in text and screenshots (Tesseract {ver}).", icon="💳")
    else:
        st.error(f"🔒 Card numbers are hidden in text, but screenshots are switched off: {ver}. "
                 "Streamlit Cloud: make sure packages.txt (tesseract-ocr) is in GitHub and reboot. "
                 "Laptop/server: see README → Card numbers.", icon="💳")
    lob_settings(user)
    backup_settings(user)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='om-section'>AI connection</div>", unsafe_allow_html=True)
        st.write(f"Answering now: **{llm.label()}**")
        st.write(f"Order it tries: **{llm.chain_label()}**")
        st.caption("If one AI fails (free quota used up, key rejected, Ollama closed) Vidya moves to the next "
                   "one at once and skips the failed one for a few minutes. With no AI at all it still answers "
                   "from the documents.")
        st.write("Meaning search: **" + llm.embed_label() + "**")
        import rerank
        rs, rd = rerank.status()
        icon = {"ready": "✅", "loading": "⏳", "failed": "⚠️"}.get(rs, "➖")
        st.write(f"Reranker: **{icon} {rs}** · {rd}")
        if rs == "loading":
            st.caption("The reranker model is downloading / starting in the background (first time only). "
                       "Answers use rules-only ranking until it is ready.")
        alive = llm.local_alive()
        st.write("Ollama (local AI): **" + (f"running at {config.OLLAMA_URL}" if alive else "not reachable") + "**"
                 + (f" · model {config.LOCAL_MODEL}" if alive else ""))
        if "local" in llm.providers():
            st.caption("Screenshots with local AI: "
                       + (config.LOCAL_VISION_MODEL or "off (set LOCAL_VISION_MODEL, e.g. gemma3:4b)"))
        elif not alive:
            st.caption("Ollama works only when Vidya runs on the SAME computer or company server as Ollama "
                       "(START_APP.bat). The Streamlit Cloud app cannot reach Ollama on your laptop.")
        if (config.MEANING_SEARCH or "").lower() == "builtin" and not llm.builtin_available():
            st.warning("MEANING_SEARCH is 'builtin' but the fastembed library is not installed "
                       "(check requirements.txt).")
        if kb.check_embed_space():
            st.info("The meaning-search model changed - press 'Build semantic index now' once.")
        if st.button("Test AI connection"):
            ok, msg = llm.test_connection()
            (st.success if ok else st.warning)(msg)
        done, total = kb.semantic_coverage()
        st.write(f"Semantic index: **{done} of {total}** knowledge pieces")
        if llm.embeddings_on() and done < total and st.button("Build semantic index now", type="primary"):
            bar = st.progress(0.0)
            try:
                n = kb.reindex_embeddings(bar.progress)
                st.success(f"Indexed {n} items.")
            except llm.AIError as e:
                st.error(str(e))
        if llm.provider() == "gemini" or llm.embedder() == "gemini":
            st.warning("Gemini free tier: Google may use the content you send to improve its products. "
                       "Use sample or non-confidential SOPs on the free tier; for real company SOPs use a "
                       "paid key or company-approved AI.")
        st.caption("Add keys in Streamlit → App → Settings → Secrets (each on ONE line):  "
                   "GEMINI_API_KEY = \"...\"  ·  ANTHROPIC_API_KEY = \"...\" (optional backup)  ·  "
                   "MEANING_SEARCH = \"builtin\" (meaning search with no key and no quota)  ·  "
                   "AI_BACKUP = \"auto\" | \"on\" | \"off\".  Laptop with Ollama: AI_PROVIDER = \"local\" "
                   "(SOPs never leave the laptop) or keep \"auto\" to use Ollama only as backup.")
    with c2:
        st.markdown("<div class='om-section'>Standard replies</div>", unsafe_allow_html=True)
        st.write("**When no answer is found:**")
        st.info(config.FALLBACK_TEXT)
        st.write("**When documents disagree:**")
        st.warning(config.CONFLICT_TEXT)
        st.caption("Change these wordings in config.py (FALLBACK_TEXT, CONFLICT_TEXT).")
        st.markdown("<div class='om-section'>Self-learning</div>", unsafe_allow_html=True)
        st.write(f"Auto-learn from 👍: **{'On' if config.AUTO_LEARN else 'Off'}** - after 👍 from "
                 f"{config.AUTO_LEARN_MIN_VOTES} different people, or 1 admin")
        st.write("👍 / 👎 also move passages up / down for similar questions; new uploads re-check open gaps.")


def lob_settings(user):
    st.markdown("<div class='om-section'>Lines of business (tabs) and sub-LOBs</div>", unsafe_allow_html=True)
    st.caption("Each LOB is a tab above the chat; its sub-LOBs are the chips under it. Questions search the "
               "selected LOB first (sub-LOB documents ranked highest), then the other LOBs if needed. "
               "Separate sub-LOBs with commas.")
    lobs = kb.get_lobs()
    base = pd.DataFrame([{"LOB": k, "Sub-LOBs": ", ".join(v)} for k, v in lobs.items()])
    with st.form("lobs"):
        edited = st.data_editor(base, num_rows="fixed", hide_index=True, width="stretch", disabled=["LOB"],
                                column_config={"Sub-LOBs": st.column_config.TextColumn(width="large")})
        if st.form_submit_button("Save sub-LOBs", type="primary"):
            try:
                kb.save_lobs({r["LOB"]: [x.strip() for x in str(r["Sub-LOBs"] or "").split(",")]
                              for _, r in edited.iterrows()})
                st.success("Saved. The chips update immediately.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    st.caption("Adding a new LOB tab (besides " + ", ".join(config.REGIONS) + ") is a one-line change in "
               "config.py (REGIONS).")


def backup_settings(user):
    st.markdown("<div class='om-section'>Backup & restore</div>", unsafe_allow_html=True)
    st.caption("Free Streamlit hosting can erase uploaded documents and learned answers when the app restarts. "
               "Download a backup after uploading documents, and restore it if the library is ever empty.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Prepare backup", key="mkbackup"):
            st.session_state["backup_bytes"] = kb.make_backup()
        if st.session_state.get("backup_bytes"):
            st.download_button("⬇ Download backup (.zip)", st.session_state["backup_bytes"], type="primary",
                               file_name=f"Vidya_backup_{datetime.now():%Y-%m-%d_%H%M}.zip", mime="application/zip")
    with c2:
        up = st.file_uploader("Restore from a backup .zip", type=["zip"], key="restore_zip")
        if up is not None and st.button("Restore now (replaces current library)", key="dorestore"):
            try:
                n = kb.restore_backup(up.getvalue())
                st.success(f"Restored {n} documents with their questions, learned answers and settings.")
            except ValueError as e:
                st.error(str(e))
