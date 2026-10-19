"""
farerule.py - reads a fare rule (category 16 PENALTIES) the agent pasted from Amadeus (FQN / TQN ...PE)
or Sabre (RD ...*PN / *RE), or a screenshot of it, and says which fee applies.

How it decides (no AI needed, every line it relies on is shown to the agent):
  1. Split the rule into CANCELLATIONS and CHANGES, and each into ANY TIME / BEFORE DEPARTURE /
     AFTER DEPARTURE blocks.
  2. Read the agent's case from the question: change or refund, before or after departure, no-show.
     When the question does not say, every case is shown.
  3. For each case: the fee line (CHARGE GBP 150.00 FOR REISSUE...), NON-REFUNDABLE, NOT PERMITTED or
     PERMITTED, the no-show lines, and the conditions around it (fare difference, residual value,
     per ticket / per fare component, highest penalty of the combined fares, taxes / YQ / YR, waivers,
     EMD collection, reissue deadline).
  4. If the rule text does not clearly say, Vidya says so - it never guesses a fee.
The agent always confirms with the GDS (Amadeus FXQ / ATC, Sabre WFR / refund quote) before collecting.
"""
import re

# The three checks shown with every fare-penalty answer (wording from the process team)
CHECKS = [
    ("Verify Penalties", "Confirm change fees, cancellation charges, fare differences and applicable taxes."),
    ("Check Ticket Status", "Validate ticket coupons, Ticket/MCO validity, no-show conditions and eligibility "
                            "before processing."),
    ("Check PNR history", "PNR history to validate ticket cancellation time - for now show, cancellation by "
                          "airline or voluntary."),
]

_PENALTY_Q = re.compile(r"\b(penalt\w*|change fee|cancell?ation (fee|charge|penalty)|refund (fee|charge|penalty)|"
                        r"no[- ]?show|fare rules?|cat(egory)? ?16|reissue fee|rebooking fee|date change fee|"
                        r"refundable|non[- ]refundable|re-?issue charge|change charge)\b", re.I)


def is_penalty_question(text):
    return bool(_PENALTY_Q.search(text or ""))


# ---------------------------------------------------------------------------------------------
# 1. Is this a pasted fare rule?
# ---------------------------------------------------------------------------------------------
_RULE_WORDS = ["PENALTIES", "CANCELLATIONS", "CHANGES", "NON-REFUNDABLE", "REISSUE", "REVALIDATION",
               "NO-SHOW", "NO SHOW", "BEFORE DEPARTURE", "AFTER DEPARTURE", "ANY TIME", "FARE COMPONENT",
               "CHARGE ", "REFUND", "WAIVED", "TICKET IS"]


_NOISE = re.compile(r"^\s*(PAGE\s*\d+\s*/\s*\d+|[_>-]?\s*MD|[-~=_\s]{4,}|W[Tt]|WE|SEE OE)\s*$")


def split_question(text):
    """(agent's own question, rule text). Rule lines are the upper-case GDS lines."""
    ask, rule = [], []
    for line in (text or "").replace("\r", "").split("\n"):
        if _NOISE.match(line.upper()):
            continue                                       # PAGE 2/7, > MD, ----- lines
        letters = [c for c in line if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) >= 0.85 and len(letters) >= 4:
            rule.append(line)
        elif line.strip():
            ask.append(line.strip())
    return " ".join(ask), "\n".join(rule)


def detect(text):
    """True when the text contains a category 16 fare rule (pasted or read from a screenshot)."""
    _, rule = split_question(text)
    up = rule.upper()
    hits = sum(1 for w in _RULE_WORDS if w in up)
    return len(rule.split("\n")) >= 3 and hits >= 4 and ("CANCELLATION" in up or "CHANGES" in up)


# ---------------------------------------------------------------------------------------------
# 2. Parse
# ---------------------------------------------------------------------------------------------
_SECTION = re.compile(r"^\s*(?:\d+\s*\.\s*PENALTIES\s*)?(CANCELL?ATIONS?|CHANGES)\s*$")
_TIMING = re.compile(r"^\s*(ANY TIME|BEFORE DEPARTURE|AFTER DEPARTURE)\s*$")


def clean(rule):
    out = []
    for line in rule.upper().split("\n"):
        line = line.replace("¥", "Y").rstrip()
        line = re.sub(r"^[(\[|]+", "", line.strip())                   # OCR: "(CANCELLATIONS"
        if not line or _NOISE.match(line):
            continue
        out.append(line)
    return out


def parse(rule):
    """{'CANCELLATIONS': {'ANY TIME': [lines], ...}, 'CHANGES': {...}} - lines in order."""
    lines = clean(rule)
    data = {"CANCELLATIONS": {}, "CHANGES": {}}
    sec, tim = None, "ANY TIME"
    prev = ""
    for line in lines:
        m = _SECTION.match(line)
        if m and re.search(r"\b(AND|OR|BOTH|OF|IN)\s*$", prev):
            m = None                                   # "... IN BOTH CANCELATION AND / CHANGES" is text
        prev = line
        if m:
            sec = "CHANGES" if m.group(1).startswith("CHANGE") else "CANCELLATIONS"
            tim = "ANY TIME"
            continue
        if re.match(r"^\s*\d+\s*\.\s*PENALTIES\s*$", line):
            continue
        t = _TIMING.match(line)
        if t and sec:
            tim = t.group(1)
            continue
        if sec:
            data[sec].setdefault(tim, []).append(line.strip())
    return data


_START = re.compile(r"^\s*(\(?[A-H]\s*[.)/]|\d{1,2}\s*[.)\-]\s*[A-Z]|NOTE\s*-|//|.*-{3,}|[A-Z][A-Z /]+ -\s*$)")


_STARTER = re.compile(r"^\s*(REFUND|WAIVED|TICKET IS|CHARGE|CHANGES?|CANCELL?ATIONS?|IN CASES? OF|WHEN|ANY |INFANTS?|"
                      r"CHILD|YQ|YR|A CHANGE|REISSUE|REVALIDATION|NO[- ]?SHOW|EXCEPT|THE (FARE|NEW|ORIGINAL|CHANGE|"
                      r"OPTIONAL|REFUND)|FEES?|APPLICABLE|FOR (ALL|ANY|THE)|IF |GERMAN|PASSENGER)")


def _sentences(lines):
    """Join wrapped GDS lines into sentences. A new sentence starts at a full stop, a list item
    (A. / 1. / 1-), a NOTE line or a ---- heading line - never in the middle of a wrapped line."""
    chunks, cur = [], []
    for line in lines:
        prev = cur[-1] if cur else ""
        joined = bool(re.search(r"(\b(AND|OR|THE|OF|TO|A|AN|FOR|WITH|IS|ARE|BE|IN|ON|AT|BY|IF|NOT|THAN|FROM|PER|"
                                r"WILL|MUST|SHALL|ANY|ALL|NO)|[,/-])\s*$", prev))
        if cur and not joined and _STARTER.match(line):
            chunks.append(" ".join(cur))
            cur = []
        if _START.match(line) and cur:
            chunks.append(" ".join(cur))
            cur = []
        if re.match(r"^\s*.*-{3,}", line) or re.match(r"^\s*NOTE\s*-\s*$", line):
            continue                                       # headings / NOTE - carry no rule text
        cur.append(line.strip())
    if cur:
        chunks.append(" ".join(cur))
    out = []
    for c in chunks:
        c = re.sub(r"\s+", " ", c)
        for p in re.split(r"(?<=[A-Z0-9)\]])\s*\.\s+(?=[A-Z(])", c):
            p = p.strip(" .")
            if len(p) >= 6:
                out.append(p + ".")
    return out


# ---------------------------------------------------------------------------------------------
# 3. The agent's case
# ---------------------------------------------------------------------------------------------
def scenario(ask):
    a = (ask or "").lower()
    action = None
    if re.search(r"\b(refund|cancel|cancell?ation|cxl)\w*", a):
        action = "refund"
    if re.search(r"\b(change|reissue|re-issue|rebook|date change|exchange|revalidat|reschedul|modif)\w*", a):
        action = "change" if action is None else "both"
    noshow = bool(re.search(r"no[- ]?show|did not fly|didn't fly|missed (the )?flight|not (turn|show)", a))
    timing = None
    if re.search(r"\b(after departure|after travel|partly used|partially used|part used|flown|"
                 r"half used|return (sector|leg) (only|unused)|outbound (used|flown)|travel (started|commenced))", a):
        timing = "AFTER DEPARTURE"
    elif re.search(r"\b(before departure|before travel|unused|not (yet )?flown|fully unused|totally unused|"
                   r"before (the )?(first )?flight|not travelled)", a):
        timing = "BEFORE DEPARTURE"
    return action, timing, noshow


# ---------------------------------------------------------------------------------------------
# 4. Decide
# ---------------------------------------------------------------------------------------------
_FEE = re.compile(r"\b(?:PER\s+(?:TICKET|TRANSACTION|DIRECTION|PERSON)\s+)?CHARGE\s+([A-Z]{3})\s*([\d,]+(?:\.\d+)?)"
                  r"(?:\s+FOR\s+([A-Z/ -]+?))?(?=[.,;]|\s*$|\s+(?:NOTE|WAIVED|PER))")
_FEE2 = re.compile(r"\b(NO[- ]?SHOW FEE|SERVICE FEE|CANCELL?ATION FEE)\s*(?:OF\s*)?([A-Z]{3})\s*([\d,]+(?:\.\d+)?)")


def _blocks(data, sec, timing):
    """Lines that apply: the timing block, else ANY TIME (both when both exist)."""
    d = data.get(sec, {})
    if timing and timing in d:
        return d[timing] + d.get("ANY TIME", [])
    if timing is None:
        return [l for v in d.values() for l in v]
    return d.get("ANY TIME", []) or [l for v in d.values() for l in v]


def _find(sents, *pats):
    return [s for s in sents if any(re.search(p, s) for p in pats)]


_SHOT = {"on": False}         # True while reading text that came from a screenshot (digits may be misread)


def _money(cur, amt):
    amt = amt.replace(",", "")
    if _SHOT["on"] and re.fullmatch(r"\d+\.[08]{1,2}", amt):
        amt = amt.split(".")[0] + ".00"                  # screenshot: ".80" / ".08" is almost always ".00"
    try:
        return f"{cur} {float(amt):,.2f}"
    except ValueError:
        return f"{cur} {amt}"


def decide(data, action, timing, noshow):
    """One case -> dict(verdict, fee, lines (evidence), notes)."""
    sec = "CHANGES" if action == "change" else "CANCELLATIONS"
    if not data.get(sec):
        return {"verdict": "unknown", "fee": None, "evidence": [],
                "notes": [f"The pasted text has no {sec} part - paste the full category 16."]}
    sents = _sentences(_blocks(data, sec, timing))
    all_sec = _sentences(_blocks(data, sec, None))
    evidence, notes = [], []
    verdict, fee = "unknown", None
    word = "change" if action == "change" else "refund"

    def ns(s):
        """A sentence only about no-show (a fee line "FOR NO-SHOW/REISSUE/..." covers every case)."""
        return bool(re.search(r"NO[- ]?SHOW", s)) and not re.search(
            r"(CHARGE [A-Z]{3}.{0,20}|PERMITTED) FOR [A-Z/ -]*(REISSUE|REVALIDATION|CANCEL|REFUND)", s)

    # --- no-show first: it overrides the before/after line
    if noshow:
        whole = _sentences([l for sct in data.values() for v in sct.values() for l in v])
        bad = _find(all_sec, r"NO[- ]?SHOW.{0,110}NOT PERMITTED", r"NOT PERMITTED.{0,40}NO[- ]?SHOW",
                    r"NON[- ]?REFUNDABLE.{0,40}NO[- ]?SHOW", r"NO[- ]?SHOW.{0,80}NON[- ]?REFUNDABLE")
        dead = _find(whole, r"NO[- ]?SHOW.{0,60}NO REMAINING VALUE", r"NO REMAINING VALUE.{0,40}NO[- ]?SHOW")
        allowed = _find(all_sec, r"PERMITTED FOR [A-Z/ -]*NO[- ]?SHOW")
        fees = [(s, m) for s in all_sec for m in [_FEE.search(s)] if m and m.group(3) and "SHOW" in m.group(3)]
        fees2 = [(s, m) for s in all_sec for m in [_FEE2.search(s)] if m and "SHOW" in m.group(1)]
        if bad or (dead and action == "change"):
            evidence += (bad or dead)[:2]
            verdict = "non_refundable" if action == "refund" else "not_permitted"
            if any("BECOMES NON" in b for b in bad):
                notes.append("Only the fare component with the no-show becomes non-refundable.")
        elif fees2:
            s, m = fees2[0]
            evidence.append(s)
            verdict, fee = "fee", _money(m.group(2), m.group(3))
            extra = [x for x in _FEE2.finditer(s) if "CANCEL" in x.group(1)]
            if extra and "HIGHER OF" not in s:
                fee += " no-show fee + " + _money(extra[0].group(2), extra[0].group(3)) + " cancellation"
            if "HIGHER OF THE TWO" in s:
                notes.append("Charge the HIGHER of the no-show fee and the highest change fee of the changed "
                             "fare components.")
        elif fees:
            s, m = fees[0]
            evidence.append(s)
            verdict, fee = "fee", _money(m.group(1), m.group(2))
        elif allowed:
            evidence.append(allowed[0])
            verdict = "free"
        else:
            ns_lines = _find(all_sec, r"NO[- ]?SHOW")
            if ns_lines:
                evidence += ns_lines[:2]
                notes.append("Read the no-show line(s) below - the rule has no simple fee for no-show.")
            else:
                notes.append("The rule says nothing specific about no-show in this part - the after-departure "
                             "conditions are shown; confirm with the airline / your SME.")
                timing = "AFTER DEPARTURE"
                sents = _sentences(_blocks(data, sec, timing))
    if verdict == "unknown":
        main = [s for s in sents if not ns(s)]
        fee_lines = [(s, _FEE.search(s)) for s in main if _FEE.search(s)]
        nonref = _find(main, r"TICKET IS NON[- ]?REFUNDABLE(?!.{0,30}NO[- ]?SHOW)(?!.{0,20}IF )",
                       r"FARE COMPONENT IS NON[- ]?REFUNDABLE")
        notperm = _find(main, r"\b(CHANGES|REFUNDS?|CANCELL?ATIONS?) (ARE )?NOT PERMITTED(?!.{0,30}NO[- ]?SHOW)")
        perm = _find(main, r"^(CHANGES|CANCELL?ATIONS|REISSUE.{0,40}) PERMITTED", r"IS REFUNDABLE",
                     r"WHOLLY UNUSED TICKET IS REFUNDABLE")
        if action == "change" and fee_lines:
            s, m = fee_lines[0]
            verdict, fee = "fee", _money(m.group(1), m.group(2))
            evidence.append(s)
        elif action == "refund" and nonref and not fee_lines:
            verdict = "non_refundable"
            evidence += nonref[:1]
        elif action == "refund" and fee_lines:
            s, m = fee_lines[0]
            verdict, fee = "fee", _money(m.group(1), m.group(2))
            evidence.append(s)
            if nonref:
                evidence += nonref[:1]
                notes.append("Part of the fare is non-refundable - see the line below.")
        elif notperm:
            verdict = "not_permitted"
            evidence += notperm[:1]
        elif perm:
            verdict = "free"
            evidence += perm[:1]
        elif nonref and action == "change":
            notes.append("The fare is non-refundable; the rule does not show a change fee here.")
            evidence += nonref[:1]
    # --- conditions around it
    cond = _conditions(sents if verdict != "unknown" else all_sec, action)
    whole_rule = _sentences([l for sct in data.values() for v in sct.values() for l in v])
    if action == "refund" and not any(l == "Taxes / YQ / YR" for l, _ in cond):
        tax = _conditions(whole_rule, "refund")                 # often in a note shared by both parts
        cond += [(l, x) for l, x in tax if l.startswith("Taxes")][:2]
    waiver = _find(all_sec, r"WAIVED FOR") or _find(whole_rule, r"WAIVED FOR")
    return {"verdict": verdict, "fee": fee, "evidence": list(dict.fromkeys(evidence)), "notes": notes,
            "conditions": cond, "waiver": waiver[:1], "word": word}


_COND = [
    ("Fare difference", [r"DIFFERENCE (WILL BE|IS) COLLECTED", r"CHARGE (THE )?(FARE )?DIFFERENCE",
                         r"DIFFERENCE IN FARE", r"FARE DIFFERENCE"]),
    ("Lower new fare / residual", [r"RESIDUAL", r"NO CREDIT", r"CREDIT OF THE RESIDUAL",
                                   r"EQUAL (TO )?OR HIGHER THAN (THE )?PREVIOUS"]),
    ("Fee is charged", [r"PER (TRANSACTION|TICKET|FARE COMPONENT|DIRECTION|PRICING UNIT|PERSON)",
                        r"APPLIES FOR EVERY", r"SUBJECT TO EACH CHANGE TRANSACTION", r"APPLY BY FARE COMPONENT"]),
    ("Several fare components", [r"HIGHEST (PENALTY|FEE|CHANGE FEE|CANCELL?ATION)", r"SUM OF THE CANCELL?ATION",
                                 r"MOST RESTRICTIVE", r"STRICTER FARE CONDITIONS"]),
    ("Taxes / YQ / YR", [r"UNUSED TAX", r"(?<![-/A-Z])YQ\b", r"(?<![-/A-Z])YR\b", r"CARRIER IMPOSED"]),
    ("How to collect", [r"COLLECTED VIA AN EMD", r"EMD ISSUANCE"]),
    ("Deadline", [r"SAME DAY", r"WITHIN 24 HOURS", r"WITHIN (THE )?TICKET VALIDITY", r"BEFORE DEPARTURE OF THE "
                  r"ORIGINALLY", r"NO LATER THAN"]),
    ("Child / infant", [r"CHILD.{0,15}DISCOUNT", r"INFANTS? WITHOUT (A )?SEATS?"]),
    ("Exception", [r"REFUND PERMITTED .{0,40}IN CASE OF", r"REJECTION OF VISA"]),
]


def _conditions(sents, action):
    out = []
    for label, pats in _COND:
        if action == "refund" and label in ("Fare difference", "Lower new fare / residual"):
            continue
        hit = _find(sents, *pats)
        if hit:
            out.append((label, hit[0]))
            if label == "Taxes / YQ / YR" and len(hit) > 1:
                out.append((label + " ", hit[1]))
    return out


# ---------------------------------------------------------------------------------------------
# 5. Answer
# ---------------------------------------------------------------------------------------------
_LABEL = {("change", "BEFORE DEPARTURE"): "Change before departure", ("change", "AFTER DEPARTURE"): "Change after departure",
          ("refund", "BEFORE DEPARTURE"): "Refund before departure", ("refund", "AFTER DEPARTURE"): "Refund after departure",
          ("change", None): "Change", ("refund", None): "Refund"}


def _verdict_text(r):
    w = r["word"]
    kind = "" if "+" in (r["fee"] or "") else (" change fee" if w == "change" else " cancellation fee")
    if _SHOT["on"]:
        kind += " ⚠ read from a screenshot - check the amount on screen"
    return {"fee": f"**{r['fee']}**{kind}",
            "non_refundable": "**Non-refundable**",
            "not_permitted": f"**{w.title()} not permitted**",
            "free": f"**{'Change' if w == 'change' else 'Refund'} permitted - no fee in the rule**",
            "unknown": "**Not clear from the rule text**"}[r["verdict"]]


def answer(text, from_screenshot=False):
    """Returns dict(answer markdown, cases, determined (bool), scenario) or None if no rule found."""
    ask, rule = split_question(text)
    if not detect(text):
        return None
    _SHOT["on"] = bool(from_screenshot)
    try:
        return _answer(ask, rule, from_screenshot)
    finally:
        _SHOT["on"] = False


def _answer(ask, rule, from_screenshot):
    data = parse(rule)
    action, timing, noshow = scenario(ask)
    if action in (None, "both"):
        actions = ["change", "refund"]
    else:
        actions = [action]
    if noshow:
        cases = [(a, timing, True) for a in actions]
    elif timing:
        cases = [(a, timing, False) for a in actions]
    else:
        cases = [(a, t, n) for a in actions for t, n in (("BEFORE DEPARTURE", False), ("AFTER DEPARTURE", False),
                                                            (None, True))]
    results = []
    for a, t, n in cases:
        r = decide(data, a, t, n)
        r["label"] = "No-show - " + ("change" if a == "change" else "refund") if n else _LABEL[(a, t)]
        results.append(r)
    single = len(results) == 1
    out = []
    if single:
        r = results[0]
        out.append(f"**{r['label']}:** {_verdict_text(r)}")
    else:
        out.append("| Case | What the fare rule says |\n| --- | --- |")
        for r in results:
            out.append(f"| {r['label']} | {_verdict_text(r)} |")
        out.append("")
        if not ask:
            out.append("_Tell Vidya the case (e.g. \"change before departure\" or \"no-show refund\") for one answer._")
    # conditions + evidence (for the cases shown; one block for single, merged for table)
    seen = set()
    cond_lines = []
    for r in results:
        for label, s in r["conditions"]:
            if (label, s) not in seen and label not in {l for l, _ in cond_lines}:
                seen.add((label, s))
                cond_lines.append((label, s))
    if cond_lines:
        out.append("\n**Conditions in this rule:**")
        out += [f"- **{label.strip()}:** {s}" for label, s in cond_lines]
    notes = list(dict.fromkeys(n for r in results for n in r["notes"]))
    waivers = list(dict.fromkeys(w for r in results for w in r["waiver"]))
    if waivers:
        notes.append("Waiver: " + waivers[0])
    if notes:
        out.append("\n**Note:** " + " ".join(notes))
    ev = list(dict.fromkeys(e for r in results for e in r["evidence"]))
    if ev:
        out.append("\n**Rule lines Vidya used:**")
        out += [f"> {e}" for e in ev[:6]]
    if from_screenshot:
        out.append("\n_Read from a screenshot: text recognition can misread digits (0 / 8). Check every amount "
                   "on the GDS screen - or paste the fare rule as text for an exact reading._")
    out.append("\n_Confirm the final amount with the GDS before collecting (Amadeus FXQ / ATC refund, "
               "Sabre WFR / refund quote)._")
    determined = all(r["verdict"] != "unknown" for r in results)
    return {"answer": "\n".join(out), "cases": results, "determined": determined,
            "scenario": {"action": action, "timing": timing, "noshow": noshow}, "ask": ask}
