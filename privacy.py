"""
privacy.py - hides payment card numbers BEFORE anything is saved, shown or sent to an AI.

  Text        : card numbers become ************1234 (last 4 kept, like on a receipt);
                CVV / CVC / security codes become ***.
  Screenshots : the picture is read with Tesseract OCR (free, open source, runs on this
                server - nothing is sent out) and every card number is covered with a black box.
                The original picture is never written to disk.

If Tesseract is not installed, screenshots are REFUSED (fail closed) - see config.CARD_CHECK.
No masking is perfect (very blurry or handwritten numbers), so agents still see the masked
preview before they send, and are asked to crop card details out themselves.
"""
import io
import re
import shutil

import config


class PrivacyError(Exception):
    pass


# ---------------------------------------------------------------------------
# 1. Which numbers are card numbers?
#    By the card's first digits + length (Visa, Mastercard, Amex, RuPay, Discover, Diners, JCB,
#    UnionPay, Maestro, UATP). The Luhn check is NOT required on purpose: a mistyped card number
#    is still private. (What is HIDDEN is wider: see must_mask - any 13-19 digit number.)
# ---------------------------------------------------------------------------
_CANDIDATE = re.compile(r"(?<!\d)\d(?:[ \-]?\d){11,18}(?!\d)")
_CVV = re.compile(r"\b(cvv2?|cvc2?|csc|cid|security\s*code|card\s*code)(\s*(?:no\.?|number)?\s*[:#=\-]?\s*)(\d{3,4})\b",
                  re.I)


def is_card(digits):
    n = len(digits)
    if n < 14 or n > 19:
        return False
    p2, p3, p4, p6 = int(digits[:2]), int(digits[:3]), int(digits[:4]), int(digits[:6])
    return any([
        digits[0] == "4" and n in (16, 19),                       # Visa
        (51 <= p2 <= 55 or 2221 <= p4 <= 2720) and n == 16,         # Mastercard
        p2 in (34, 37) and n == 15,                                 # Amex
        (p2 in (60, 65, 81, 82) or p3 == 508) and n == 16,          # RuPay
        (p4 == 6011 or p2 == 65 or 644 <= p3 <= 649 or 622126 <= p6 <= 622925) and n >= 16,  # Discover
        (300 <= p3 <= 305 or p2 in (36, 38, 39)) and n >= 14,      # Diners
        3528 <= p4 <= 3589 and n >= 16,                             # JCB
        p2 == 62 and n >= 16,                                       # UnionPay
        p2 in (50, 56, 57, 58, 67) and n >= 14,                     # Maestro
        digits[0] == "1" and n == 15,                               # UATP (airline card)
    ])


# Public TEST card numbers used in training material / SOP examples. Not real cards: they are still
# masked when shown, but a document that contains only these can be uploaded.
TEST_CARDS = {"4111111111111111", "4242424242424242", "4012888888881881", "4000056655665556",
              "5555555555554444", "5105105105105100", "5500000000000004", "2223003122003222",
              "5200828282828210", "378282246310005", "371449635398431", "340000000000009",
              "6011111111111117", "6011000990139424", "30569309025904", "38520000023237",
              "3530111333300000", "3566002020360505", "122000000000003"}


# Ticket numbers (13 digits, also 176-2345678901-02) stay visible only when MASK_MIN_DIGITS is 14
_TICKET = re.compile(r"^\d{3}[ \-]?\d{10}([ \-/]\d{1,2})?$")


def min_digits():
    return max(12, min(19, int(getattr(config, "MASK_MIN_DIGITS", 13))))


def must_mask(raw):
    """What is HIDDEN on screens, in chats, screenshots and saved text: ANY number of
    MASK_MIN_DIGITS (13) to 19 digits (spaces / hyphens allowed), whatever its first digits."""
    digits = re.sub(r"\D", "", raw)
    if not (min_digits() <= len(digits) <= 19):
        return False
    return not (min_digits() >= 14 and _TICKET.match(raw.strip()))


def find_cards(text):
    """All card numbers in a text (digits only)."""
    return [d for d in (re.sub(r"\D", "", m.group()) for m in _CANDIDATE.finditer(text or "")) if is_card(d)]


def real_cards(numbers):
    """Card numbers that are not public test numbers, shown masked (for messages)."""
    return [_mask_digits(d) for d in numbers if d not in TEST_CARDS]


def _mask_digits(digits):
    keep = 4 if getattr(config, "MASK_KEEP_LAST4", True) else 0
    return "*" * (len(digits) - keep) + digits[len(digits) - keep:]


def mask_text(text):
    """Returns (masked text, how many card numbers / codes were hidden)."""
    if not text:
        return text, 0
    count = 0

    def card(m):
        nonlocal count
        digits = re.sub(r"\D", "", m.group())
        if not must_mask(m.group()):
            return m.group()
        count += 1
        return _mask_digits(digits)

    def cvv(m):
        nonlocal count
        count += 1
        return f"{m.group(1)}{m.group(2)}***"

    out = _CANDIDATE.sub(card, text)
    out = _CVV.sub(cvv, out)
    return out, count


def mask(text):
    """Masked text only (for places that do not need the count)."""
    return mask_text(text)[0]


def mask_md(text):
    """Masked text for showing as Markdown: the stars of ************1234 are escaped, so they are
    shown as stars and never turn the rest of the line bold or italic."""
    return re.sub(r"\*{4,}", lambda m: "\\*" * len(m.group()), mask(text or ""))


# ---------------------------------------------------------------------------
# 2. Screenshots
# ---------------------------------------------------------------------------
_OCR_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1", "!": "1",
                          "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9", "q": "9"})


def ocr_ready():
    """(True, version) when Tesseract can be used, else (False, reason)."""
    try:
        import pytesseract
    except ImportError:
        return False, "pytesseract library missing (requirements.txt)"
    cmd = getattr(config, "TESSERACT_CMD", "") or shutil.which("tesseract")
    if not cmd:
        return False, "Tesseract OCR is not installed on this server"
    pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        return True, str(pytesseract.get_tesseract_version())
    except Exception as e:                                   # noqa: BLE001
        return False, f"Tesseract could not start ({str(e)[:80]})"


_BETWEEN = re.compile(r"(?<=\d)[OoDQIl|!iSsBZzgq]+(?=\d)")


def _fix_word(w):
    """OCR often reads 0 as O and 1 as l: fix only letters sitting BETWEEN two digits
    (so the card type in 'CCVI4111...' stays a letter). Same length, so positions don't move."""
    return _BETWEEN.sub(lambda m: m.group().translate(_OCR_FIX), w)


def _boxes_to_cover(data, found):
    """From Tesseract word boxes, find the words that hold card numbers or CVV codes.
    The card numbers read are added to 'found'."""
    lines = {}
    for i, word in enumerate(data["text"]):
        if word and word.strip():
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(i)
    cover = set()
    for idx in lines.values():
        idx.sort(key=lambda i: data["left"][i])
        text, spans = "", []
        for i in idx:
            w = _fix_word(data["text"][i].strip())
            start = len(text) + (1 if text else 0)
            text = (text + " " + w) if text else w
            spans.append((start, start + len(w), i))
        hits = []
        for m in _CANDIDATE.finditer(text):
            d = re.sub(r"\D", "", m.group())
            if must_mask(m.group()):
                hits.append((m.start(), m.end()))       # every 13-19 digit number is covered...
                if is_card(d):
                    found.append(d)                     # ...but only real card patterns can refuse an upload
        for m in _CVV.finditer(text):
            hits.append((m.start(3), m.end(3)))
        for s, e in hits:
            cover.update(i for (a, b, i) in spans if a < e and b > s)
    return cover


def mask_image(img_bytes):
    """Returns (PNG bytes with card numbers covered, how many areas were covered, card numbers read).
    Raises PrivacyError when the picture can't be checked - the caller must then refuse it."""
    ok, why = ocr_ready()
    if not ok:
        raise PrivacyError(f"Screenshots are switched off: {why}. Card numbers could not be checked.")
    import pytesseract
    from PIL import Image, ImageDraw, ImageOps, ImageStat
    try:
        im = Image.open(io.BytesIO(img_bytes))
        im = ImageOps.exif_transpose(im).convert("RGB")      # also drops hidden photo data (EXIF)
    except Exception:                                        # noqa: BLE001
        raise PrivacyError("This file is not a picture Vidya can read.")
    im.thumbnail((2400, 2400))
    scale = 2 if im.width < 1600 else 1                      # small text reads better when enlarged
    big = im.resize((im.width * scale, im.height * scale)) if scale > 1 else im
    gray = ImageOps.grayscale(big)
    dark = ImageStat.Stat(gray).mean[0] < 110                 # GDS screens: light text on dark
    covered, found = [], []
    for g in ((ImageOps.invert(gray),) if dark else (gray,)):
        for psm in (6, 11):
            try:
                data = pytesseract.image_to_data(g, config=f"--psm {psm}", output_type=pytesseract.Output.DICT,
                                                 timeout=30)
            except Exception as e:                           # noqa: BLE001
                raise PrivacyError(f"The screenshot could not be checked for card numbers ({str(e)[:60]}).")
            for i in _boxes_to_cover(data, found):
                covered.append((data["left"][i], data["top"][i], data["width"][i], data["height"][i]))
    draw = ImageDraw.Draw(im)
    for (x, y, w, h) in covered:
        pad = 3 * scale
        draw.rectangle([(x - pad) / scale, (y - pad) / scale, (x + w + pad) / scale, (y + h + pad) / scale],
                       fill=(110, 110, 110), outline=(255, 255, 255))      # grey: visible on dark and light
    out = io.BytesIO()
    im.save(out, format="PNG")
    areas = {(round(x / (40 * scale)), round(y / (20 * scale))) for (x, y, w, h) in covered}
    return out.getvalue(), len(areas), sorted(set(found))


_done = {}          # sha256 of a checked picture -> masked PNG (so the same picture is checked once)


def safe_image(img_bytes):
    """Masked PNG for saving / showing / sending to an AI. Raises PrivacyError (= refuse it).
    A picture that is already the output of this function is returned as it is."""
    import hashlib
    h = hashlib.sha256(img_bytes).hexdigest()
    if h in _done:
        return _done[h]
    png, n, found = mask_image(img_bytes)
    if len(_done) > 200:
        _done.clear()
    _done[h] = (png, n, found)
    _done[hashlib.sha256(png).hexdigest()] = (png, 0, [])
    return png, n, found


def check_image(img_bytes):
    """For the screens: (masked PNG, message for the agent). Raises PrivacyError to refuse."""
    if not getattr(config, "CARD_CHECK", True):
        return img_bytes, ""
    png, n, _ = safe_image(img_bytes)
    return png, (f"🔒 {n} card number area(s) hidden - check the picture before sending." if n
                 else "🔒 Checked - no card number found. Still crop out any card details yourself.")
