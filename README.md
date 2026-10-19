# Vidya v5.3 - Your AI Learning & Knowledge Assistant (NORAM · EUROPE · APAC)

Vidya is the team's AI knowledge assistant (formerly iKnow / OpsMate). Agents ask questions in
plain words (typos and Hindi/Kannada/Telugu in English letters are fine) or paste a screenshot,
and get one short answer with its SOP source and a confidence level. When Vidya is not sure it
says: "I do not have answer to this question, please consult with your team supervisor."

## What's new in Vidya v5.3 - GDS question, stricter learning, feedback report
- **"Which GDS?" every time:** a GDS question (commands, entries, PNR, TST, void, reissue, exchange, ...)
  without Amadeus / Sabre named always gets the buttons **Amadeus · Sabre** first - a learned answer, 👍 votes or
  a document covering both no longer skip it. Not asked when the GDS is clear from the words (Amadeus, 1A, Sabre,
  1S, a misspelling like "Amadus") or from a GDS-only command (TQT, FXP, TST = Amadeus; WPNC, WFR, PQ = Sabre),
  nor for meaning questions ("What is a PNR?"). Works with screenshots too. After the choice, the other GDS's
  passages and SME answers drop down, and an SME answer written for both shows only the chosen GDS's part.
  If the documents have nothing for the chosen GDS, Vidya says so and sends it to the SME (never the other GDS's
  command). Settings in config.py: `GDS_ALWAYS_ASK`, `GDS_OPTIONS`, `GDS_WORDS`, `GDS_COMMANDS`.
- **Learning from 👍 has a strict rule:** an answer is learned only with 👍 from **10 different agents** and
  **2 different SMEs**, and **80%+** of everyone who rated it said 👍 (each person's last vote counts once).
  Below the rule, votes change nothing in either direction (no boost, no penalty); 👎 still goes to the SME
  Desk. A learned answer that falls below 80% is switched off automatically. SMEs vote in
  **SME Desk → 👍 Answers to confirm**. Answers auto-learned under the old 2-vote rule are switched off (not
  deleted - see Learned Answers). Settings: `LEARN_AGENT_VOTES`, `LEARN_SME_VOTES`, `LEARN_MIN_POSITIVE_PCT`.
- **Questions & feedback report** (Admin → Dashboard, bottom): filter by 👎 / 👍 / no answer / all and by name;
  shows question, Vidya's answer, rating, comment, who asked; "Who rated what" lists every vote with name and
  role (agent / SME). **Download report (Excel)** gives both as sheets. Card numbers are already masked.
- Images are found in the `assets` folder or the main folder.

## What's new in Vidya v5.2 - new look and the Fare Rule Reader
- **New look:** the Vidya robot in the header (round icon), the welcome banner with the robot reading,
  tagline "Your AI Learning & Knowledge Assistant", the robot as chat avatar and browser-tab icon. No plane
  anywhere. Images: `assets/vidya_icon.png` (tab / avatar), `assets/vidya_icon.webp` (header),
  `assets/vidya_hero.webp` (banner). The old `assets/vidya_logo.png` is no longer used.
- **Fare Rule Reader:** the agent pastes the fare rule (category 16 PENALTIES - Amadeus FQN/TQN PE, Sabre RD
  *PN) into the question box, with or without a question ("change before departure fee?", "pax no show,
  refund?"). Vidya answers from THAT text, not the documents:
  - the fee that applies (e.g. **GBP 300.00** cancellation fee), or Non-refundable / Not permitted /
    Permitted - no fee in the rule; one case when the question says it, otherwise a table of all six cases
    (change / refund x before departure / after departure / no-show);
  - the conditions found in the rule: fare difference, residual value, per ticket / transaction / fare
    component, highest penalty or sum when fares are combined, taxes / YQ / YR, EMD collection, deadline,
    child / infant, waivers, exceptions (e.g. visa rejection);
  - the exact rule lines it used, and "Not clear from the rule text" when the rule does not say - never a
    guessed fee (those go to the SME Desk).
  A screenshot of the rule is also read (Tesseract, no AI needed) - every amount then carries "read from a
  screenshot - check the amount on screen", because text recognition can misread 0 as 8. Pasting the text
  is exact. Tested on 10 real Amadeus / Sabre fare rules (BA, LH, SQ, IB, DL ...): 51 of 51 cases right.
- **Three checks with every fare-penalty answer** (pasted rule or question about penalties / no-show /
  refundable / fare rules): Verify Penalties · Check Ticket Status · Check PNR history.
- Search: an FAQ is not used when the most specific word of the question (e.g. "merchant", "Germany")
  is missing from it.

## What's new in Vidya v5.1 - smarter answers, stricter card masking
- **Card numbers: 13 digits too.** Any run of 13 to 19 digits (spaces / hyphens allowed), whatever its
  first digits, becomes `*********1234` in the text box while it is typed or pasted, on the server
  before saving, in chats, announcements and screenshots. This now also hides 13-digit airline ticket
  numbers - on purpose. To keep ticket numbers readable set `MASK_MIN_DIGITS = 14` in config.py.
  Data saved before v5.1 (also a restored backup) is masked again with the new rule automatically.
- **Reranker model.** After word + meaning search, a small model reads the question together with each
  of the best passages and re-orders them (works with AI on or off). It only re-orders passages that
  already contain the question's key words, so it can never lift an unrelated passage. Loads in the
  background; until it is ready, or if it cannot load, ranking works as before. Admin -> Settings shows
  its status. Secrets: `RERANKER = "off"` to switch off; `RERANK_MODEL` to change the model.
- **Questions with several parts** ("what is the penalty for no show and how do I check the fare rule
  in Sabre") are split, each part is searched and answered on its own with its own source, and a part
  the documents do not cover is named "Not found in the documents" - never guessed - and goes to the
  SME Desk as "Partly answered". Comparisons ("difference between void, refund and exchange") and exact
  FAQ matches stay one answer.
- **AI answers are checked against the documents.** Every step carries its source number [1], [2]
  (matching the numbered Sources list). Every GDS command, number and code the AI writes must be in the
  document it cites - otherwise the step is marked ⚠️ "not in the documents - check before using", the
  confidence drops, and the question goes to the SME Desk. The AI also says what the documents do NOT
  cover ("Not covered by the documents: ...").
- **Which document wins.** Uploads have an optional **Effective from** date (also editable with ✏️ in
  Document Management). When two documents disagree: the newer effective date wins, then the stronger
  type (Policy > SOP > Process Knowledge > FAQ / Quick Guide > Other). The answer says which document it
  followed and the question goes to the SME Desk so the older document can be fixed. A real tie still
  shows the conflict message. Policy / SOP passages also rank slightly ahead of FAQs.

## What's new in Vidya v5.0
- **New name and logo:** Vidya, with the logo in `assets/` (header, sign-in page, chat avatar, browser
  tab) and the tagline "Ask Vidya. Learn Faster. Work Smarter."
- **Card numbers masked while typing (PCI):** any number of 14 to 19 digits, whatever its first digits,
  is turned into `************1234` INSIDE the text box as it is typed or pasted, before it is sent.
  Neither the agent's screen nor the SME's ever shows it. The server masks again on save (two locks).
  13-digit ticket numbers (also `176-2345678901-02`) stay readable. Screenshots use the same rule.
- **Announcements by SMEs:** SMEs and admins post updates of up to 250 words (menu Vidya ->
  Announcements). They run as a **red running line** across the top of every agent screen (hover to
  pause), for one LOB or All, until a chosen date or until hidden.
- **QBAY removed:** the LOBs are NORAM, EUROPE and APAC.
- **About 15 times faster answers:** the search index is built once and reused until a document or
  answer changes (same scores as before, tested). Less CPU, so the free Streamlit plan is throttled less.
- Kept from v4.3: backup AI order (Gemini -> second Gemini model -> Claude -> Ollama -> documents),
  exact-line look-up ("BLR airport"), SME Desk, live SME chat, learning from 👍 / 👎.

## What's new in v3 (upgraded from Compass)
- New look based on the mock-up: home screen, topic buttons, Recent Questions,
  Important Updates, answer card with confidence, sources with View.
- More accurate: spelling + Hinglish fix, coverage check, re-ranking, optional Gemini
  semantic search. Tested: 26/26 real questions right, 10/10 out-of-scope questions refused.
- Never guesses: low confidence -> the supervisor message (config.FALLBACK_TEXT).
- Conflict check: if two documents give different answers -> shows both + CONFLICT_TEXT.
- Learns by itself: 👍 from 2 people (or 1 admin) saves the answer; 👍/👎 move passages
  up/down for similar questions; 👎 switches off a wrong learned answer; new uploads
  re-check open gaps.
- Knowledge Gaps page: unanswered / conflicting / 👎 questions grouped, with status,
  owner, note, "write the answer now", Excel export.
- Dashboard: KPIs, by region, top topics, trend, most asked, unanswered, feedback,
  Export to Excel (5 sheets).
- Document Management (archive/activate), Announcements, Quick Guides, Settings.
- AI: Gemini (free tier available) or Claude - optional. Voice questions with Gemini.

## Files
| File | What it does |
|---|---|
| `app.py` | Staff screens + menu |
| `admin_pages.py` | Dashboard, Document Management, Review Queue, Knowledge Gaps, Learned Answers, Settings |
| `ui.py` | Colours, header bar, logo, red running line, live card-number guard |
| `assets/` | Vidya logo and icon |
| `knowledge.py` | Documents, search, 3-layer answering, self-learning, gaps, Excel export |
| `brain.py` | Spelling/Hinglish, topics, confidence, conflict check, short answers |
| `farerule.py` | Fare Rule Reader: reads a pasted category 16 and says which fee applies |
| `tests_data/` | 10 sample fare rules + expected answers (used by the tests) |
| `rerank.py` | Reranker model (second look at the best passages), loads in the background |
| `llm.py` | Connection to Gemini, Claude or local AI (Ollama) + built-in meaning search (all optional) |
| `LOCAL_AI_SETUP.md` | How to run AI with no key (Ollama + built-in meaning search) |
| `prompts.py` | Instructions given to the AI - edit to change answer style |
| `config.py` | All settings: name, standard replies, regions, topics, thresholds, AI keys |
| `.streamlit/config.toml` | Colours (blue theme) |
| `sample_docs/` | Demo documents loaded automatically |
| `demo_files/` | A document that deliberately conflicts with the FAQ - upload it to EUROPE to demo the conflict warning |

## Run on a laptop
Double-click `START_APP.bat` (installs libraries the first time), or:
`python -m pip install -r requirements.txt` then `python -m streamlit run app.py`

## Turn on AI (optional)
Streamlit Cloud → App → Settings → Secrets:
```
ADMIN_PASSWORD = "your-admin-password"
GEMINI_API_KEY = "your-gemini-key"
```
Laptop: `setx GEMINI_API_KEY "your-gemini-key"` once, then restart.
Then Settings → "Test AI connection" and "Build semantic index now".
Gemini free tier: Google may use what you send to improve its products - use sample /
non-confidential SOPs on the free tier.

## SME Desk (v4.2) - an expert answers what Vidya could not
* Goes to the **SME Desk**: no answer found, documents disagree, confidence **below 70 %**
  (`SME_REVIEW_BELOW_PCT` in config.py), or an agent marked 👎. The same question asked many
  times shows once ("Asked 5×").
* The SME signs in as **SME** (password `SME_PASSWORD` in Secrets, one line; until it is set SMEs
  use the admin password) and for each question: **Save SME answer** · **Vidya's answer is
  right** · **SOP update needed** (goes to Admin → Knowledge Gaps) · **Not a process question**.
* The answer is saved as a verified answer together with the agents' own wording. Vidya
  searches it first next time (also for similar wording) and shows it as "SME / process team
  answer". Uploaded SOPs and documents are never changed.
* Tab "Answers given by SMEs": search, edit when the process changes, switch off.
* Agents see "sent to an SME" under a no-answer or a below-70 % answer.

## Live chat with an SME (v4.2)
* Agents: menu **Vidya → Chat with SME**, or the button **"Chat live with an SME"** under a
  no-answer / below-70 % answer (the question is filled in). Shows how many SMEs are online.
* SMEs: **SME Desk → Live chat** - switch "I'm available" on, new chats pop up by themselves
  (checked every few seconds), **Pick up** (only one SME gets each chat), reply, and look things up
  in the SOPs on the same screen. SMEs also see "agents waiting" on their home screen.
* When the chat ends the SME ticks the question/answer pairs worth keeping and corrects the
  wording → saved as verified answers that Vidya uses first next time. Greetings / "ok thanks"
  are left out automatically.
* Every chat is kept in full: **SME Desk → Chat history** (search, export to Excel). If nobody
  answered, the question goes to the SME Desk queue so it is not lost.
* Works on Streamlit Cloud and company servers with no extra service (pages refresh every 3-4 s).
  Chats are in the database, so include them in the Settings backup on Streamlit Cloud.

## Card numbers are never stored or shown (PCI)
* Typed text (questions, 👎 comments, live chat, SME answers, announcements): ANY number of 14-19
  digits becomes `************1234` in the text box while it is typed or pasted (v5.0), and again on
  the server before anything is saved, shown to an SME or sent to an AI. CVV / security codes become
  `***`. Airline ticket numbers (13 digits), phone numbers and PNRs stay readable. A phone number
  written with a long international prefix (14+ digits) is also hidden - that is on purpose.
* Screenshots (Ask page and live chat): read on this server with **Tesseract OCR** and every card
  number is covered with a grey box. The agent sees this masked preview before sending. The original
  picture is never written to disk. If Tesseract is missing, screenshots are **refused**.
* SOP uploads: a file with a **real** card number is refused (nothing kept, not even the file) and the
  message says where it is. Public test numbers such as 4111 1111 1111 1111 are allowed and shown
  masked (`FPCCVI************1111/1229`).
* Data saved before this version is masked once automatically at start-up.
* Install Tesseract:
  - Streamlit Cloud: `packages.txt` (contains `tesseract-ocr`) must be in the GitHub repo - done
    automatically on reboot.
  - Linux server: `sudo apt install tesseract-ocr`
  - Windows laptop: installer linked from the official Tesseract page
    https://tesseract-ocr.github.io/tessdoc/Installation.html (UB Mannheim build), then in Secrets
    `TESSERACT_CMD = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"` (one line).
  - Check: Admin → Settings shows "Card numbers are hidden in text and screenshots".
* Limits: very blurry, rotated or handwritten numbers and photos of a physical card may not be read -
  agents should still crop card details out. Settings → backup files contain only masked data.

## AI with NO key (v4.2) - local AI and built-in meaning search
Full steps: **LOCAL_AI_SETUP.md**. Short version:

| Setting (Secrets or environment) | What it does |
|---|---|
| `AI_PROVIDER = "local"` | answers written by a free AI model on the same laptop/server (Ollama). No key, nothing leaves the network |
| `MEANING_SEARCH = "builtin"` | meaning search with a small model inside Vidya (fastembed). No key, no quota, also works on Streamlit Cloud |
| `LOCAL_MODEL`, `LOCAL_EMBED_MODEL`, `LOCAL_VISION_MODEL`, `OLLAMA_URL` | optional, see LOCAL_AI_SETUP.md |

Local AI does **not** run on Streamlit Community Cloud (too little memory) - run Vidya and
Ollama together on a laptop or company server. If the local AI is closed, Vidya answers from
the documents as before.

## What's new in v4
- Menu bar across the top of every page (Vidya / Admin) and an always-visible **Sign out** button and
  region selector - nothing depends on the side panel any more.
- **LOB tabs** above the chat: NORAM · EUROPE · APAC, each with its sub-LOB chips
  (e.g. NORAM: OBT Cancellation, Schedule Change, Ticketing, Refund, Profile Management, Others).
  Questions search the selected LOB (plus GLOBAL documents); the chosen sub-LOB's documents rank first.
  If the LOB has no answer, Vidya searches the other LOBs and says clearly which LOB the answer came from.
  Admins edit sub-LOBs in Settings and tag documents on upload or with ✏️ in Document Management.
- **Confidence %** on every answer (High 80-98 %, Medium 60-79 %), based on match strength, how much of the
  question is covered, and how many documents agree. Hover the ⓘ next to it for the explanation.
- How-to answers come as numbered steps in the order of the SOP.
- **Backup & restore** in Settings (free hosting can erase uploads on restart).
- Warning while the demo admin password is still in use - set ADMIN_PASSWORD in Secrets.

## What's new in v4.1 - pictures, GDS screens and "which one?" questions
- **Pictures in answers**: pictures inside uploaded Word / PDF / PowerPoint files (e.g. a sample Amadeus screen)
  are kept and shown under the answer that comes from the same section. Works without the AI key.
  Documents uploaded before v4.1 get their pictures automatically the first time the app starts.
  (Finding answers *inside* a picture still needs `GEMINI_API_KEY`.)
- **GDS entries look like a terminal**: commands in answers (RTABC123, FXP, TTP/RT, WETR*1, *T …) are shown
  green-on-dark, with a box listing every entry used in the answer.
- **Vidya asks before guessing**: if the steps differ by GDS (Amadeus / Sabre / Galileo …) and the agent did not
  say which, or the best answers come from different sub-LOBs and no sub-LOB is selected, Vidya asks with
  buttons, then answers. "Search all" answers without choosing. Settings: `ASK_CLARIFY`, `CLARIFY_GROUPS` in config.py.
- **Answers even when the AI is out of quota**: if the AI fails, is out of quota or replies "not found" while the
  documents clearly answer the question (strict word-match rule), Vidya shows the document answer with a note.
  After a quota error the AI is skipped for 5 minutes (`AI_PAUSE_SECONDS`), so answers come at once.
- **Home screen**: "Try asking" now follows the selected LOB tab and sub-LOB chip (most-asked questions, then
  questions and section titles from that LOB's documents). The 10 topic buttons were removed (they looked like
  filters but did not change the search). The welcome banner is half the height.
- **Follow-up questions**: "how about Sabre?" after "How to reissue a ticket in Amadeus?" is understood as
  "How to reissue a ticket in Sabre?" (shown under the answer as "follow-up, understood as …").
