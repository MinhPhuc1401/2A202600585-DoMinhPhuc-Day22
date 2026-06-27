# Bonus Challenge — Build something real (UNGRADED)

> Vietnamese: [`BONUS-CHALLENGE.md`](BONUS-CHALLENGE.md)

**Type:** A space to **bring your own domain knowledge into a real aligned model** — no points, no deadline, no rubric.
**Audience for you:** You play *AI engineer* shipping an aligned model for a *specific audience*, like a product.
**Effort target:** 4–8 hours. Pair work (2–3 people) encouraged. Brainstorm first, code second.
**Vibe coding encouraged:** AI handles boilerplate, *domain choices + application objective* you write yourself.

> This is where you stop treating DPO as a paper exercise and start treating it as **a tool to ship something someone uses**. The real reward: a portfolio piece you can point at and say "I built this, the audience is X, it does Y" — not "I tuned β=0.05."

---

## 5 provocations — pick one, or invent your own

Each provocation has 4 axes: **Audience** (who uses it) · **Domain knowledge** (what you bring) · **Application objective** (what the model does) · **Real-world output** (a deliverable that ships).

### 1. Subject tutor for a course you're taking

> *"DPO on scaffolded pedagogy, not on direct answers."*

- **Audience:** High-school or first-year university students reviewing one specific subject you're confident in (calculus, organic chemistry, Vietnamese history, Python programming, electromagnetism).
- **Domain knowledge:** You know the subject well enough to distinguish "good pedagogical response" from "directly giving the answer." That's the core requirement.
- **Application objective:** A tutor model that *doesn't* give the answer — it hints, asks counter-questions, references formulas from the Vietnamese textbook standard, avoids English jargon students don't know.
- **Real-world output:**
  - 200 preference pairs (prompt = typical student question; chosen = scaffolded VN tutoring response; rejected = direct answer in mixed VN/English)
  - DPO adapter + GGUF Q4_K_M that runs on a student's laptop CPU (`llama-cpp-python`)
  - Mini Gradio demo (~50 lines) so a student can actually try it
  - Model card: "Math tutor for grade 12 students preparing for the THPT national exam — does not replace the textbook"

**Brainstorming questions:**
- How does pedagogy for VN high school differ from US Khan Academy? (test structure, standard textbook vocabulary)
- When *should* the model give a direct answer (problem already solved, student wants to verify)? How does DPO learn that context?
- How do you measure "tutoring quality" without a real teacher grading?

---

### 2. Customer-service chatbot for an actual Vietnamese business

> *"Pick one shop. Imagine 200 customer questions. Build a chatbot you could actually deploy."*

- **Audience:** Customers and the owner of one specific business — a Hanoi cafe, an online clothing store, a motorbike repair shop, a small English school, a Sapa homestay. You pick.
- **Domain knowledge:** You know how that business operates — opening hours, services, rough pricing, the tone of voice they use (friendly vs polite vs youthful).
- **Application objective:** A chatbot that answers FAQs on-brand, gives a clear next step (phone, booking link, Google Maps address), and refuses to fabricate information it doesn't have.
- **Real-world output:**
  - 200 preference pairs (chosen = on-brand voice + contact CTA; rejected = generic dry AI reply, or English fallback)
  - DPO adapter + GGUF
  - Deployable: a 30-line `serve.py` using `llama-cpp-python` + a FastAPI `/chat` endpoint
  - Model card with "What I built this for: <business>" + 5 sample conversations

**Brainstorming questions:**
- Can you get 200 actual customer messages? (Does a friend run a shop with an FB inbox you can read?)
- Tone: formal `anh/chị` vs casual `bạn` vs youthful `shop/em` — can DPO learn the register?
- When customers ask outside scope (competitor pricing, negative reviews), what should it do? Refuse? Defuse? Hand off?

---

### 3. Job-shadow assistant — a model for a profession you observe daily

> *"Build for someone who isn't you. Build for a Grab driver, not for a VinUni student."*

- **Audience:** A specific Vietnamese working role — Grab/be driver, food delivery rider, hotel receptionist, wholesale market vendor, TikTok content creator, neighborhood electrician.
- **Domain knowledge:** You observe or interview someone in this role. The job has its own vocabulary, time pressures, and decision patterns that differ from desk work.
- **Application objective:** A model that answers questions *the worker actually faces*, not "questions students have about that job." Fast, action-oriented, no lectures, contextually aware of the job.
- **Real-world output:**
  - 200 preference pairs (prompt = real questions you observed; chosen = short, gives the next concrete action, understands context; rejected = ChatGPT's default 5-paragraph lecture)
  - DPO adapter + GGUF (designed to run on a mid-tier Android phone — the user may not own a laptop)
  - Mini demo: a CLI script you run sitting next to the actual worker testing 5–10 real prompts
  - Model card: "Who it's for" + "I shadowed someone in this role for X hours, here are 5 design decisions based on that"

**Brainstorming questions:**
- Response length: a driver who's actually driving can't read 200 words. How do you get DPO to enforce <50-word responses?
- Industry slang (delivery rider terms: "boom đơn" = customer no-show, etc.) — does the model know it? What data covers that?
- Can you *test with an actual person in the job*? If yes — that's the best evaluation, better than any GPT-4 judge.

---

### 4. Domain-safe assistant — an aligned model with explicit boundaries in a sensitive area

> *"The hardest balance: not refusing too much, not over-helping. You build alignment for one specific domain."*

- **Audience:** General Vietnamese users seeking information in a high-risk domain — mental health support, basic legal information (labor law, civil), basic medication information, personal financial planning, info-bot for vaccination or HIV prevention.
- **Domain knowledge:** You know (or research) what counts as *useful information* vs *advice that should go to a professional*. You know the official Vietnamese hotlines and resources (1900-XXXX numbers, Ministry of Health, Women's Hotline).
- **Application objective:** A model that:
  1. Provides background information + official sources
  2. Doesn't give personalized advice (prescribing, diagnosis, legal advice)
  3. Stays empathetic — doesn't push users away coldly
  4. Always refers to a professional + provides VN hotlines when needed
- **Real-world output:**
  - 200 preference pairs (chosen = info + sources + hotline + soft handoff; rejected = direct advice OR cold refusal)
  - DPO adapter + GGUF
  - **A special model card:** explicit "What this model will NOT do" list — this is the most important part. Document boundaries clearly.
  - 20-prompt test set: 10 benign-but-sensitive (model SHOULD answer with resources), 10 boundary-crossing (model MUST refuse + hand off). Report precision/recall on both.

**Brainstorming questions:**
- What is "empathetic refusal" in Vietnamese? How does it differ from US "I can't help with that"?
- Users may be in real distress — could DPO accidentally train the model into "robotic" mode? How do you prevent that?
- Reference: Vietnamese Psychology Association, the 1900-1567 hotline, or Anthropic's Claude constitutional AI model card.

---

### 5. Style mimic — a model that writes like someone you admire

> *"Pick one person. 50 samples. DPO style transfer."*

- **Audience:** Yourself (writing assistant), a brand you're a fan of (style ghostwriter), or an organization (Tuổi Trẻ newspaper, VTV24, a specific TikTok channel).
- **Domain knowledge:** You read enough of that person/brand's output to *feel* the style — long vs short sentences, heavy vs light Sino-Vietnamese vocabulary, humorous vs serious, emoji or none, single-sentence paragraphs vs five.
- **Application objective:** A model that writes Vietnamese in that style — not copying content, but *matching register*.
- **Real-world output:**
  - 50–100 samples of writing in the target style (essays, blogs, tweet threads, voice-memo transcripts)
  - 200 preference pairs (prompt = "write a paragraph about <topic> in style X"; chosen = style match; rejected = generic AI flat tone). *Chosen can be your own re-edits — that's the deepest domain knowledge you bring.*
  - DPO adapter
  - 5 side-by-side outputs: same prompt, before/after DPO, given to a fan of the style for *blind testing*
  - (Optional) HF Hub push with a model card naming the inspiration source + license/attribution

**Brainstorming questions:**
- How is style transfer different from content imitation? Can DPO separate them, or does it learn both?
- Ethics: if you mimic a living writer, is it homage or infringement? What does your model card say?
- Are 50 samples enough? Some research suggests style transfer needs less data than content learning — verify empirically.

---

## Or — invent your own

A good template for inventing your own provocation, hitting all 4 axes:

```
AUDIENCE:           who will use this model? (specific, not "everyone")
DOMAIN KNOWLEDGE:   what do you bring? (personal experience, skills,
                    relationships in the industry, data you have rights to)
APPLICATION OBJECT: what does the model DO for the audience? (one clear
                    use case, not "improve helpfulness in general")
REAL-WORLD OUTPUT:  the deliverable's shape (GGUF? Gradio? CLI? FastAPI?
                    HF Hub model card? integration into an existing app?)
```

Other angles worth exploring:

- **Accessibility-first:** model output friendly to screen readers, short sentences for low-literacy users, support for ESL learners studying Vietnamese
- **Cultural register:** train preference for Northern vs Southern VN, formal vs casual, context-appropriate register (chatting with a boss vs a friend)
- **Niche cultural archive:** Vietnamese poetry, dynasty history, regional cuisine, folklore — a model that doesn't hallucinate VN history
- **Code-review assistant for one framework:** FastAPI/React/Laravel idiomatic comments
- **Voice-to-action:** transcript → action items, e.g., meeting notes → todo list

Self-invented provocations usually produce deeper portfolio pieces than the 5 suggested ones — because *you* care, not because *I* care.

---

## Self-checklist for strong submissions

Strong submissions usually have:

- [ ] **`bonus/README.md` ≥ 400 words** answering all 4 axes: Audience / Domain knowledge / Application objective / Real-world output. The first page should make it clear "who uses this and what for."
- [ ] **Preference data from your domain**, not translated UltraFeedback. At least 100 of 200 pairs constructed by you with domain judgment.
- [ ] **A working deliverable**: someone (TA, classmate, outsider) can clone the repo + run one command + interact with the model.
- [ ] **A model card** with an explicit "What this model is for / NOT for / Known limitations" — especially crucial for provocation 4.
- [ ] **5 sample interactions** in the README to show quality directly, not just numbers.
- [ ] **Honest limitations**: a paragraph "What this POC doesn't handle yet" — privacy, bias, scale, licensing.
- [ ] **(Optional) HF Hub push** with full README — convention: include "v0" or "experimental" in the model name to avoid implying production-ready.

---

## Accepted format

Free-form. Suggested structure:

```
bonus/
├── README.md            # 1 page: Audience, Domain, Objective, Output
├── data/
│   ├── prompts.jsonl    # 200 prompts from your domain
│   └── pairs.parquet    # preference pairs
├── train.py (or .ipynb) # training run (re-use scripts/train_dpo.py with your data)
├── adapters/dpo-bonus/  # gitignored — adapter output
├── demo/
│   ├── serve.py         # one deployable file (FastAPI / Gradio / CLI)
│   └── 5-samples.md     # 5 prompts + before/after DPO outputs
└── MODEL-CARD.md        # ship-ready documentation
```

**Pair / triple work encouraged** — note contributors at the top of `bonus/README.md`.

**Vibe coding workflow log** — if you used AI heavily, write ~100 words in `MODEL-CARD.md`: "one prompt that worked, one that failed."

---

## Submission

Add a `bonus/` folder to your public repo (same repo as the main Lab 22). Mention in `submission/REFLECTION.md` that you did the bonus. The grader will review from the same public LMS URL.

> Bonus does **not** affect the core grade. A strong bonus submission earns substantive instructor commentary on your *judgment* + *application thinking* — not on β/loss/reward gap.

---

## "Everyone has a domain"

You don't have to be a full-time AI researcher to build something meaningful. You know one subject, you know one job, you know one community, you know one person's writing style. That's enough domain knowledge to make your aligned model *serve someone specific* rather than be the 1001st generic ChatBot clone.

Open a PR back to the upstream repo if you want to share with the next cohort.
