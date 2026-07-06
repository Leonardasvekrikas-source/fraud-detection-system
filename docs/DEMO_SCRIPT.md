# 2-minute walkthrough — narration script

A read-aloud script for a short, recruiter-friendly video: show the system working, then the
engineering behind it. Record with [Loom](https://www.loom.com/) (easiest — screen + webcam +
instant shareable link) or OBS. Then paste the link into the README's walkthrough slot.

**You can read the grey lines almost word-for-word.** Tweak the wording so it sounds like *you* —
it's your project, so first person, plain, honest (no buzzwords). Aim for ~2 minutes; a way to cut
it to ~75 seconds is noted at the end.

### Setup (2 minutes, before you hit record)
- Open two browser tabs: **(1)** the [live demo](https://leonardasvekrikas-source-fraud-detection-demo.hf.space),
  **(2)** the [GitHub repo](https://github.com/Leonardasvekrikas-source/fraud-detection-system) README.
- On the demo tab: browser zoom ~110%, close other tabs, silence notifications.
- Do one silent practice run of the click order below so the demo feels smooth.
- Turn the Loom **webcam bubble on** — a face makes it personable; recruiters like seeing you.

> Legend below: **▶ DO** = what to click/show. The **quoted grey line** = what to say.

---

### 0:00–0:15 · Hook
**▶ DO:** Start on the demo page (result area empty, the three buttons visible).

> "Hi, I'm **[your name]**. This is a real-time credit-card fraud detector I built and deployed. What
> makes it interesting is that it runs **two** different models and explains **both** of them —
> let me show you."

### 0:15–0:50 · Score a fraud (the money shot)
**▶ DO:** Click **🔴 Random Fraud**. Let the result appear; point at the P1/P2 line, then each panel.

> "I'll score a real fraudulent transaction. Two models look at it — a **LightGBM** gives this
> probability, an **LSTM** gives this one — and they're fused into one verdict: **Fraud**. The
> important part is that each model explains itself. On the left, **SHAP** shows the LightGBM's top
> drivers; on the right, **LIME** explains the black-box LSTM — and they land on the *same* features.
> It's never just a score."

**▶ DO:** Point at a couple of the named feature rows / hover one for the tooltip.

> "And about these names — the features are anonymized, they're PCA components, so their real meaning
> is unknown. Rather than invent names, I labelled each with how much the model actually *relies* on
> it and which direction points to fraud. The names in quotes are clearly marked as illustrative."

### 0:50–1:10 · The clever bit — Expert-Checking
**▶ DO:** Click **🟡 Random Expert-Checking**. Point at the two disagreeing scores and the note.

> "This is the part I'm proudest of. Sometimes the two models **disagree** — here the LightGBM flags
> fraud, but the LSTM isn't convinced. Instead of forcing a call, the system returns
> **Expert-Checking** — send it to a human. That human-in-the-loop tier is built into the fusion
> rule itself, not bolted on afterward."

### 1:10–1:20 · Contrast — Normal
**▶ DO:** Click **🟢 Random Normal**.

> "A normal transaction comes back **Normal**, both models low. So it discriminates — and explains
> itself — either way."

### 1:20–1:50 · The engineering
**▶ DO:** Switch to the GitHub README tab. Slowly scroll the architecture diagram, results table, CI badge.

> "Behind the demo: this started as an independent reproduction of a 2025 research paper — I
> reproduced the LightGBM results **bit-for-bit** — and I turned it into a tested, CI-checked,
> Dockerised package. I deliberately chose **decision-level fusion** — keeping the two models
> separate — over merging them, because it stays simpler, faster, and explainable. And I backed that
> choice with tracked experiments, including an honest **negative result** where the more complex
> approach actually did worse."

### 1:50–2:00 · Close
**▶ DO:** Scroll to the monitoring / Airflow section, then stop.

> "And it doesn't stop at deployment — there's drift monitoring and an automated retraining pipeline
> that only promotes a new model if it beats the live one. It's all reproducible, and the code and
> live demo are linked below. Thanks for watching."

---

### Delivery tips (for anyone who hates recording — that's everyone)
- **Read, don't memorise.** Glancing at these lines is fine; nobody can tell.
- **Go ~10% slower** than feels natural — nerves speed you up.
- **Re-record freely.** Loom makes retakes one click; take three, keep the best.
- **Smile on the first line.** Energy carries more than perfection; small stumbles are human.
- **One breath between sections** — it reads as confidence, and makes editing easier.
- Do a throwaway take first to shake off the nerves; the second is always better.

### Want it shorter (~75s)?
Keep **Hook → Fraud → Expert-Checking → one-sentence close.** Drop the Normal contrast and the
engineering scroll — or save the engineering for a second, separate clip. Short and punchy beats
long and complete for a first impression.
