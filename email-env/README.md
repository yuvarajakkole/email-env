# 📧 Email Triage RL Environment v3

> A competition-grade OpenEnv reinforcement learning environment where AI agents learn to triage enterprise emails with **persistent state, long-term consequences, and adversarial difficulty** — modelled on the Enron email corpus.

---
python3 -m venv venv
source venv/bin/activate  
pip install --upgrade pip
pip install -r requirements.txt
python server.py  

## 🎯 Real-World Motivation

Knowledge workers spend 28% of their week managing email (McKinsey Global Institute). Poor triage decisions compound: an ignored urgent email becomes an executive escalation; an inadequate response to an invoice triggers a chaser; a deferred thread resurfaces at higher stakes. This environment captures that **compounding nature of decisions** — a critical property missing from toy environments.

### RL Training Use Case
This environment is designed for training RL agents that:
- Learn multi-step decision-making under uncertainty (ambiguous senders, casual urgency)
- Discover long-horizon policies: good inbox health → bonus rewards → higher scores
- Generalise across email domains without memorising surface patterns
- Are robust to adversarial inputs (spear-phishing that mimics legitimate email)

---

## 🏗️ Architecture

```
email_triage_env/
├── env/
│   ├── core.py        ← OpenEnv: reset/step/state + long-term effects engine
│   └── models.py      ← Pydantic models: Observation (+ EpisodeState), Action, Reward
├── tasks/
│   └── manager.py     ← Task configs (easy/medium/hard), episode management
├── graders/
│   └── engine.py      ← Hybrid graders: exact + fuzzy + TF-IDF cosine semantic
├── rewards/
│   └── engine.py      ← Cost-aware shaped rewards with streak penalties
├── data/
│   └── dataset.py     ← 60 seed emails × 3 = 180+ total (noise-injected variants)
├── server.py          ← FastAPI REST server
├── inference.py       ← LLM agent with CoT, state awareness, retry
└── openenv.yaml       ← OpenEnv specification
```

---

## 👁️ Observation Space

Each observation contains the email **plus the full persistent episode state**:

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | str | Unique identifier |
| `subject` | str | Subject line |
| `sender` | str | Sender email address |
| `recipient` | str | Recipient address |
| `timestamp` | str | ISO-8601 datetime |
| `body` | str | Full body text |
| `thread_history` | List[ThreadMessage] | Prior thread messages |
| `has_attachments` | bool | Attachment indicator |
| `episode_state.inbox_health` | float 0–1 | Degrades on bad decisions |
| `episode_state.user_satisfaction` | float 0–1 | Drops on poor responses |
| `episode_state.unresolved_threads` | List[str] | Deferred/ignored email IDs |
| `episode_state.consecutive_errors` | int | Current misclassification streak |
| `episode_state.total_cost` | float | Cumulative action cost |
| `is_followup` | bool | True if triggered by a previous bad action |

---

## ⚙️ Action Space (Extended)

| Action | Cost | Description |
|--------|------|-------------|
| `respond` | 0.010 | Reply to sender (requires response_text ≥ 25 words) |
| `archive` | 0.002 | File away, no reply (FYI emails, newsletters) |
| `escalate` | 0.050 | Hand to senior management (requires response_text) |
| `ignore` | 0.000 | Discard (spam only) |
| `defer` | 0.005 | Snooze for later (**triggers angry follow-up** if non-trivial) |
| `request_clarification` | 0.015 | Ask sender a question before acting |

---

## 🔁 Long-Term Effects (Unique Feature)

| Bad Action | Consequence |
|-----------|-------------|
| Ignored urgent email | `inbox_health -= 0.15`, angry escalation email injected 2 steps later |
| Poor response to finance | `user_satisfaction -= 0.08`, payment chaser email injected |
| Deferred non-trivial email | Email resurfaces with `priority=urgent` + deadline-passed framing |
| Consecutive errors (≥2) | Streak penalty applied each step: `-0.08 × min(streak, 5)` |
| Misclassified high-stakes | `-0.20` penalty + `inbox_health -= 0.06` |

**Final score** is modulated by `inbox_health`: `score × (0.85 + 0.15 × inbox_health)`

---

## 🧮 Grader Design (spec-exact weights)

| Component | Weight (Hard) | Scoring |
|-----------|--------------|---------|
| Classification | 0.40 | Exact(1.0) → Fuzzy/typo(0.7) → Adjacent(0.5) → Substring(0.4) → 0.0 |
| Priority | 0.20 | Exact(1.0) → Off-by-1(0.5) → Off-by-2(0.25) + urgency context bonus |
| Action | 0.20 | Exact(1.0) → Acceptable alt(0.4–0.6) → Harmful(-0.25 to -0.5) |
| Response | 0.20 | TF-IDF cosine(×0.4) + keyword coverage(×0.4) + length(×0.2) |

---

## 🧪 Tasks

### 🟢 Easy — Spam Classification
- 15 emails/episode, classification only
- Tests: obvious spam, leet-speak phishing, spear-phishing, chain mail

### 🟡 Medium — Classification + Priority  
- 20 emails/episode, classification + priority graded
- Tests: ambiguous urgency, multi-category signals, casual-tone high-stakes

### 🔴 Hard — Full Adversarial Pipeline
- 25 emails + up to 5 injected follow-ups per episode
- ≥40% annotated `difficulty=hard` emails
- Tests: insider information, downplayed CEO deadlines, multi-intent, sarcasm
- Requires: reasoning through context, not pattern matching

---

## 🌍 Dataset

- **180+ emails** (60 seed × 3 with noise injection)
- Noise types: typo injection (swap/drop/double chars), salutation variation, sentence reordering
- 8 categories: spam, work, finance, promotions, personal, urgent, newsletter, support
- 12 emails with multi-message thread history
- 15 adversarial hard emails with conflicting urgency signals

---

## 🚀 Setup & Run

```bash
# Install
pip install -r requirements.txt

# Run server
PYTHONPATH=. python server.py

# Run inference (all tasks)
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
PYTHONPATH=. python inference.py --task all --seed 42

# Single task
PYTHONPATH=. python inference.py --task hard --seed 42
```

### Docker
```bash
docker build -t email-triage-env .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... email-triage-env
```

---

## 📊 Baseline Results (gpt-4o-mini, seed=42)

| Task | Score | Classification | Priority | Action | Response |
|------|-------|---------------|----------|--------|----------|
| Easy | 0.91 | 0.91 | — | — | — |
| Medium | 0.76 | 0.82 | 0.67 | — | — |
| Hard | 0.68 | 0.74 | 0.62 | 0.70 | 0.62 |
| **Average** | **0.78** | — | — | — | — |

*Hard task final score includes inbox_health modifier.*

---

## 📄 License

MIT License
