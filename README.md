---
title: Email Triage RL Environment
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_file: server.py
pinned: false
---

# 📧 Email Triage RL Environment v3

> A competition-grade OpenEnv reinforcement learning environment where AI agents learn to triage enterprise emails with **persistent state, long-term consequences, and adversarial difficulty** — modelled on the Enron email corpus.

---

## 🚀 Quick Start

```bash
python3 -m venv venv
source venv/bin/activate  
pip install --upgrade pip
pip install -r requirements.txt
python server.py  


# 📧 Email Triage RL Environment v3

> A **competition-grade OpenEnv reinforcement learning environment** where AI agents learn to triage enterprise emails with persistent state, long-term consequences, and adversarial difficulty — modelled on the Enron email corpus.

----------

## Why Email Triage?

Enterprise email triage is one of the highest-value automation targets in the industry:

- **Knowledge workers spend 28% of their workweek managing email** (McKinsey Global Institute)
- **Fortune 500 companies lose $1.8 trillion/year** to communication inefficiency (Loom, 2023)
- **Misclassified urgent emails cause real harm**: missed regulatory deadlines, ignored safety alerts, unanswered executive requests
- **A trained triage agent** reduces support ticket resolution time by 40–60% (Zendesk benchmark)

This environment teaches agents to handle the full complexity of real enterprise communication: ambiguous urgency, adversarial phishing that mimics legitimate email, multi-intent messages, and long-term consequences of poor decisions.

---

## Environment Overview

```
EmailTriageEnv
├── Tasks:   easy / medium / hard
├── Corpus:  180+ emails (Enron-style, 8 categories)
├── State:   inbox_health, user_satisfaction, consecutive_errors (persistent)
├── Rewards: partial scoring on classification + priority + action + response
└── Dynamics: wrong actions inject follow-up emails (angry escalations, chasers)
```

### What Makes This Hard

- **Adversarial phishing** that mimics legitimate Enron HR/IT/Payroll
- **Downplayed urgency**: "No rush, but the CEO needs this by close of business"
- **Multi-intent emails**: casual chit-chat that embeds a time-sensitive invoice
- **Insider information signals** that require legal escalation regardless of tone
- **Persistent state**: repeated mistakes compound (error streak tax, health decay)
- **Follow-up injection**: ignoring an urgent email causes an angrier one to appear two steps later

---

## Action & Observation Spaces

### Observation (`EmailObservation`)

| Field | Type | Description |
|-------|------|-------------|
| `email_id` | str | Unique email identifier |
| `subject` | str | Email subject line |
| `sender` | str | Sender address |
| `body` | str | Full email body |
| `thread_history` | List[ThreadMessage] | Prior messages in thread |
| `has_attachments` | bool | Whether attachments are present |
| `episode_state` | EpisodeState | Persistent state (health, errors, cost) |
| `is_followup` | bool | True if triggered by a previous bad action |
| `followup_reason` | str | Why this follow-up was injected |

### EpisodeState (persistent across steps)

| Field | Type | Description |
|-------|------|-------------|
| `inbox_health` | float [0,1] | Degrades on wrong classifications |
| `user_satisfaction` | float [0,1] | Drops on poor responses |
| `consecutive_errors` | int | Current error streak (triggers streak tax) |
| `unresolved_threads` | List[str] | Emails deferred/ignored needing follow-up |
| `total_cost` | float | Cumulative action cost this episode |
| `ignored_urgent` | int | Urgent emails wrongly ignored/deferred |

### Action (`AgentAction`)

| Field | Type | Values |
|-------|------|--------|
| `label` | EmailLabel | spam, work, finance, promotions, personal, urgent, newsletter, support |
| `priority` | PriorityLevel | low, medium, high, urgent |
| `action_type` | ActionType | respond, archive, escalate, ignore, defer, request_clarification |
| `response_text` | str \| None | Required for respond/escalate/request_clarification (≥5 words) |
| `confidence` | float [0,1] | Agent's stated confidence |

### Action Cost Model

| Action | Cost | Rationale |
|--------|------|-----------|
| escalate | 0.050 | Most expensive — avoid unnecessary escalation |
| request_clarification | 0.015 | Costs time, but smart when truly ambiguous |
| respond | 0.010 | Standard response effort |
| defer | 0.005 | Low cost, but triggers follow-ups |
| archive | 0.002 | Near-free |
| ignore | 0.000 | Free, but severely penalised on non-spam |

---

## Tasks

### 🟢 Easy — Spam vs. Legitimate (15 emails)

**Objective**: Binary classification: is this email spam or legitimate?

**Grader**: Classification score only (weight 1.0)
- Exact match → 1.0
- Fuzzy/typo match → 0.7
- Adjacent category → 0.5
- Wrong → 0.0

**Challenge**: Spear-phishing emails that spoof Enron HR/IT/Payroll domains with subtle typos (`enronn.com`, `enr0n-corp.com`).

**Expected baseline score**: 0.75–0.90

---

### 🟡 Medium — Multi-class + Priority (20 emails)

**Objective**: Classify into 8 categories AND assign correct priority level.

**Grader**: Classification (60%) + Priority (40%)

**Challenge**: 
- Emails where casual language masks urgency
- Finance emails that could be confused with work/support
- Persistent state: consecutive errors trigger streak penalties

**Expected baseline score**: 0.55–0.75

---

### 🔴 Hard — Full Triage Pipeline — Adversarial (25–30 emails)

**Objective**: Full pipeline — classify → prioritise → action → generate response.

**Grader**: Classification (40%) + Priority (20%) + Action (20%) + Response Quality (20%)

**Challenge**:
- ≥40% adversarial emails (downplayed urgency, insider info, multi-intent, sarcasm)
- Wrong actions trigger follow-up emails (injected 2 steps later)
- Persistent `inbox_health` and `user_satisfaction` evolve across steps
- Response quality graded via TF-IDF cosine similarity + keyword coverage
- Hard-email penalty scaling (×1.20 on mistakes)

**Expected baseline score**: 0.35–0.60

---

## Reward Function

```
total_reward = base_score + bonuses - penalties - action_cost
```

Clipped to `[-1.0, 1.0]`.

### Base Score (task-weighted)
| Task | cls | pri | action | response |
|------|-----|-----|--------|----------|
| easy | 1.0 | — | — | — |
| medium | 0.6 | 0.4 | — | — |
| hard | 0.4 | 0.2 | 0.2 | 0.2 |

### Penalties
| Trigger | Penalty |
|---------|---------|
| Ignored urgent email | −0.45 |
| Escalated junk email | −0.30 |
| Deferred urgent email | −0.30 |
| Empty/trivial response | −0.25 |
| Misclassified high-stakes | −0.20 |
| Archived finance/respond | −0.15 |
| Error streak (×consecutive) | −0.08 per step |
| Overconfident wrong answer | −0.05 |

### Bonuses
| Trigger | Bonus |
|---------|-------|
| Correctly handled adversarial email | +0.10 |
| Calibrated confidence (0.70–0.92) | +0.05 |
| Efficient high-quality response (20–80 words) | +0.05 |
| Good inbox health + excellent decision | +0.02 |

### Long-term Dynamics
- **inbox_health** decays on wrong classifications and bad responses; recovers on perfect decisions
- **user_satisfaction** drops on poor responses; recovers on excellent ones
- **Follow-up injection**: ignoring urgent → angry exec follow-up at step+2
- **Finance chasers**: inadequate response → overdue invoice chaser
- **Deferred threads**: resurface at higher priority 2 steps later

---

## Episode Result

```python
result = env.episode_result()
result.final_score      # float [0,1] — health-modulated mean reward
result.grader_breakdown # avg classification, priority, action, response scores
result.final_state      # inbox_health, user_satisfaction, etc.
```

**Final score formula**:
```
final_score = clip(avg_reward × (0.85 + 0.15 × inbox_health), 0, 1)
```

---

## Setup & Usage

### Local Development

```bash
# 1. Clone and set up
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: set HF_TOKEN, API_BASE_URL, MODEL_NAME

# 3. Run smoke tests
PYTHONPATH=. python smoke_test.py

# 4. Start the server
python server.py
# → http://localhost:7860/docs

# 5. Run baseline inference
python inference.py --task all --seed 42
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_... \
  -e API_BASE_URL=https://api-inference.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  email-triage-env
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | ✅ | Hugging Face / API key for LLM calls |
| `API_BASE_URL` | ✅ | LLM endpoint (OpenAI-compatible) |
| `MODEL_NAME` | ✅ | Model identifier |
| `PORT` | ❌ | Server port (default: 7860) |

---

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/tasks` | GET | List available tasks |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit action, get reward |
| `/state/{session_id}` | GET | Current state snapshot |
| `/result/{session_id}` | GET | Episode result |
| `/docs` | GET | Swagger UI |

### Quick Test

```bash
# Health check
curl https://<your-space>.hf.space/health

# Start episode
curl -X POST https://<your-space>.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard", "seed": 42}'

# Submit action
curl -X POST https://<your-space>.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "hard_42",
    "action": {
      "label": "urgent",
      "priority": "urgent",
      "action_type": "escalate",
      "response_text": "Escalating immediately to senior management and incident response.",
      "confidence": 0.92
    }
  }'
```

---

## Baseline Scores

Measured with `gpt-4o-mini`, seed=42, temperature=0:

| Task | Score | Classification | Priority | Action | Response |
|------|-------|---------------|----------|--------|----------|
| Easy | ~0.82 | 0.89 | — | — | — |
| Medium | ~0.64 | 0.71 | 0.68 | — | — |
| Hard | ~0.48 | 0.64 | 0.55 | 0.52 | 0.61 |
| **Average** | **~0.65** | | | | |

---

## Email Corpus

- **180+ emails** across 8 categories (60 seed + 2× paraphrase variants)
- **Modelled on Enron Email Corpus** (Klimt & Yang, 2004)
- **Adversarial patterns**: leet-speak phishing, spear-phishing (domain spoofing), advance-fee fraud, credential harvesting
- **Ambiguity patterns**: downplayed urgency, multi-intent, insider information
- **Thread history**: 12+ emails with realistic reply chains

### Category Distribution
| Label | Count | Notes |
|-------|-------|-------|
| spam | 36 | Including adversarial spear-phishing |
| promotions | 12 | Amazon, retailer promos |
| newsletter | 24 | HR, industry newsletters |
| work | 30 | Normal work communication |
| urgent | 24 | Including disguised urgency |
| finance | 24 | Invoices, payroll, budget |
| support | 18 | IT helpdesk requests |
| personal | 12 | Social / non-work |

---

## Agent Architecture (Provided Baseline)

The included `inference.py` implements a full LLM-based agent:

```
Email → LLM (deterministic, temperature=0)
             ↓
        Heuristic fallback (if LLM fails)
             ↓
        Validated AgentAction
             ↓
        Environment step()
```

For the full RL training loop (with memory + strategy layer), see `inference.py --mode train` in the extended version, which uses:
- **EpisodeMemory**: persists mistakes across episodes, injects correction hints
- **StrategyLayer**: risk-aware overrides (clarification, priority downgrade, escalation)
- **PolicyTrainer**: curriculum learning easy→medium→hard

---

## Failure Cases & Agent Weaknesses

Known hard cases that challenge frontier models:

1. **Domain typo spam** (`enronn.com` vs `enron.com`) — GPT-4o-mini fails ~15% of the time
2. **Downplayed urgency** — "No rush, but CEO needs this by 5:30" — classified as `work/medium` instead of `work/urgent`
3. **Multi-intent with finance** — email that starts as social and embeds an invoice
4. **Insider information escalation** — casual "between us" tone masking MNPI requiring legal escalation
5. **Spear-phishing that cites real Enron events** — stock option vesting, merger news

---

## Project Structure

```
.
├── server.py          # FastAPI REST server (OpenEnv interface)
├── inference.py       # Baseline LLM agent (competition entry point)
├── openenv.yaml       # OpenEnv metadata
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container definition
│
├── env/
│   ├── core.py        # EmailTriageEnv (reset/step/state)
│   └── models.py      # Pydantic models (AgentAction, EmailObservation, etc.)
│
├── tasks/
│   └── manager.py     # TaskConfig + TaskManager
│
├── graders/
│   └── engine.py      # Deterministic graders (classification/priority/action/response)
│
├── rewards/
│   └── engine.py      # Reward shaping with penalties + bonuses
│
├── data/
│   └── dataset.py     # 180+ email corpus + dataset accessor
│
└── agent/             # Optional: full LLM agent with memory + strategy
    ├── policy.py
    ├── memory.py
    ├── strategy.py
    └── trainer.py
```

---

## License

MIT — see LICENSE file.

---

## Citation

If you use this environment in your research:

```bibtex
@misc{email-triage-rl-env,
  title  = {Email Triage RL Environment: An OpenEnv benchmark for enterprise communication},
  year   = {2025},
  note   = {Modelled on Enron Email Corpus (Klimt \& Yang, 2004)},
  url    = {https://huggingface.co/spaces/...}
}
```