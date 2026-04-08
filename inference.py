"""
inference.py — Email Triage RL Environment Baseline Inference Script
----------------------------------------------------------------------
Environment variables (REQUIRED):
  API_BASE_URL   — LLM endpoint
  MODEL_NAME     — Model identifier
  HF_TOKEN       — API key / Hugging Face token
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from openai import OpenAI

try:
    from env.core import EmailTriageEnv
    from env.models import AgentAction, EpisodeState
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    from core import EmailTriageEnv
    from models import AgentAction, EpisodeState

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def _clamp(v: float) -> float:
    """Strictly between 0 and 1 — never 0.0 or 1.0."""
    return round(max(0.001, min(0.999, float(v))), 4)


_SYSTEM_PROMPT = """You are an enterprise email triage agent at Enron (2001).
Classify each email and decide the correct action. Be precise and deterministic.

LABEL OPTIONS:  spam | work | finance | promotions | personal | urgent | newsletter | support
PRIORITY OPTIONS: low | medium | high | urgent
ACTION OPTIONS: respond | archive | escalate | ignore | defer | request_clarification

RULES:
1. Sender domain typos (enr0n, paypa1, enronn.com, fedx-) -> spam -> archive
2. "No rush but CEO needs today" or "close of business" = URGENT priority
3. Legal / subpoena / SEC / breach / margin call -> urgent, escalate
4. Invoice / wire / payroll / budget -> finance
5. Newsletter / unsubscribe / digest -> newsletter, archive
6. response_text REQUIRED (>=20 words) when action is respond/escalate/request_clarification
7. response_text must be null when action is archive/ignore/defer
8. confidence must be between 0.1 and 0.9

Return ONLY valid JSON:
{
  "label": "<label>",
  "priority": "<priority>",
  "action_type": "<action>",
  "response_text": "<text or null>",
  "confidence": <0.1-0.9>
}"""

_TEMPLATES = {
    "work":     "Thank you for your message. I have reviewed the details and will action this before the deadline. I will follow up with a full update shortly.",
    "finance":  "Thank you — I have received this and forwarded it to accounts payable for processing. I will confirm once payment has been actioned.",
    "support":  "I have raised a support ticket and escalated your issue to the IT team for immediate resolution. I will keep you updated on progress.",
    "escalate": "Acknowledged — escalating immediately to senior management and the relevant response team. I will provide updates every 30 minutes until resolved.",
    "clarify":  "Thank you for your message. To ensure I take the correct action, could you please confirm the deadline and the primary contact I should coordinate with?",
}


def _heuristic(text: str) -> Dict[str, Any]:
    t = text.lower()
    if any(s in t for s in ["lottery","won $","nigerian","acc0unt","v3rify","paypa1","enr0n.","enronn.","fedx","routing number","bank details"]):
        return {"label":"spam","priority":"low","action_type":"archive","response_text":None,"confidence":0.9}
    if any(s in t for s in ["critical","emergency","subpoena","breach","server down","margin call","wire failed","pipeline rupture"]):
        return {"label":"urgent","priority":"urgent","action_type":"escalate","response_text":_TEMPLATES["escalate"],"confidence":0.85}
    if any(s in t for s in ["invoice","payment due","payroll","wire transfer","budget approval","expense report"]):
        return {"label":"finance","priority":"high","action_type":"respond","response_text":_TEMPLATES["finance"],"confidence":0.8}
    if any(s in t for s in ["vpn","helpdesk","cannot access","error 403","laptop","bloomberg terminal","workstation"]):
        return {"label":"support","priority":"medium","action_type":"respond","response_text":_TEMPLATES["support"],"confidence":0.75}
    if any(s in t for s in ["unsubscribe","newsletter","weekly digest","open enrollment","building closure","training module"]):
        return {"label":"newsletter","priority":"low","action_type":"archive","response_text":None,"confidence":0.85}
    if any(s in t for s in ["black friday","50% off","shipped","amazon order","pre-approved"]):
        return {"label":"promotions","priority":"low","action_type":"archive","response_text":None,"confidence":0.8}
    return {"label":"work","priority":"medium","action_type":"respond","response_text":_TEMPLATES["work"],"confidence":0.65}


def _call_llm(email_body: str, task_id: str) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"TASK: {task_id.upper()}\n\n{email_body}\n\nReturn JSON only."},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return _heuristic(email_body)


def _build_action(parsed: Dict[str, Any]) -> AgentAction:
    action_type   = parsed.get("action_type", "archive")
    response_text = parsed.get("response_text") or ""
    response_text = response_text.strip() if isinstance(response_text, str) else ""

    needs_response = action_type in ("respond", "escalate", "request_clarification")
    if needs_response and len(response_text.split()) < 5:
        label = parsed.get("label", "work")
        if action_type == "escalate":
            response_text = _TEMPLATES["escalate"]
        elif action_type == "request_clarification":
            response_text = _TEMPLATES["clarify"]
        elif label == "finance":
            response_text = _TEMPLATES["finance"]
        elif label == "support":
            response_text = _TEMPLATES["support"]
        else:
            response_text = _TEMPLATES["work"]

    if not needs_response:
        response_text = None

    conf = max(0.1, min(0.9, float(parsed.get("confidence", 0.75))))

    return AgentAction(
        label         = parsed.get("label",    "work"),
        priority      = parsed.get("priority", "medium"),
        action_type   = action_type,
        response_text = response_text or None,
        confidence    = conf,
    )


def run_episode(task_id: str, seed: int) -> Dict[str, Any]:
    env  = EmailTriageEnv(task_id=task_id, seed=seed)
    obs  = env.reset()
    done = False
    step_rewards: List[float] = []

    print(f"[START]")
    print(f"task_id={task_id} seed={seed}")
    print(f"[/START]")
    sys.stdout.flush()

    while not done and obs is not None:
        thread_ctx = ""
        if obs.thread_history:
            thread_ctx = "\n\nTHREAD:\n" + "\n---\n".join(
                f"From: {m.sender}\n{m.body[:200]}" for m in obs.thread_history
            )

        followup_note = ""
        if obs.is_followup:
            followup_note = f"\n\n[FOLLOW-UP: {obs.followup_reason}]"

        email_prompt = (
            f"From: {obs.sender}\n"
            f"Subject: {obs.subject}\n"
            f"Date: {obs.timestamp}\n\n"
            f"{obs.body}"
            f"{thread_ctx}"
            f"{followup_note}"
        )

        parsed = _call_llm(email_prompt, task_id)

        try:
            action = _build_action(parsed)
        except Exception:
            action = _build_action(_heuristic(email_prompt))

        try:
            obs, reward, done, info = env.step(action)
        except Exception:
            safe = AgentAction(
                label="work", priority="medium",
                action_type="respond",
                response_text=_TEMPLATES["work"],
                confidence=0.5
            )
            try:
                obs, reward, done, info = env.step(safe)
            except Exception:
                break

        r_val = _clamp(reward.total_reward)
        step_rewards.append(r_val)
        email_id = info.get("email_id", "unknown")

        print(f"[STEP]")
        print(f"email_id={email_id}")
        print(f"action={action.label.value}")
        print(f"priority={action.priority.value}")
        print(f"action_type={action.action_type.value}")
        print(f"reward={r_val}")
        print(f"[/STEP]")
        sys.stdout.flush()

    result = env.episode_result()
    final_score = _clamp(result.final_score)
    avg_reward  = _clamp(sum(step_rewards) / max(1, len(step_rewards)))

    print(f"[END]")
    print(f"task_id={task_id}")
    print(f"score={final_score}")
    print(f"[/END]")
    sys.stdout.flush()

    return {
        "task_id":     task_id,
        "seed":        seed,
        "final_score": final_score,
        "avg_reward":  avg_reward,
        "n_steps":     len(step_rewards),
        "breakdown":   result.grader_breakdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Email Triage RL — Baseline Inference")
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="inference_results.json")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    print(f"# Email Triage RL — Baseline Inference")
    print(f"# model={MODEL_NAME}  base_url={API_BASE_URL}  seed={args.seed}")
    print(f"# tasks={tasks}")
    sys.stdout.flush()

    all_results: Dict[str, Any] = {}

    for task_id in tasks:
        result = run_episode(task_id=task_id, seed=args.seed)
        all_results[task_id] = result

    scores = [r["final_score"] for r in all_results.values()]
    avg    = _clamp(sum(scores) / max(1, len(scores)))

    print(f"\n# SUMMARY")
    print(f"# model={MODEL_NAME}")
    for task_id, r in all_results.items():
        print(f"# {task_id.upper():<8} score={r['final_score']}  steps={r['n_steps']}")
    print(f"# AVERAGE  {avg}")

    output = {
        "model":    MODEL_NAME,
        "base_url": API_BASE_URL,
        "seed":     args.seed,
        "results":  all_results,
        "summary":  {
            "avg_score":   avg,
            "task_scores": {t: r["final_score"] for t, r in all_results.items()},
        },
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"# Results saved -> {args.output}")


if __name__ == "__main__":
    main()




# """
# inference.py — Email Triage RL Environment Baseline Inference Script
# ----------------------------------------------------------------------
# Runs a deterministic LLM agent against all three tasks (easy / medium / hard)
# using the OpenAI client pointed at any OpenAI-compatible endpoint.

# Environment variables (REQUIRED):
#   API_BASE_URL   — LLM endpoint, e.g. https://api-inference.huggingface.co/v1
#   MODEL_NAME     — Model identifier, e.g. meta-llama/Llama-3.3-70B-Instruct
#   HF_TOKEN       — API key / Hugging Face token

# Output format (machine-readable, required by competition evaluator):
#   [START] … [/START]   — episode metadata
#   [STEP]  … [/STEP]    — per-email decision + reward
#   [END]   … [/END]     — episode result

# Usage:
#   python inference.py                       # run all tasks
#   python inference.py --task hard           # single task
#   python inference.py --task all --seed 42  # explicit seed
# """

# from __future__ import annotations

# import argparse
# import json
# import os
# import sys
# import time
# from typing import Any, Dict, List, Optional, Tuple

# # ── OpenAI client (uses HF_TOKEN + API_BASE_URL) ──────────────────────────
# from openai import OpenAI

# # ── Environment imports (same directory) ──────────────────────────────────
# # Support both flat layout (competition zip) and package layout
# try:
#     from env.core import EmailTriageEnv
#     from env.models import AgentAction, EpisodeState
# except ModuleNotFoundError:
#     # flat layout: core.py / models.py at root
#     sys.path.insert(0, os.path.dirname(__file__))
#     from core import EmailTriageEnv          # type: ignore
#     from models import AgentAction, EpisodeState  # type: ignore


# # ─────────────────────────────────────────────────────────────────────────────
# # Configuration (strict env-var reading — no hardcoded keys)
# # ─────────────────────────────────────────────────────────────────────────────

# API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
# MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
# HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))

# if not HF_TOKEN:
#     print("[ERROR] HF_TOKEN (or OPENAI_API_KEY) environment variable is not set.", file=sys.stderr)
#     sys.exit(1)

# client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# # ─────────────────────────────────────────────────────────────────────────────
# # Deterministic LLM system prompt
# # ─────────────────────────────────────────────────────────────────────────────

# _SYSTEM_PROMPT = """You are an enterprise email triage agent at Enron (2001).
# Classify each email and decide the correct action. Be precise and deterministic.

# LABEL OPTIONS:  spam | work | finance | promotions | personal | urgent | newsletter | support
# PRIORITY OPTIONS: low | medium | high | urgent
# ACTION OPTIONS: respond | archive | escalate | ignore | defer | request_clarification

# RULES:
# 1. Sender domain typos (enr0n, paypa1, enronn.com, fedx-) → spam → archive
# 2. "No rush but CEO needs today" or "close of business" = URGENT priority
# 3. Legal / subpoena / SEC / breach / margin call → urgent, escalate
# 4. Invoice / wire / payroll / budget → finance
# 5. Newsletter / unsubscribe / digest → newsletter, archive
# 6. Multi-intent emails → classify by most consequential element
# 7. response_text is REQUIRED (≥20 words) when action is respond / escalate / request_clarification
# 8. response_text must be null when action is archive / ignore / defer

# Return ONLY valid JSON — no markdown, no extra text:
# {
#   "label": "<label>",
#   "priority": "<priority>",
#   "action_type": "<action>",
#   "response_text": "<text or null>",
#   "confidence": <0.0-1.0>
# }"""

# _RESPONSE_TEMPLATES = {
#     "respond_work":    "Thank you for your message. I have reviewed the details and will complete this before the deadline. I will follow up with a full update shortly.",
#     "respond_finance": "Thank you — I have received this and forwarded it to accounts payable for processing. I will confirm once payment has been actioned.",
#     "respond_support": "I have raised a support ticket and escalated your issue to the IT team for immediate resolution. I will keep you updated on progress.",
#     "escalate":        "Acknowledged — escalating immediately to senior management and the relevant response team. I will provide updates every 30 minutes until this is resolved.",
#     "request_clarification": "Thank you for your message. To ensure I take the correct action, could you please confirm the deadline and the primary contact I should coordinate with?",
# }


# # ─────────────────────────────────────────────────────────────────────────────
# # LLM call — deterministic (temperature=0, no retries for reproducibility)
# # ─────────────────────────────────────────────────────────────────────────────

# def _call_llm(email_body: str, task_id: str) -> Dict[str, Any]:
#     """Single deterministic LLM call. Returns parsed JSON dict."""
#     user_msg = f"TASK: {task_id.upper()}\n\n{email_body}\n\nReturn JSON only."
#     try:
#         resp = client.chat.completions.create(
#             model       = MODEL_NAME,
#             messages    = [
#                 {"role": "system", "content": _SYSTEM_PROMPT},
#                 {"role": "user",   "content": user_msg},
#             ],
#             temperature = 0.0,
#             max_tokens  = 300,
#         )
#         raw = resp.choices[0].message.content.strip()
#         # Strip markdown fences if present
#         if raw.startswith("```"):
#             raw = raw.split("```")[1]
#             if raw.startswith("json"):
#                 raw = raw[4:]
#         return json.loads(raw.strip())
#     except Exception as e:
#         # Hard fallback — heuristic only, no second LLM call
#         return _heuristic(email_body)


# def _heuristic(text: str) -> Dict[str, Any]:
#     """Pure rule-based fallback — 100% deterministic, no API calls."""
#     t = text.lower()

#     spam_signals = ["lottery","won $","nigerian","acc0unt","v3rify","paypa1",
#                     "enr0n.","enronn.","fedx","routing number","bank details","ssn"]
#     if any(s in t for s in spam_signals):
#         return {"label":"spam","priority":"low","action_type":"archive",
#                 "response_text":None,"confidence":0.95}

#     urgent_signals = ["critical","emergency","subpoena","sec investigation","breach",
#                       "server down","margin call","wire failed","pipeline rupture",
#                       "data breach","trading platform offline"]
#     if any(s in t for s in urgent_signals):
#         return {"label":"urgent","priority":"urgent","action_type":"escalate",
#                 "response_text": _RESPONSE_TEMPLATES["escalate"],"confidence":0.90}

#     finance_signals = ["invoice","payment due","payroll","wire transfer","accounts payable",
#                        "budget approval","expense report","fx hedging"]
#     if any(s in t for s in finance_signals):
#         return {"label":"finance","priority":"high","action_type":"respond",
#                 "response_text": _RESPONSE_TEMPLATES["respond_finance"],"confidence":0.80}

#     support_signals = ["vpn","helpdesk","ticket","cannot access","error 403","laptop",
#                        "bloomberg terminal","printer","workstation"]
#     if any(s in t for s in support_signals):
#         return {"label":"support","priority":"medium","action_type":"respond",
#                 "response_text": _RESPONSE_TEMPLATES["respond_support"],"confidence":0.75}

#     newsletter_signals = ["unsubscribe","newsletter","weekly digest","open enrollment",
#                           "congratulations on","building closure","training module"]
#     if any(s in t for s in newsletter_signals):
#         return {"label":"newsletter","priority":"low","action_type":"archive",
#                 "response_text":None,"confidence":0.85}

#     promotions_signals = ["black friday","50% off","shipped","amazon order","pre-approved","loan"]
#     if any(s in t for s in promotions_signals):
#         return {"label":"promotions","priority":"low","action_type":"archive",
#                 "response_text":None,"confidence":0.82}

#     return {"label":"work","priority":"medium","action_type":"respond",
#             "response_text": _RESPONSE_TEMPLATES["respond_work"],"confidence":0.65}


# def _build_action(parsed: Dict[str, Any]) -> AgentAction:
#     """Convert parsed LLM JSON to a validated AgentAction, with safety fills."""
#     action_type  = parsed.get("action_type", "archive")
#     response_text = parsed.get("response_text") or ""
#     response_text = response_text.strip() if isinstance(response_text, str) else ""

#     # Ensure response_text is non-empty when required
#     needs_response = action_type in ("respond", "escalate", "request_clarification")
#     if needs_response and len(response_text.split()) < 5:
#         label = parsed.get("label", "work")
#         if action_type == "escalate":
#             response_text = _RESPONSE_TEMPLATES["escalate"]
#         elif action_type == "request_clarification":
#             response_text = _RESPONSE_TEMPLATES["request_clarification"]
#         elif label == "finance":
#             response_text = _RESPONSE_TEMPLATES["respond_finance"]
#         elif label == "support":
#             response_text = _RESPONSE_TEMPLATES["respond_support"]
#         else:
#             response_text = _RESPONSE_TEMPLATES["respond_work"]

#     # archive/ignore/defer must have null response
#     if not needs_response:
#         response_text = None  # type: ignore

#     return AgentAction(
#         label         = parsed.get("label",    "work"),
#         priority      = parsed.get("priority", "medium"),
#         action_type   = action_type,
#         response_text = response_text or None,
#         confidence    = float(parsed.get("confidence", 0.75)),
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# # Episode runner
# # ─────────────────────────────────────────────────────────────────────────────

# def run_episode(task_id: str, seed: int) -> Dict[str, Any]:
#     """
#     Run one full episode and emit the required [START]/[STEP]/[END] blocks.
#     Returns a summary dict.
#     """
#     env  = EmailTriageEnv(task_id=task_id, seed=seed)
#     obs  = env.reset()
#     done = False
#     step_rewards: List[float] = []

#     # ── [START] block ────────────────────────────────────────────────────
#     print(f"[START]")
#     print(f"task_id={task_id} seed={seed}")
#     print(f"[/START]")
#     sys.stdout.flush()

#     while not done and obs is not None:
#         # Build compact email prompt (no PII fields needed beyond content)
#         thread_ctx = ""
#         if obs.thread_history:
#             thread_ctx = "\n\nTHREAD:\n" + "\n---\n".join(
#                 f"From: {m.sender}\n{m.body[:200]}" for m in obs.thread_history
#             )

#         followup_note = ""
#         if obs.is_followup:
#             followup_note = f"\n\n[FOLLOW-UP: {obs.followup_reason}]"

#         email_prompt = (
#             f"From: {obs.sender}\n"
#             f"Subject: {obs.subject}\n"
#             f"Date: {obs.timestamp}\n\n"
#             f"{obs.body}"
#             f"{thread_ctx}"
#             f"{followup_note}"
#         )

#         # Get LLM decision (deterministic)
#         parsed = _call_llm(email_prompt, task_id)

#         # Build validated action
#         try:
#             action = _build_action(parsed)
#         except Exception:
#             action = _build_action(_heuristic(email_prompt))

#         # Step the environment
#         try:
#             obs, reward, done, info = env.step(action)
#         except Exception as e:
#             # Last-resort: safe archive
#             safe = AgentAction(label="spam", priority="low",
#                                action_type="archive", confidence=0.5)
#             obs, reward, done, info = env.step(safe)

#         step_rewards.append(max(0.001, min(0.999, reward.total_reward)))
#         email_id = info.get("email_id", "unknown")

#         # ── [STEP] block ─────────────-----────────────────────────────────────
#         print(f"[STEP]")
#         print(f"email_id={email_id}")
#         print(f"action={action.label.value}")
#         print(f"priority={action.priority.value}")
#         print(f"action_type={action.action_type.value}")
#         print(f"reward={round(max(0.001, min(0.999, reward.total_reward)), 4)}")
#         print(f"[/STEP]")
#         sys.stdout.flush()

#     # Episode result
#     result = env.episode_result()
#     final_score = round(max(0.001, min(0.999, result.final_score)), 4)
#     avg_reward  = round(sum(step_rewards) / max(1, len(step_rewards)), 4)

#     # ── [END] block ──────────────────────────────────────────────────────
#     print(f"[END]")
#     print(f"task_id={task_id}")
#     print(f"score={final_score}")
#     print(f"[/END]")
#     sys.stdout.flush()

#     return {
#         "task_id":     task_id,
#         "seed":        seed,
#         "final_score": final_score,
#         "avg_reward":  avg_reward,
#         "n_steps":     len(step_rewards),
#         "breakdown":   result.grader_breakdown,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     parser = argparse.ArgumentParser(description="Email Triage RL — Baseline Inference")
#     parser.add_argument("--task", default="all",
#                         choices=["easy", "medium", "hard", "all"],
#                         help="Which task(s) to run")
#     parser.add_argument("--seed", type=int, default=42,
#                         help="Random seed for reproducibility")
#     parser.add_argument("--output", default="inference_results.json",
#                         help="Path to write JSON results")
#     args = parser.parse_args()

#     tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

#     print(f"# Email Triage RL — Baseline Inference")
#     print(f"# model={MODEL_NAME}  base_url={API_BASE_URL}  seed={args.seed}")
#     print(f"# tasks={tasks}")
#     sys.stdout.flush()

#     all_results: Dict[str, Any] = {}

#     for task_id in tasks:
#         result = run_episode(task_id=task_id, seed=args.seed)
#         all_results[task_id] = result

#     # ── Summary ────────────────────────────────────────────────────────────
#     scores = [r["final_score"] for r in all_results.values()]
#     avg    = round(sum(scores) / max(1, len(scores)), 4)

#     print(f"\n# ── SUMMARY ──────────────────────────────────────────")
#     print(f"# model={MODEL_NAME}")
#     for task_id, r in all_results.items():
#         bd = r.get("breakdown", {})
#         print(f"# {task_id.upper():<8} score={r['final_score']:.4f}  "
#               f"steps={r['n_steps']}  "
#               f"cls={bd.get('avg_classification',0):.3f}  "
#               f"pri={bd.get('avg_priority',0):.3f}  "
#               f"act={bd.get('avg_action',0):.3f}  "
#               f"rsp={bd.get('avg_response',0):.3f}")
#     print(f"# AVERAGE  {avg:.4f}")

#     # Save JSON results
#     output = {
#         "model":    MODEL_NAME,
#         "base_url": API_BASE_URL,
#         "seed":     args.seed,
#         "results":  all_results,
#         "summary":  {
#             "avg_score":   avg,
#             "task_scores": {t: r["final_score"] for t, r in all_results.items()},
#         },
#     }
#     with open(args.output, "w") as f:
#         json.dump(output, f, indent=2)
#     print(f"# Results saved → {args.output}")


# if __name__ == "__main__":
#     main()