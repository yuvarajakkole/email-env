"""
agent/policy.py — Intelligent Triage Policy
--------------------------------------------
The TriagePolicy is the complete agent "brain".
It combines:

  LLM (language understanding)
    ↓
  EpisodeMemory (what went wrong before)
    ↓
  StrategyLayer (risk-aware action selection)
    ↓
  Final AgentAction

Architecture:
  - LLM generates a candidate action with chain-of-thought reasoning
  - Memory injects learned corrections into the system prompt
  - Strategy layer applies uncertainty quantification and risk overrides
  - Final action is logged back to memory for future learning

The policy maintains a live "policy state" across steps:
  - Tracks which emails in the current episode were high-ambiguity
  - Adjusts prompting strategy based on consecutive_errors
  - Uses different temperature/few-shot configs per difficulty
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI, APIError, RateLimitError

from env.models import AgentAction, ActionType, EmailObservation, EpisodeState
from agent.memory import EpisodeMemory
from agent.strategy import StrategyLayer, UncertaintyEstimate
from config import settings


# ─────────────────────────────────────────────
# Static system prompt components
# ─────────────────────────────────────────────

_BASE_SYSTEM = """You are an expert enterprise email triage system at Enron Corporation (2001).
Your decisions have lasting consequences: poor choices degrade inbox_health and user_satisfaction,
missed urgent emails trigger angry executive escalations, and repeated errors compound.

## Extended Action Space
- respond: reply to sender (response_text ≥ 25 words, specific to email content)
- archive: file away, no reply (FYI, newsletters, resolved items)
- escalate: hand to senior management (response_text required explaining what + who)
- ignore: discard (ONLY for obvious spam/junk — never for legitimate email)
- defer: snooze (COSTLY for urgent/finance — triggers follow-up chaser)
- request_clarification: ask sender a question (response_text = your question, ≥ 20 words)

## Core Decision Rules
1. Sender domain typos (enr0n, paypa1, enronn.com, fedx) → SPAM → archive/ignore
2. "No rush but CEO needs today" = URGENT — casual framing does NOT reduce priority
3. Multi-intent email → classify by the most consequential/time-sensitive element
4. Finance + deadline → HIGH; Finance + same-day → URGENT
5. Legal/compliance/insider-info signals → ESCALATE regardless of tone
6. Board/C-suite sender → respond promptly even if phrased casually
7. Use request_clarification only when acting without info would cause harm
8. Never defer urgent items — it triggers a penalty and an angry follow-up

## Response Requirements
- Minimum 25 words, maximum 100 words
- Address the specific ask in the email
- For escalate: name what you're escalating and to whom
- Never write meta-commentary ("As an AI...", "I cannot...")
- Never write generic one-liners ("OK", "Noted", "Will do")

## Episode State Context
The episode_state block shows your running history.
If consecutive_errors > 2, you are making systematic mistakes — slow down and reason carefully.
If inbox_health < 0.6, you are in trouble — be conservative and prefer escalate over ignore.

## Output — JSON ONLY, no markdown, no extra text:
{
  "reasoning": "2-3 sentences: what signals you used, why this label/action",
  "label": "spam|work|finance|promotions|personal|urgent|newsletter|support",
  "priority": "low|medium|high|urgent",
  "action_type": "respond|archive|escalate|ignore|defer|request_clarification",
  "response_text": "your reply (null only if action is archive or ignore)",
  "confidence": 0.0-1.0
}"""

_FEW_SHOTS: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "From: payroll-update@enronn.com\n"
            "Subject: URGENT: Verify your direct deposit details\n\n"
            "ACTION REQUIRED: All employees must re-verify bank account details by EOD. "
            "Submit at: https://enronn.com/payroll/verify — HR Department"
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "reasoning": "Domain typo 'enronn.com' (double-n) is a spear-phishing indicator. Requests banking credentials under false urgency — classic social engineering.",
            "label": "spam", "priority": "low", "action_type": "archive",
            "response_text": None, "confidence": 0.97
        })
    },
    {
        "role": "user",
        "content": (
            "From: executive.assistant@enron.com\n"
            "Subject: No rush, but the CEO needs this by close of business\n\n"
            "Hey, no huge rush, but Jeff mentioned he'd like the California trading desk "
            "summary before he leaves at 5:30 today. Shouldn't take more than an hour."
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "reasoning": "Soft framing ('no rush') masks a real CEO deadline at 5:30pm today. Executive assistant sender + CEO request = urgent regardless of phrasing.",
            "label": "work", "priority": "urgent", "action_type": "respond",
            "response_text": "Understood — I am prioritising the California trading desk summary immediately and will have it in your inbox before 5:30pm. Will flag at once if I hit any blockers.",
            "confidence": 0.88
        })
    },
    {
        "role": "user",
        "content": (
            "From: board.advisor@enron.com\n"
            "Subject: RE: The Dynegy merger\n\n"
            "Between us — I think the merger is dead. Stock in freefall. "
            "You may want to think about your exposure. Not saying anything officially. Just as a friend.\n\n"
            "EPISODE STATE: consecutive_errors=0 inbox_health=0.85"
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "reasoning": "Casual framing conceals potential MNPI (material non-public information) about a pending merger. Legal/compliance obligation to escalate regardless of social framing.",
            "label": "urgent", "priority": "urgent", "action_type": "escalate",
            "response_text": "Escalating immediately to Legal and Compliance. This message may contain material non-public information regarding the Dynegy merger and requires immediate review by legal counsel before any action is taken.",
            "confidence": 0.93
        })
    },
]


# ─────────────────────────────────────────────
# Triage Policy
# ─────────────────────────────────────────────

class TriagePolicy:
    """
    The complete intelligent agent policy.

    Lifecycle:
        policy = TriagePolicy(memory=memory, strategy=strategy)
        policy.begin_episode(task_id)
        for each step:
            action, uncertainty = policy.act(obs, task_id)
            policy.observe(obs, action, reward, ground_truth, info)
        policy.end_episode(final_score, episode_state)
    """

    MAX_RETRIES = 3
    RETRY_WAIT  = 2.0

    # Per-task LLM temperatures: easy=deterministic, hard=slightly creative
    TEMPERATURES = {"easy": 0.0, "medium": 0.05, "hard": 0.10}
    # Max tokens per task
    MAX_TOKENS   = {"easy": 200, "medium": 300, "hard": 600}

    def __init__(
        self,
        memory:   EpisodeMemory,
        strategy: StrategyLayer,
        verbose:  bool = False,
    ):
        self.memory   = memory
        self.strategy = strategy
        self.verbose  = verbose

        self._client  = OpenAI(
            api_key  = str(settings.openai_api_key),
            base_url = settings.api_base_url,
        )
        self._model     = settings.model_name
        self._task_id   = "hard"
        self._step      = 0
        self._call_count = 0
        self._fail_count = 0

    # ── Episode lifecycle ─────────────────────────────────────────────────

    def begin_episode(self, task_id: str) -> None:
        self._task_id = task_id
        self._step    = 0

    def observe(
        self,
        obs:          EmailObservation,
        action:       AgentAction,
        reward:       Any,              # StepReward
        ground_truth: Dict[str, Any],
        info:         Dict[str, Any],
    ) -> None:
        """Record step outcome into memory."""
        self.memory.record_step(
            step         = self._step,
            obs          = obs,
            action       = action,
            reward       = reward,
            ground_truth = ground_truth,
            is_followup  = info.get("is_followup", False),
        )
        self._step += 1

    def end_episode(self, final_score: float, episode_state: Any) -> None:
        self.memory.commit_episode(self._task_id, final_score, episode_state)

    # ── Core act() ────────────────────────────────────────────────────────

    def act(
        self,
        obs:     EmailObservation,
        task_id: str,
    ) -> Tuple[AgentAction, UncertaintyEstimate]:
        """
        Generate an action with full uncertainty quantification.
        Steps:
          1. Build memory-augmented prompt
          2. Call LLM (with retry)
          3. Apply strategy layer
          4. Return final action + uncertainty estimate
        """
        ep = obs.episode_state

        for attempt in range(self.MAX_RETRIES):
            try:
                llm_action = self._llm_call(obs, task_id, ep)
                break
            except (APIError, RateLimitError) as e:
                wait = self.RETRY_WAIT * (2 ** attempt)
                if self.verbose:
                    print(f"  ⚠️  API error attempt {attempt+1}: {e}. Retry in {wait:.0f}s")
                time.sleep(wait)
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Parse error attempt {attempt+1}: {e}")
                if attempt == self.MAX_RETRIES - 1:
                    self._fail_count += 1
                    llm_action = self._heuristic_fallback(obs)
                    break
        else:
            self._fail_count += 1
            llm_action = self._heuristic_fallback(obs)

        # Apply strategy layer (risk-aware override)
        final_action, uncertainty = self.strategy.apply(
            llm_action    = llm_action,
            obs           = obs,
            episode_state = ep,
            calibrator    = self.memory.calibrator,
            task_id       = task_id,
        )

        if self.verbose and uncertainty.recommended_action != llm_action.action_type.value:
            print(f"  🧠 Strategy override: "
                  f"{llm_action.action_type.value} → {uncertainty.recommended_action} | "
                  f"{uncertainty.reasoning}")

        self._call_count += 1
        return final_action, uncertainty

    # ── LLM call ──────────────────────────────────────────────────────────

    def _llm_call(
        self,
        obs:     EmailObservation,
        task_id: str,
        ep:      EpisodeState,
    ) -> AgentAction:
        messages = self._build_messages(obs, task_id, ep)
        temp     = self.TEMPERATURES.get(task_id, 0.05)
        max_tok  = self.MAX_TOKENS.get(task_id, 400)

        kwargs: Dict[str, Any] = dict(
            model      = self._model,
            messages   = messages,
            temperature= temp,
            max_tokens = max_tok,
        )
        if "gpt" in self._model.lower():
            kwargs["response_format"] = {"type": "json_object"}

        raw = self._client.chat.completions.create(
            **kwargs).choices[0].message.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        rsp    = (parsed.get("response_text") or "").strip() or None

        return AgentAction(
            label         = parsed["label"],
            priority      = parsed["priority"],
            action_type   = parsed["action_type"],
            response_text = rsp,
            confidence    = float(parsed.get("confidence", 0.8)),
        )

    def _build_messages(
        self,
        obs:     EmailObservation,
        task_id: str,
        ep:      EpisodeState,
    ) -> List[Dict[str, str]]:
        # Base system prompt + memory corrections
        correction_block = self.memory.correction_prompt_block()
        system_content   = _BASE_SYSTEM
        if correction_block:
            system_content += f"\n\n{correction_block}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        # Few-shot examples for medium + hard tasks
        if task_id in ("medium", "hard"):
            messages.extend(_FEW_SHOTS)

        # Build user message
        thread_ctx = ""
        if obs.thread_history:
            thread_ctx = "\n\nTHREAD HISTORY (oldest → newest):\n" + "\n---\n".join(
                f"From: {m.sender} [{m.timestamp}]\n{m.body}"
                for m in obs.thread_history
            )

        followup_ctx = ""
        if obs.is_followup:
            followup_ctx = (
                f"\n\n⚠️  FOLLOW-UP ALERT: This email was triggered by a previous bad decision "
                f"({obs.followup_reason}). The sender is now more frustrated. "
                f"Handle with elevated urgency and a genuine apology."
            )

        # Adaptive instruction based on episode state
        adaptive = ""
        if ep.consecutive_errors >= 3:
            adaptive = (
                "\n\n⚠️  PERFORMANCE WARNING: You have made "
                f"{ep.consecutive_errors} consecutive errors. Slow down. "
                "Reason step by step. Prioritise accuracy over speed."
            )
        elif ep.inbox_health < 0.5:
            adaptive = (
                "\n\n⚠️  INBOX HEALTH CRITICAL. Be conservative. "
                "When in doubt, escalate rather than ignore or defer."
            )

        weakest = self.memory.weakest_labels(2)
        weak_hint = ""
        if weakest:
            weak_hint = (
                f"\n\n💡 FOCUS: You historically struggle with: {', '.join(weakest)}. "
                "Pay extra attention if this email might belong to those categories."
            )

        user_msg = (
            f"TASK: {task_id.upper()}\n"
            f"EPISODE STATE: step={ep.step} inbox_health={ep.inbox_health:.2f} "
            f"user_satisfaction={ep.user_satisfaction:.2f} "
            f"consecutive_errors={ep.consecutive_errors} "
            f"unresolved_threads={len(ep.unresolved_threads)} "
            f"total_cost={ep.total_cost:.3f}\n\n"
            f"From: {obs.sender}\n"
            f"To:   {obs.recipient}\n"
            f"Date: {obs.timestamp}\n"
            f"Subject: {obs.subject}\n"
            f"Has Attachments: {obs.has_attachments}"
            f"{thread_ctx}"
            f"\n\nBODY:\n{obs.body}"
            f"{followup_ctx}{adaptive}{weak_hint}\n\n"
            f"Return JSON only."
        )

        messages.append({"role": "user", "content": user_msg})
        return messages

    # ── Heuristic fallback ─────────────────────────────────────────────────

    def _heuristic_fallback(self, obs: EmailObservation) -> AgentAction:
        """Rule-based fallback when LLM fails."""
        body   = (obs.body + " " + obs.subject).lower()
        sender = obs.sender.lower()

        spam_kw = ["lottery", "prize", "won $", "nigerian", "prince",
                   "bank details", "acc0unt", "v3rify", "paypa1",
                   "enr0n.", "enronn.", "fedx", "routing number"]
        if any(k in body for k in spam_kw) or \
                any(s in sender for s in [".ru", ".ng", "secret", "prize", "enronn"]):
            return AgentAction(label="spam", priority="low",
                               action_type="archive", confidence=0.90)

        if any(k in body for k in ["critical", "emergency", "subpoena", "breach",
                                    "server down", "margin call", "wire failed"]):
            return AgentAction(
                label="urgent", priority="urgent", action_type="escalate",
                response_text=(
                    "Acknowledged — escalating immediately to senior management and the "
                    "relevant incident response team. I will monitor and provide updates "
                    "every 30 minutes until this is resolved."
                ),
                confidence=0.85)

        if any(k in body for k in ["invoice", "payment due", "payroll", "wire transfer"]):
            return AgentAction(
                label="finance", priority="high", action_type="respond",
                response_text=(
                    "Thank you — I have received this and forwarded it to accounts payable "
                    "for processing. I will confirm once payment has been actioned."
                ),
                confidence=0.75)

        if any(k in body for k in ["unsubscribe", "newsletter", "weekly digest"]):
            return AgentAction(label="newsletter", priority="low",
                               action_type="archive", confidence=0.82)

        return AgentAction(
            label="work", priority="medium", action_type="respond",
            response_text=(
                "Thank you for your message. I have reviewed the details and will action "
                "this before the deadline. I will follow up with a full update shortly."
            ),
            confidence=0.62)

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "total_calls":   self._call_count,
            "failures":      self._fail_count,
            "fallback_rate": round(self._fail_count / max(1, self._call_count), 3),
            "memory_episodes": self.memory.n_episodes(),
            "ece":           self.memory.calibrator.calibration_error(),
            "strategy":      self.strategy.summary(),
            "weakest_labels": self.memory.weakest_labels(3),
        }
