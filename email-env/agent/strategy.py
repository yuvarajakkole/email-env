"""
agent/strategy.py — Risk-Aware Strategy Layer
----------------------------------------------
The Strategy Layer sits BETWEEN the raw LLM output and the final action.
It performs four functions:

1. UNCERTAINTY QUANTIFICATION
   Estimates true confidence from multiple signals:
   - LLM stated confidence
   - Historical calibration error
   - Email ambiguity score (conflicting signals)
   - Episode state context (error streak, health)

2. RISK-AWARE ACTION SELECTION
   Maps uncertainty + stakes to the safest profitable action:
   - High uncertainty + high stakes → request_clarification
   - High uncertainty + medium stakes → downgrade priority, still act
   - Very high uncertainty + low stakes → archive (fail safe)
   - Low uncertainty → use LLM recommendation directly

3. CONFIDENCE INTERVAL REASONING
   Maintains a [low, high] confidence interval for each email.
   Wide interval → ambiguous → trigger clarification or conservative action.

4. STRATEGY ADAPTATION
   Reads episode state (consecutive_errors, inbox_health) and shifts
   strategy from aggressive → conservative when the episode is going badly.
   Conversely, rewards good episodes by allowing higher-confidence fast actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from env.models import AgentAction, EpisodeState, ActionType, EmailLabel, PriorityLevel


# ─────────────────────────────────────────────
# Ambiguity Scoring
# ─────────────────────────────────────────────

# Words that make an email genuinely ambiguous
_AMBIGUITY_SIGNALS = [
    # Downplayed urgency
    "no rush", "when you have a chance", "not urgent", "totally fine if busy",
    "just curious", "casual thought", "no pressure", "take your time",
    # Conflicting tone + content
    "between us", "off the record", "just as a friend", "don't mention",
    "just a heads up", "nothing official",
    # Multi-intent markers
    "two things", "also", "separately", "one more thing", "quick question",
]

_URGENCY_CLEAR = [
    "critical", "emergency", "asap", "immediately", "now", "subpoena",
    "breach", "outage", "margin call", "wire failed", "ceo", "board",
]

_SPAM_CLEAR = [
    "lottery", "won $", "nigerian", "acc0unt", "v3rify", "paypa1",
    "enronn.com", "enr0n", "bank details", "ssn", "routing number",
]


@dataclass
class UncertaintyEstimate:
    """Full uncertainty characterisation for one email decision."""
    stated_confidence:   float     # From LLM
    adjusted_confidence: float     # After calibration correction
    confidence_low:      float     # Lower bound
    confidence_high:     float     # Upper bound
    ambiguity_score:     float     # 0=clear, 1=highly ambiguous
    is_high_stakes:      bool
    recommended_action:  str       # strategy-adjusted action type
    reasoning:           str       # Why the strategy was applied


class StrategyLayer:
    """
    Risk-aware wrapper around LLM decisions.

    Usage:
        strategy = StrategyLayer()
        final_action = strategy.apply(llm_action, obs, episode_state, calibrator)
    """

    # Risk thresholds
    CLARIFY_THRESHOLD     = 0.50   # confidence below this → consider clarification
    CONSERVATIVE_THRESHOLD= 0.60   # confidence below this → conservative action
    HIGH_STAKES_LABELS    = {"urgent", "finance"}
    ARCHIVE_SAFE_LABELS   = {"spam", "promotions", "newsletter"}

    def __init__(self, risk_mode: str = "balanced"):
        """
        risk_mode:
          'aggressive'  — prefer faster actions, tolerate more uncertainty
          'balanced'    — default, adapt to episode state
          'conservative'— always prefer safety when uncertain
        """
        self.risk_mode = risk_mode
        self._decisions: List[Dict[str, Any]] = []

    def apply(
        self,
        llm_action:     AgentAction,
        obs:            Any,              # EmailObservation
        episode_state:  EpisodeState,
        calibrator:     Any,              # ConfidenceCalibrator
        task_id:        str = "hard",
    ) -> Tuple[AgentAction, UncertaintyEstimate]:
        """
        Apply strategy layer to the raw LLM action.
        Returns (final_action, uncertainty_estimate).
        """
        # 1. Score ambiguity of this email
        ambiguity = self._score_ambiguity(obs)

        # 2. Estimate true confidence (calibration-corrected)
        stated_conf = llm_action.confidence
        calibrated  = calibrator.reliability_at(stated_conf)
        ece         = calibrator.calibration_error()

        # Blend stated and calibrated (trust calibration more as history grows)
        history_weight = min(0.7, ece * 2)   # more history → more correction
        adj_conf = (1 - history_weight) * stated_conf + history_weight * calibrated

        # Widen confidence interval based on ambiguity
        half_width  = 0.15 + 0.25 * ambiguity
        conf_low    = max(0.0, adj_conf - half_width)
        conf_high   = min(1.0, adj_conf + half_width * 0.5)

        # 3. Determine risk level from episode state
        risk_level = self._compute_risk_level(episode_state)

        # 4. Override action based on uncertainty + stakes
        label        = llm_action.label.value
        is_high_stake= label in self.HIGH_STAKES_LABELS
        is_safe_arch = label in self.ARCHIVE_SAFE_LABELS
        effective_threshold = self._effective_threshold(risk_level)

        final_action = llm_action
        reasoning    = "Strategy: no override — confidence sufficient"

        # Only apply strategy on hard task or when errors are high
        if task_id == "hard" or episode_state.consecutive_errors >= 2:

            # Case A: Very high ambiguity + high stakes + low confidence
            if (ambiguity > 0.6 and is_high_stake
                    and adj_conf < effective_threshold
                    and not obs.is_followup):
                # Ask for clarification first (unless we've already done this)
                if llm_action.action_type.value != "request_clarification":
                    clarify_text = self._build_clarification(obs, label)
                    final_action = AgentAction(
                        label         = llm_action.label,
                        priority      = llm_action.priority,
                        action_type   = ActionType.REQUEST_CLARIFICATION,
                        response_text = clarify_text,
                        confidence    = adj_conf,
                    )
                    reasoning = (
                        f"Strategy: REQUEST_CLARIFICATION — "
                        f"ambiguity={ambiguity:.2f}, adj_conf={adj_conf:.2f}, "
                        f"high-stakes={label}"
                    )

            # Case B: Safe-to-archive label but agent tried to escalate
            elif is_safe_arch and llm_action.action_type.value == "escalate":
                final_action = AgentAction(
                    label         = llm_action.label,
                    priority      = PriorityLevel.LOW,
                    action_type   = ActionType.ARCHIVE,
                    response_text = None,
                    confidence    = max(adj_conf, 0.85),
                )
                reasoning = f"Strategy: ARCHIVE override — {label} should never be escalated"

            # Case C: Low confidence, non-urgent → downgrade priority to avoid false urgency
            elif adj_conf < self.CONSERVATIVE_THRESHOLD and not is_high_stake:
                # Downgrade priority by one level
                current_prio = llm_action.priority.value
                prios        = ["low", "medium", "high", "urgent"]
                idx          = prios.index(current_prio)
                safe_prio    = prios[max(0, idx - 1)]
                final_action = AgentAction(
                    label         = llm_action.label,
                    priority      = safe_prio,
                    action_type   = llm_action.action_type,
                    response_text = llm_action.response_text,
                    confidence    = adj_conf,
                )
                reasoning = (
                    f"Strategy: PRIORITY_DOWNGRADE — "
                    f"adj_conf={adj_conf:.2f} < {self.CONSERVATIVE_THRESHOLD}, "
                    f"priority {current_prio}→{safe_prio}"
                )

            # Case D: Episode going badly (error streak) → conservative escalation
            elif (episode_state.consecutive_errors >= 3
                    and is_high_stake
                    and llm_action.action_type.value not in ("escalate", "respond")):
                resp = (llm_action.response_text or
                        "Flagging for senior review given the sensitivity of this matter.")
                final_action = AgentAction(
                    label         = llm_action.label,
                    priority      = PriorityLevel.HIGH,
                    action_type   = ActionType.ESCALATE,
                    response_text = resp,
                    confidence    = adj_conf,
                )
                reasoning = (
                    f"Strategy: ESCALATE override — error streak "
                    f"({episode_state.consecutive_errors}), high-stakes label"
                )

        estimate = UncertaintyEstimate(
            stated_confidence   = stated_conf,
            adjusted_confidence = round(adj_conf, 3),
            confidence_low      = round(conf_low, 3),
            confidence_high     = round(conf_high, 3),
            ambiguity_score     = round(ambiguity, 3),
            is_high_stakes      = is_high_stake,
            recommended_action  = final_action.action_type.value,
            reasoning           = reasoning,
        )

        self._decisions.append({
            "email_id":     obs.email_id,
            "ambiguity":    ambiguity,
            "adj_conf":     adj_conf,
            "original":     llm_action.action_type.value,
            "final":        final_action.action_type.value,
            "override":     final_action.action_type != llm_action.action_type,
            "reasoning":    reasoning,
        })

        return final_action, estimate

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _score_ambiguity(self, obs: Any) -> float:
        """
        Compute email ambiguity in [0, 1].
        Combines lexical signals, sender trust, and structural markers.
        """
        text   = (obs.body + " " + obs.subject).lower()
        sender = obs.sender.lower()

        # Count ambiguity signals
        amb_hits   = sum(1 for s in _AMBIGUITY_SIGNALS if s in text)
        clear_hits = sum(1 for s in _URGENCY_CLEAR if s in text)
        spam_hits  = sum(1 for s in _SPAM_CLEAR if s in text)

        # Thread context adds uncertainty
        thread_factor = 0.1 * min(len(obs.thread_history), 3)

        # Multi-intent signal: email mentions multiple topics
        multi_intent = 0.2 if any(s in text for s in ["two things", "also,", "separately,"]) else 0.0

        # Downplayed urgency is the hardest pattern
        downplay = 0.35 if any(s in text for s in
                               ["no rush", "not urgent", "casual", "when you have a chance"]) else 0.0

        # Clear signals reduce ambiguity
        clarity_reduction = 0.15 * min(clear_hits, 3) + 0.2 * min(spam_hits, 2)

        raw = (0.15 * min(amb_hits, 4) + thread_factor + multi_intent
               + downplay - clarity_reduction)

        return max(0.0, min(1.0, raw))

    def _compute_risk_level(self, ep: EpisodeState) -> str:
        """
        'conservative' if episode going badly, 'aggressive' if healthy.
        """
        if self.risk_mode != "balanced":
            return self.risk_mode
        if ep.consecutive_errors >= 3 or ep.inbox_health < 0.5:
            return "conservative"
        if ep.consecutive_errors == 0 and ep.inbox_health > 0.85:
            return "aggressive"
        return "balanced"

    def _effective_threshold(self, risk_level: str) -> float:
        return {"conservative": 0.65, "balanced": 0.50, "aggressive": 0.40}[risk_level]

    def _build_clarification(self, obs: Any, label: str) -> str:
        """Build a context-appropriate clarification question."""
        if label == "urgent":
            return (
                f"To ensure I handle this with the appropriate urgency, could you please "
                f"confirm: (1) the exact deadline, and (2) which team should be the primary "
                f"point of contact? I want to make sure I escalate this correctly."
            )
        elif label == "finance":
            return (
                f"Thank you for your message. Before I process this, could you please "
                f"confirm the purchase order number and the authorising budget holder? "
                f"I want to ensure this is routed to the correct approval chain."
            )
        else:
            return (
                f"Thank you for reaching out. To ensure I take the right action on this, "
                f"could you clarify the priority level and whether there is a specific "
                f"deadline I should be aware of?"
            )

    def override_rate(self) -> float:
        if not self._decisions:
            return 0.0
        overrides = sum(1 for d in self._decisions if d["override"])
        return round(overrides / len(self._decisions), 3)

    def summary(self) -> Dict[str, Any]:
        avg_amb = sum(d["ambiguity"] for d in self._decisions) / max(1, len(self._decisions))
        return {
            "decisions":    len(self._decisions),
            "override_rate": self.override_rate(),
            "avg_ambiguity": round(avg_amb, 3),
            "risk_mode":    self.risk_mode,
        }
