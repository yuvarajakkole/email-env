"""
agent/memory.py — Episode Memory & Cross-Episode Learning
----------------------------------------------------------
Implements two memory systems:

1. StepMemory — stores every (observation, action, reward, state) tuple
   within a single episode. Enables the agent to recall what it has
   already seen and adjust its strategy mid-episode.

2. EpisodeMemory — persists learned patterns ACROSS episodes.
   After each episode, the agent extracts:
     - which email patterns it got wrong
     - which action strategies worked best
     - the calibration of its own confidence scores
   This drives adaptation: future episodes benefit from past mistakes.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# Step-level memory record
# ─────────────────────────────────────────────

@dataclass
class StepRecord:
    step:           int
    email_id:       str
    subject:        str
    sender:         str
    body_snippet:   str          # first 200 chars
    label_pred:     str
    label_true:     str
    priority_pred:  str
    priority_true:  str
    action_pred:    str
    action_true:    str
    confidence:     float
    reward:         float
    penalty:        float
    cls_score:      float
    is_correct:     bool         # cls_score >= 0.7
    difficulty:     str
    is_followup:    bool

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# ─────────────────────────────────────────────
# Pattern learner (what the agent got wrong)
# ─────────────────────────────────────────────

@dataclass
class MistakePattern:
    """Represents a recurring classification or action mistake."""
    true_label:     str
    pred_label:     str
    signal_words:   List[str]    # words that appeared in misclassified emails
    count:          int = 0
    last_reward:    float = 0.0


class PatternLearner:
    """
    Mines step records to find recurring mistakes and extract
    discriminative signal words for corrections.
    """

    def __init__(self):
        self._mistake_counts: Dict[Tuple[str,str], int] = defaultdict(int)
        self._signal_words:   Dict[Tuple[str,str], List[str]] = defaultdict(list)

    def record(self, record: StepRecord) -> None:
        if not record.is_correct:
            key = (record.label_true, record.label_pred)
            self._mistake_counts[key] += 1
            # Extract distinctive words from body
            words = set(record.body_snippet.lower().split())
            existing = set(self._signal_words[key])
            new_words = [w for w in words if len(w) > 4 and w not in existing][:5]
            self._signal_words[key].extend(new_words)

    def top_mistakes(self, n: int = 5) -> List[MistakePattern]:
        patterns = []
        for (true_l, pred_l), count in sorted(
                self._mistake_counts.items(), key=lambda x: -x[1])[:n]:
            patterns.append(MistakePattern(
                true_label   = true_l,
                pred_label   = pred_l,
                signal_words = list(set(self._signal_words[(true_l, pred_l)]))[:10],
                count        = count,
            ))
        return patterns

    def correction_hints(self) -> str:
        """
        Build a natural-language correction hint block for the LLM prompt.
        Injected into the system prompt so the agent learns from past mistakes.
        """
        top = self.top_mistakes(3)
        if not top:
            return ""
        lines = ["## Learned Corrections (from previous episodes)"]
        for m in top:
            words_str = ", ".join(m.signal_words[:5]) if m.signal_words else "n/a"
            lines.append(
                f"- You previously confused '{m.true_label}' → predicted '{m.pred_label}' "
                f"({m.count}x). Signal words: [{words_str}]. Correct this."
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Confidence calibration tracker
# ─────────────────────────────────────────────

class ConfidenceCalibrator:
    """
    Tracks predicted confidence vs actual accuracy to compute calibration error.
    Used by StrategyLayer to decide when to request_clarification vs act.
    """

    def __init__(self, n_bins: int = 5):
        self._n_bins = n_bins
        # Each bin: (sum_correct, count)
        self._bins: List[List[float]] = [[0.0, 0.0] for _ in range(n_bins)]

    def record(self, confidence: float, was_correct: bool) -> None:
        bin_idx = min(int(confidence * self._n_bins), self._n_bins - 1)
        self._bins[bin_idx][0] += float(was_correct)
        self._bins[bin_idx][1] += 1.0

    def calibration_error(self) -> float:
        """Expected Calibration Error (ECE) — lower is better."""
        total_samples = sum(b[1] for b in self._bins)
        if total_samples == 0:
            return 0.0
        ece = 0.0
        for i, (correct, count) in enumerate(self._bins):
            if count == 0:
                continue
            bin_conf   = (i + 0.5) / self._n_bins
            bin_acc    = correct / count
            ece       += (count / total_samples) * abs(bin_acc - bin_conf)
        return round(ece, 4)

    def reliability_at(self, confidence: float) -> float:
        """What fraction of predictions at this confidence level were correct?"""
        bin_idx = min(int(confidence * self._n_bins), self._n_bins - 1)
        correct, count = self._bins[bin_idx]
        return correct / count if count > 0 else confidence  # fallback to stated conf

    def should_seek_clarification(self, confidence: float,
                                   ece: float, threshold: float = 0.45) -> bool:
        """
        True if the agent's calibration is poor AND stated confidence is low.
        Signals that requesting clarification is better than acting blindly.
        """
        reliability = self.reliability_at(confidence)
        return reliability < threshold and ece > 0.15


# ─────────────────────────────────────────────
# Episode Memory (persists across episodes)
# ─────────────────────────────────────────────

class EpisodeMemory:
    """
    Full memory system — persists to disk across runs.

    Tracks:
      - All step records per episode
      - Mistake patterns (for prompt injection)
      - Confidence calibration
      - Per-label accuracy history
      - Action efficiency history
    """

    def __init__(self, memory_path: Optional[str] = None):
        self._memory_path = memory_path or os.getenv("MEMORY_PATH", "agent_memory.json")
        self._episodes:    List[Dict[str, Any]] = []
        self._step_buffer: List[StepRecord]     = []
        self.pattern_learner  = PatternLearner()
        self.calibrator       = ConfidenceCalibrator()

        # Per-label accuracy (label → [correct_count, total_count])
        self._label_acc: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        # Action efficiency (action → avg_reward)
        self._action_rewards: Dict[str, List[float]] = defaultdict(list)

        self._load()

    # ── Within-episode recording ──────────────────────────────────────────

    def record_step(
        self,
        step:         int,
        obs:          Any,           # EmailObservation
        action:       Any,           # AgentAction
        reward:       Any,           # StepReward
        ground_truth: Dict[str, Any],
        is_followup:  bool = False,
    ) -> None:
        cls_score  = reward.classification_score
        is_correct = cls_score >= 0.7

        rec = StepRecord(
            step          = step,
            email_id      = obs.email_id,
            subject       = obs.subject[:80],
            sender        = obs.sender,
            body_snippet  = obs.body[:200],
            label_pred    = action.label.value,
            label_true    = ground_truth.get("label", ""),
            priority_pred = action.priority.value,
            priority_true = ground_truth.get("priority", ""),
            action_pred   = action.action_type.value,
            action_true   = ground_truth.get("action_type", ""),
            confidence    = action.confidence,
            reward        = reward.total_reward,
            penalty       = reward.penalty,
            cls_score     = cls_score,
            is_correct    = is_correct,
            difficulty    = ground_truth.get("difficulty", "medium"),
            is_followup   = is_followup,
        )

        self._step_buffer.append(rec)
        self.pattern_learner.record(rec)
        self.calibrator.record(action.confidence, is_correct)

        # Update per-label accuracy
        true_label = ground_truth.get("label", "")
        if true_label:
            self._label_acc[true_label][1] += 1
            if is_correct:
                self._label_acc[true_label][0] += 1

        # Update action efficiency
        self._action_rewards[action.action_type.value].append(reward.total_reward)

    def commit_episode(self, task_id: str, final_score: float,
                       episode_state: Optional[Any] = None) -> None:
        """Called at episode end — commits buffer to long-term memory."""
        ep_summary = {
            "task_id":     task_id,
            "final_score": final_score,
            "n_steps":     len(self._step_buffer),
            "n_correct":   sum(1 for r in self._step_buffer if r.is_correct),
            "avg_reward":  round(
                sum(r.reward for r in self._step_buffer) / max(1, len(self._step_buffer)), 4),
            "top_mistakes": [m.__dict__ for m in self.pattern_learner.top_mistakes(3)],
            "ece":         self.calibrator.calibration_error(),
            "steps":       [r.to_dict() for r in self._step_buffer],
        }
        if episode_state:
            ep_summary["final_inbox_health"]  = getattr(episode_state, "inbox_health", 1.0)
            ep_summary["final_user_sat"]       = getattr(episode_state, "user_satisfaction", 1.0)
            ep_summary["ignored_urgent"]       = getattr(episode_state, "ignored_urgent", 0)

        self._episodes.append(ep_summary)
        self._step_buffer = []
        self._save()

    # ── Querying ──────────────────────────────────────────────────────────

    def weakest_labels(self, n: int = 3) -> List[str]:
        """Return labels the agent most often misclassifies."""
        accuracies = {
            label: correct / total
            for label, (correct, total) in self._label_acc.items()
            if total >= 2
        }
        return [k for k, _ in sorted(accuracies.items(), key=lambda x: x[1])[:n]]

    def best_action_for_label(self, label: str) -> Optional[str]:
        """Return the action that has historically yielded the highest reward."""
        if not self._action_rewards:
            return None
        return max(self._action_rewards.items(),
                   key=lambda kv: sum(kv[1]) / max(1, len(kv[1])))[0]

    def correction_prompt_block(self) -> str:
        """Full correction block to inject into LLM system prompt."""
        hints = self.pattern_learner.correction_hints()
        weak  = self.weakest_labels(3)
        ece   = self.calibrator.calibration_error()
        parts = []
        if hints:
            parts.append(hints)
        if weak:
            parts.append(f"## Labels to Watch\nYou struggle most with: {', '.join(weak)}. Pay extra attention.")
        if ece > 0.15:
            parts.append(
                f"## Confidence Calibration Warning\n"
                f"Your calibration error is {ece:.2f} (target < 0.10). "
                f"You are overconfident. Use lower confidence scores (0.65–0.80) unless truly certain."
            )
        return "\n\n".join(parts)

    def n_episodes(self) -> int:
        return len(self._episodes)

    def recent_scores(self, n: int = 5) -> List[float]:
        return [ep["final_score"] for ep in self._episodes[-n:]]

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            data = {
                "episodes":     self._episodes[-50:],  # keep last 50
                "label_acc":    dict(self._label_acc),
                "action_rewards": {k: v[-20:] for k, v in self._action_rewards.items()},
            }
            with open(self._memory_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # memory persistence is best-effort

    def _load(self) -> None:
        path = Path(self._memory_path)
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._episodes     = data.get("episodes", [])
            raw_label          = data.get("label_acc", {})
            self._label_acc    = defaultdict(lambda: [0.0, 0.0],
                                             {k: v for k, v in raw_label.items()})
            raw_actions        = data.get("action_rewards", {})
            self._action_rewards = defaultdict(list,
                                               {k: v for k, v in raw_actions.items()})
        except Exception:
            pass
