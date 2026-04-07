"""
Email Triage Environment — Core v3
-------------------------------------
OpenEnv specification:
  - reset()  → initialises episode + persistent state, returns first Observation
  - step()   → grades action, applies long-term effects, injects follow-up emails
  - state()  → returns full structured state snapshot

Long-term effects implemented:
  1. Wrong classification → inbox_health decreases, consecutive_errors increases
  2. Ignored urgent email → injected angry follow-up 2 steps later + penalty
  3. Poor response → user_satisfaction drops, deferred threads resurface
  4. Deferred emails → added to unresolved_threads, resurface as harder variant
  5. Consecutive errors → difficulty modifier applied to grading weights
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    AgentAction, EmailObservation, EpisodeState, StepReward,
    EpisodeResult, ThreadMessage, ActionType, ACTION_COSTS,
)
from tasks.manager import TaskManager
from graders.engine import get_grader
from rewards.engine import compute_reward


# ── Long-term effect constants ───────────────────────────────────────────────
INBOX_HEALTH_HIT_WRONG_CLASS   = 0.06
INBOX_HEALTH_HIT_IGNORE_URGENT = 0.15
INBOX_HEALTH_HIT_BAD_RESPONSE  = 0.04
INBOX_HEALTH_GAIN_PERFECT      = 0.02

USER_SAT_HIT_BAD_RESPONSE      = 0.08
USER_SAT_HIT_WRONG_ACTION      = 0.05
USER_SAT_GAIN_GOOD_RESPONSE    = 0.03

CONSECUTIVE_ERROR_PENALTY      = 0.05   # extra per step while streak active


class EmailTriageEnv:
    """
    Production RL environment for email triage with persistent state.

    Usage:
        env = EmailTriageEnv(task_id="hard", seed=42)
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        result = env.episode_result()
    """

    def __init__(self, task_id: str = "medium", seed: int = 42):
        self.task_id  = task_id
        self.seed     = seed
        self._manager = TaskManager(task_id=task_id, seed=seed)
        self._grader  = get_grader(task_id)

        self._step_rewards: List[StepReward] = []
        self._cumulative_reward: float = 0.0
        self._done:    bool = False
        self._started: bool = False

        # Persistent episode state
        self._ep_state: EpisodeState = EpisodeState()

        # Follow-up email injection queue: list of email dicts to insert
        self._injection_queue: List[Dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────────────
    # OpenEnv Interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> EmailObservation:
        """Initialise a new episode with fresh persistent state."""
        self._manager.reset()
        self._step_rewards        = []
        self._cumulative_reward   = 0.0
        self._done                = False
        self._started             = True
        self._injection_queue     = []
        self._ep_state            = EpisodeState()
        return self._build_observation()

    def step(
        self,
        action: AgentAction,
    ) -> Tuple[Optional[EmailObservation], StepReward, bool, Dict[str, Any]]:
        """
        Apply action → grade → compute reward → update persistent state →
        potentially inject follow-up email → advance cursor.
        """
        if not self._started:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode done. Call reset() to start a new episode.")

        # Serve injected follow-up first if queued
        current_email = self._get_current_email()
        if current_email is None:
            self._done = True
            return None, StepReward(), True, {}

        ground_truth = current_email["ground_truth"]
        is_followup  = current_email.get("_injected", False)

        # ── 1. Grade action ───────────────────────────────────────────────
        grader_result = self._grader.grade(action, ground_truth)

        # ── 2. Compute reward (with episode state context) ────────────────
        reward = compute_reward(
            task_id       = self.task_id,
            action        = action,
            grader_result = grader_result,
            ground_truth  = ground_truth,
            episode_state = self._ep_state,
        )

        # ── 3. Apply action cost ──────────────────────────────────────────
        cost = ACTION_COSTS.get(action.action_type.value, 0.0)
        self._ep_state.total_cost = round(self._ep_state.total_cost + cost, 4)

        # ── 4. Update persistent state (long-term effects) ────────────────
        self._update_episode_state(action, grader_result, ground_truth, reward)

        # ── 5. Maybe inject follow-up email ───────────────────────────────
        self._maybe_inject_followup(action, ground_truth, current_email)

        # ── 6. Accumulate ─────────────────────────────────────────────────
        self._step_rewards.append(reward)
        self._cumulative_reward = round(self._cumulative_reward + reward.total_reward, 4)

        # ── 7. Advance ────────────────────────────────────────────────────
        self._advance()
        self._ep_state.step += 1

        done = self._is_done()
        self._done = done
        next_obs   = self._build_observation() if not done else None

        info = {
            "email_id":         current_email["email_id"],
            "ground_truth":     ground_truth,
            "grader_result":    grader_result,
            "step":             self._ep_state.step,
            "emails_remaining": self._emails_remaining(),
            "is_followup":      is_followup,
            "episode_state":    self._ep_state.dict(),
            "action_cost":      cost,
        }
        return next_obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Full structured state snapshot."""
        current = self._get_current_email()
        return {
            "task_id":            self.task_id,
            "step":               self._ep_state.step,
            "emails_remaining":   self._emails_remaining(),
            "cumulative_reward":  self._cumulative_reward,
            "done":               self._done,
            "current_email_id":   current["email_id"] if current else None,
            "n_steps_completed":  len(self._step_rewards),
            "inbox_health":       self._ep_state.inbox_health,
            "user_satisfaction":  self._ep_state.user_satisfaction,
            "unresolved_threads": self._ep_state.unresolved_threads,
            "consecutive_errors": self._ep_state.consecutive_errors,
            "total_cost":         self._ep_state.total_cost,
            "ignored_urgent":     self._ep_state.ignored_urgent,
            "injections_queued":  len(self._injection_queue),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Episode Summary
    # ─────────────────────────────────────────────────────────────────────────

    def episode_result(self) -> EpisodeResult:
        if not self._step_rewards:
            return EpisodeResult(
                task_id=self.task_id, total_steps=0, cumulative_reward=0.0,
                final_score=0.0, per_step_rewards=[], grader_breakdown={},
                final_state=self._ep_state.dict(),
            )
        n = len(self._step_rewards)
        avg = lambda key: round(sum(getattr(r, key) for r in self._step_rewards) / n, 4)

        avg_total = avg("total_reward")

        # Final score: mean reward clipped to [0,1], modulated by final inbox health
        health_modifier = 0.85 + 0.15 * self._ep_state.inbox_health
        final_score = max(0.0, min(1.0, avg_total * health_modifier))

        return EpisodeResult(
            task_id           = self.task_id,
            total_steps       = n,
            cumulative_reward = self._cumulative_reward,
            final_score       = round(final_score, 4),
            per_step_rewards  = self._step_rewards,
            grader_breakdown  = {
                "avg_classification":  avg("classification_score"),
                "avg_priority":        avg("priority_score"),
                "avg_action":          avg("action_score"),
                "avg_response":        avg("response_score"),
                "avg_total_reward":    avg_total,
                "final_inbox_health":  self._ep_state.inbox_health,
                "final_user_sat":      self._ep_state.user_satisfaction,
                "total_action_cost":   self._ep_state.total_cost,
                "ignored_urgent":      self._ep_state.ignored_urgent,
                "unresolved_threads":  len(self._ep_state.unresolved_threads),
            },
            final_state = self._ep_state.dict(),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: Long-term State Updates
    # ─────────────────────────────────────────────────────────────────────────

    def _update_episode_state(
        self,
        action:        AgentAction,
        grader_result: Dict[str, float],
        ground_truth:  Dict[str, Any],
        reward:        StepReward,
    ) -> None:
        cls_score   = grader_result.get("classification_score", 0.0)
        rsp_score   = grader_result.get("response_score", 0.0)
        act_score   = grader_result.get("action_score", 0.0)
        true_label  = ground_truth["label"]
        pred_action = action.action_type.value

        s = self._ep_state

        # ── Inbox health ─────────────────────────────────────────────────
        if reward.total_reward >= 0.85:
            s.inbox_health = min(1.0, round(s.inbox_health + INBOX_HEALTH_GAIN_PERFECT, 4))
        if cls_score < 0.5:
            s.inbox_health = max(0.0, round(s.inbox_health - INBOX_HEALTH_HIT_WRONG_CLASS, 4))
        if true_label == "urgent" and pred_action in ("ignore", "defer"):
            s.inbox_health = max(0.0, round(s.inbox_health - INBOX_HEALTH_HIT_IGNORE_URGENT, 4))
            s.ignored_urgent += 1
        if rsp_score < 0.3 and pred_action in ("respond", "escalate"):
            s.inbox_health = max(0.0, round(s.inbox_health - INBOX_HEALTH_HIT_BAD_RESPONSE, 4))

        # ── User satisfaction ─────────────────────────────────────────────
        if rsp_score >= 0.75 and pred_action in ("respond", "escalate"):
            s.user_satisfaction = min(1.0, round(s.user_satisfaction + USER_SAT_GAIN_GOOD_RESPONSE, 4))
        elif rsp_score < 0.35 and pred_action in ("respond", "escalate"):
            s.user_satisfaction = max(0.0, round(s.user_satisfaction - USER_SAT_HIT_BAD_RESPONSE, 4))
        if act_score < 0:
            s.user_satisfaction = max(0.0, round(s.user_satisfaction - USER_SAT_HIT_WRONG_ACTION, 4))

        # ── Consecutive error streak ──────────────────────────────────────
        if cls_score < 0.5:
            s.consecutive_errors += 1
        else:
            s.consecutive_errors = 0   # reset streak on correct classification

        # ── Unresolved threads ────────────────────────────────────────────
        email_id = ground_truth.get("email_id", "")
        if pred_action in ("defer", "ignore") and true_label not in ("spam", "promotions", "newsletter"):
            if email_id and email_id not in s.unresolved_threads:
                s.unresolved_threads.append(email_id)
        elif email_id in s.unresolved_threads:
            s.unresolved_threads.remove(email_id)

    def _maybe_inject_followup(
        self,
        action:       AgentAction,
        ground_truth: Dict[str, Any],
        current_email: Dict[str, Any],
    ) -> None:
        """
        Inject consequence emails based on bad decisions.
        These are inserted at position cursor+2 (two steps later).
        """
        pred_action = action.action_type.value
        true_label  = ground_truth["label"]
        true_action = ground_truth["action_type"]
        email_id    = current_email["email_id"]

        # Case 1: Ignored urgent email → angry escalation from exec
        if true_label == "urgent" and pred_action in ("ignore", "defer"):
            followup = _make_followup_urgent(current_email)
            self._injection_queue.append(followup)

        # Case 2: Bad response to finance email → chaser from sender
        elif true_label == "finance" and pred_action == "respond":
            rsp = (action.response_text or "").strip()
            if len(rsp.split()) < 10:
                followup = _make_followup_finance_chaser(current_email)
                self._injection_queue.append(followup)

        # Case 3: Deferred non-trivial email → resurfaces with higher stakes
        elif pred_action == "defer" and true_label not in ("spam", "promotions", "newsletter"):
            followup = _make_followup_deferred(current_email)
            self._injection_queue.append(followup)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: Queue / Cursor Management
    # ─────────────────────────────────────────────────────────────────────────

    def _get_current_email(self) -> Optional[Dict[str, Any]]:
        """Return next email: injections take priority over queue."""
        if self._injection_queue:
            return self._injection_queue[0]
        return self._manager.current_email()

    def _advance(self) -> None:
        if self._injection_queue:
            self._injection_queue.pop(0)
        else:
            self._manager.advance()

    def _is_done(self) -> bool:
        return not self._injection_queue and not self._manager.has_next()

    def _emails_remaining(self) -> int:
        return self._manager.emails_remaining + len(self._injection_queue)

    def _build_observation(self) -> Optional[EmailObservation]:
        email = self._get_current_email()
        if email is None:
            return None
        thread = [
            ThreadMessage(sender=m["sender"], timestamp=m["timestamp"], body=m["body"])
            for m in email.get("thread_history", [])
        ]
        return EmailObservation(
            email_id         = email["email_id"],
            subject          = email["subject"],
            sender           = email["sender"],
            recipient        = email["recipient"],
            timestamp        = email["timestamp"],
            body             = email["body"],
            thread_history   = thread,
            has_attachments  = email.get("has_attachments", False),
            task_id          = self.task_id,
            step_number      = self._ep_state.step,
            emails_remaining = self._emails_remaining(),
            episode_state    = copy.deepcopy(self._ep_state),
            is_followup      = email.get("_injected", False),
            followup_reason  = email.get("_followup_reason"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up Email Constructors
# ─────────────────────────────────────────────────────────────────────────────

def _make_followup_urgent(original: Dict[str, Any]) -> Dict[str, Any]:
    oid = original["email_id"]
    return {
        "email_id":   f"{oid}_followup_urgent",
        "subject":    f"RE: {original['subject']} — WHY HAS THIS NOT BEEN ADDRESSED?",
        "sender":     "jeff.skilling@enron.com",
        "recipient":  "user@enron.com",
        "timestamp":  "2001-12-01T08:00:00",
        "body": (
            f"This was flagged as critical and has not been responded to. "
            f"I am now escalating to the board. This is completely unacceptable. "
            f"Original issue: {original['subject']}. Whoever is responsible — "
            f"you have 30 minutes to respond or this goes to HR."
        ),
        "has_attachments": False,
        "thread_history":  [{"sender": original["sender"],
                              "timestamp": original["timestamp"],
                              "body": original["body"][:200]}],
        "_injected":       True,
        "_followup_reason": f"Ignored urgent email: {original['email_id']}",
        "ground_truth": {
            "label":             "urgent",
            "priority":          "urgent",
            "action_type":       "escalate",
            "response_keywords": ["apologize", "addressing", "immediately", "escalate"],
            "difficulty":        "hard",
        }
    }


def _make_followup_finance_chaser(original: Dict[str, Any]) -> Dict[str, Any]:
    oid = original["email_id"]
    return {
        "email_id":   f"{oid}_followup_finance",
        "subject":    f"CHASER: {original['subject']} — payment overdue",
        "sender":     original["sender"],
        "recipient":  "user@enron.com",
        "timestamp":  "2001-12-02T09:00:00",
        "body": (
            f"I sent you an invoice previously and received an inadequate response. "
            f"Payment is now overdue. If this is not resolved today I will need to "
            f"escalate to your accounts department and apply late payment charges. "
            f"Please confirm payment status immediately."
        ),
        "has_attachments": False,
        "thread_history":  [{"sender": original["sender"],
                              "timestamp": original["timestamp"],
                              "body": original["body"][:200]}],
        "_injected":       True,
        "_followup_reason": "Inadequate response to finance email",
        "ground_truth": {
            "label":             "finance",
            "priority":          "urgent",
            "action_type":       "respond",
            "response_keywords": ["payment", "processing", "apologize", "confirm", "accounts"],
            "difficulty":        "hard",
        }
    }


def _make_followup_deferred(original: Dict[str, Any]) -> Dict[str, Any]:
    oid = original["email_id"]
    orig_gt = original.get("ground_truth", {})
    return {
        "email_id":   f"{oid}_followup_deferred",
        "subject":    f"FOLLOW UP: {original['subject']} — still waiting",
        "sender":     original["sender"],
        "recipient":  "user@enron.com",
        "timestamp":  "2001-12-02T14:00:00",
        "body": (
            f"I have not heard back regarding my previous message. "
            f"This matter is now more time-sensitive than before. "
            f"Please respond urgently. The original deadline has now passed."
        ),
        "has_attachments": False,
        "thread_history":  [{"sender": original["sender"],
                              "timestamp": original["timestamp"],
                              "body": original["body"][:200]}],
        "_injected":       True,
        "_followup_reason": f"Deferred email resurfaced: {oid}",
        "ground_truth": {
            "label":    orig_gt.get("label", "work"),
            "priority": "urgent",          # priority escalated because of deferral
            "action_type": orig_gt.get("action_type", "respond"),
            "response_keywords": orig_gt.get("response_keywords", []) + ["apologies", "delay"],
            "difficulty": "hard",
        }
    }
