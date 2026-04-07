"""
Task Manager v3 — Extended episode configs with persistent-state support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from data.dataset import EmailDataset


@dataclass
class TaskConfig:
    task_id:          str
    name:             str
    description:      str
    difficulty:       str
    n_emails:         int
    max_steps:        int
    label_space:      List[str]
    action_space:     List[str]
    require_action:   bool = False
    require_response: bool = False


ALL_LABELS  = ["spam","work","finance","promotions","personal","urgent","newsletter","support"]
ALL_ACTIONS = ["respond","archive","escalate","ignore","defer","request_clarification"]

TASK_CONFIGS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy", name="Spam vs. Legitimate Classification",
        description=(
            "Binary classification only. The agent identifies spam "
            "(leet-speak phishing, advance-fee fraud, spear-phishing) "
            "vs legitimate emails. Graded on classification accuracy."
        ),
        difficulty="easy", n_emails=15, max_steps=15,
        label_space=ALL_LABELS, action_space=ALL_ACTIONS,
    ),
    "medium": TaskConfig(
        task_id="medium", name="Multi-class Classification + Priority",
        description=(
            "Classify into 8 categories AND assign priority level. "
            "Includes ambiguous cases where casual language masks urgency. "
            "Persistent state tracks consecutive errors."
        ),
        difficulty="medium", n_emails=20, max_steps=20,
        label_space=ALL_LABELS, action_space=ALL_ACTIONS,
    ),
    "hard": TaskConfig(
        task_id="hard", name="Full Triage Pipeline — Adversarial",
        description=(
            "Full pipeline: classify → prioritise → action → generate response. "
            "≥40% adversarial: downplayed urgency, insider info, multi-intent, "
            "sarcasm. Wrong actions trigger follow-up emails. "
            "Persistent inbox_health + user_satisfaction evolve across steps."
        ),
        difficulty="hard", n_emails=25, max_steps=30,  # extra steps for injections
        label_space=ALL_LABELS, action_space=ALL_ACTIONS,
        require_action=True, require_response=True,
    ),
}


class TaskManager:
    def __init__(self, task_id: str, seed: int = 42):
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task_id}")
        self.config  = TASK_CONFIGS[task_id]
        self.dataset = EmailDataset(seed=seed)
        self._emails: List[Dict[str, Any]] = []
        self._cursor = 0

    def reset(self) -> None:
        self._emails = self.dataset.get_emails_for_task(
            self.config.task_id, n=self.config.n_emails)
        self._cursor = 0

    def has_next(self) -> bool:
        return self._cursor < len(self._emails)

    def current_email(self) -> Optional[Dict[str, Any]]:
        return self._emails[self._cursor] if self.has_next() else None

    def advance(self) -> None:
        self._cursor += 1

    @property
    def emails_remaining(self) -> int:
        return max(0, len(self._emails) - self._cursor)

    @property
    def step_number(self) -> int:
        return self._cursor

    @staticmethod
    def list_tasks() -> List[str]:
        return list(TASK_CONFIGS.keys())
