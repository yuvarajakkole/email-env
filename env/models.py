"""
Typed Pydantic models for the Email Triage RL Environment.
v3: Extended action space, persistent episode state, long-term effect fields.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class EmailLabel(str, Enum):
    SPAM       = "spam"
    WORK       = "work"
    URGENT     = "urgent"
    PROMOTIONS = "promotions"
    PERSONAL   = "personal"
    FINANCE    = "finance"
    NEWSLETTER = "newsletter"
    SUPPORT    = "support"

class PriorityLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    URGENT = "urgent"

class ActionType(str, Enum):
    RESPOND               = "respond"
    ARCHIVE               = "archive"
    ESCALATE              = "escalate"
    IGNORE                = "ignore"
    DEFER                 = "defer"
    REQUEST_CLARIFICATION = "request_clarification"

# Action cost model (penalises expensive actions when unnecessary)
ACTION_COSTS: Dict[str, float] = {
    "escalate":              0.050,
    "respond":               0.010,
    "request_clarification": 0.015,
    "defer":                 0.005,
    "archive":               0.002,
    "ignore":                0.000,
}


# ─────────────────────────────────────────────
# Persistent Episode State
# ─────────────────────────────────────────────

class EpisodeState(BaseModel):
    """
    Persistent state that evolves across steps.
    Returned as part of every observation so agents can reason about history.
    """
    inbox_health:       float = Field(1.0,  ge=0.0, le=1.0,
                                      description="Overall inbox health (degrades on bad decisions)")
    user_satisfaction:  float = Field(1.0,  ge=0.0, le=1.0,
                                      description="Sender satisfaction score (drops on poor responses)")
    unresolved_threads: List[str] = Field(default_factory=list,
                                          description="Email IDs that were deferred/ignored but need followup")
    consecutive_errors: int   = Field(0, ge=0,
                                      description="Count of consecutive wrong classifications")
    total_cost:         float = Field(0.0, ge=0.0,
                                      description="Cumulative action cost this episode")
    ignored_urgent:     int   = Field(0, ge=0,
                                      description="Number of urgent emails wrongly ignored/deferred")
    step:               int   = Field(0, ge=0)


# ─────────────────────────────────────────────
# Observation Space
# ─────────────────────────────────────────────

class ThreadMessage(BaseModel):
    sender:    str
    timestamp: str
    body:      str

class EmailObservation(BaseModel):
    """Full observation including email content + persistent episode state."""
    email_id:         str
    subject:          str
    sender:           str
    recipient:        str
    timestamp:        str
    body:             str
    thread_history:   List[ThreadMessage] = Field(default_factory=list)
    has_attachments:  bool  = False
    task_id:          str
    step_number:      int   = 0
    emails_remaining: int   = 0
    # Persistent state visible to agent
    episode_state:    EpisodeState = Field(default_factory=EpisodeState)
    # Follow-up injection flag: True when this email was triggered by a previous bad action
    is_followup:      bool  = False
    followup_reason:  Optional[str] = None


# ─────────────────────────────────────────────
# Action Space
# ─────────────────────────────────────────────

class AgentAction(BaseModel):
    """Extended action space with validation."""
    label:         EmailLabel
    priority:      PriorityLevel
    action_type:   ActionType
    response_text: Optional[str] = None
    confidence:    float         = Field(1.0, ge=0.0, le=1.0)

    @validator("response_text", always=True)
    def response_required_when_responding(cls, v, values):
        at = values.get("action_type")
        if at in (ActionType.RESPOND, ActionType.ESCALATE,
                  ActionType.REQUEST_CLARIFICATION):
            if not v or not v.strip():
                raise ValueError(
                    f"response_text must be non-empty when action_type is '{at}'"
                )
        return v


# ─────────────────────────────────────────────
# Reward Signal
# ─────────────────────────────────────────────

class StepReward(BaseModel):
    classification_score: float = Field(0.0, ge=-1.0, le=1.0)
    priority_score:       float = Field(0.0, ge=-1.0, le=1.0)
    action_score:         float = Field(0.0, ge=-1.0, le=1.0)
    response_score:       float = Field(0.0, ge=-1.0, le=1.0)
    total_reward:         float = Field(0.0, ge=-1.0, le=1.0)
    penalty:              float = Field(0.0, ge=-1.0, le=0.0)
    action_cost:          float = Field(0.0, ge=0.0)
    feedback:             str   = ""


class EpisodeResult(BaseModel):
    task_id:           str
    total_steps:       int
    cumulative_reward: float
    final_score:       float = Field(..., gt=0.0, lt=1.0)
    per_step_rewards:  List[StepReward]
    grader_breakdown:  dict
    final_state:       dict  = Field(default_factory=dict)
