"""
Reward Engine v3 — Cost-Aware, Long-Term, Shaped Rewards
---------------------------------------------------------
Combines:
  1. Weighted grader scores (spec weights)
  2. Action cost model: escalate(0.05) respond(0.01) clarify(0.015) defer(0.005) archive(0.002)
  3. Penalties: unnecessary escalation, ignored urgent, empty response, repeated mistakes,
                misclassified high-stakes, excessive step count
  4. Bonuses: ambiguous-correct, calibrated confidence, efficient response
  5. Long-term health modifiers: consecutive_errors streak tax, user_satisfaction bonus
"""

from __future__ import annotations

from typing import Dict, Any

from env.models import AgentAction, EpisodeState, StepReward, ACTION_COSTS


# ─────────────────────────────────────────────
# Task weights (spec-mandated)
# ─────────────────────────────────────────────

TASK_WEIGHTS = {
    "easy":   {"c":1.0, "p":0.0, "a":0.0, "r":0.0},
    "medium": {"c":0.60, "p":0.40, "a":0.0, "r":0.0},
    "hard":   {"c":0.40, "p":0.20, "a":0.20, "r":0.20},
}

# Penalties
PEN_ESCALATE_JUNK       = -0.30
PEN_IGNORE_URGENT       = -0.45
PEN_DEFER_URGENT        = -0.30
PEN_EMPTY_RESPONSE      = -0.25
PEN_WRONG_HIGH_STAKES   = -0.20
PEN_ARCHIVE_FINANCE     = -0.15
PEN_REPEAT_ERROR        = -0.08   # per step in consecutive-error streak
PEN_OVERCONFIDENT_WRONG = -0.05

# Bonuses
BON_AMBIGUOUS_CORRECT   =  0.10
BON_CALIBRATED          =  0.05
BON_EFFICIENT_RESPONSE  =  0.05

MAX_R =  1.0
MIN_R = -1.0

HIGH_STAKES = {"urgent", "finance"}


def compute_reward(
    task_id:       str,
    action:        AgentAction,
    grader_result: Dict[str, float],
    ground_truth:  Dict[str, Any],
    episode_state: EpisodeState,
) -> StepReward:

    W = TASK_WEIGHTS.get(task_id, TASK_WEIGHTS["hard"])
    true_label   = ground_truth["label"]
    true_action  = ground_truth["action_type"]
    difficulty   = ground_truth.get("difficulty", "medium")

    cls = grader_result.get("classification_score", 0.0)
    pri = grader_result.get("priority_score",       0.0)
    act = grader_result.get("action_score",         0.0)
    rsp = grader_result.get("response_score",       0.0)

    pred_action = action.action_type.value
    pred_label  = action.label.value
    conf        = action.confidence

    # ── Base reward ───────────────────────────
    base = (W["c"]*cls + W["p"]*pri + W["a"]*max(0.0,act) + W["r"]*rsp)

    # ── Action cost ───────────────────────────
    cost = ACTION_COSTS.get(pred_action, 0.0)

    # ── Penalties ─────────────────────────────
    penalty = 0.0
    fb = []

    if pred_label in ("spam","promotions","newsletter") and pred_action == "escalate":
        penalty += PEN_ESCALATE_JUNK
        fb.append("❌ Escalated junk email")

    if true_label == "urgent" and pred_action == "ignore":
        penalty += PEN_IGNORE_URGENT
        fb.append("❌ Ignored urgent email — severe")

    if true_label == "urgent" and pred_action == "defer":
        penalty += PEN_DEFER_URGENT
        fb.append("❌ Deferred urgent email")

    if pred_action in ("respond","escalate","request_clarification"):
        rlen = len((action.response_text or "").split())
        if rlen < 5:
            penalty += PEN_EMPTY_RESPONSE
            fb.append("❌ Empty/trivial response")

    if true_label in HIGH_STAKES and cls < 0.5:
        penalty += PEN_WRONG_HIGH_STAKES
        fb.append(f"❌ Misclassified high-stakes {true_label}")

    if true_label == "finance" and true_action == "respond" and pred_action == "archive":
        penalty += PEN_ARCHIVE_FINANCE
        fb.append("❌ Archived finance email requiring response")

    if act < 0:
        penalty += act * W["a"]

    # Consecutive error streak tax
    if episode_state.consecutive_errors >= 2:
        streak_pen = PEN_REPEAT_ERROR * min(episode_state.consecutive_errors, 5)
        penalty += streak_pen
        fb.append(f"❌ Error streak x{episode_state.consecutive_errors}")

    # Overconfident wrong answer
    if cls < 0.5 and conf > 0.90:
        penalty += PEN_OVERCONFIDENT_WRONG
        fb.append("❌ Overconfident wrong classification")

    # User satisfaction modifier (bad history → slight penalty multiplier)
    if episode_state.user_satisfaction < 0.5 and penalty < 0:
        penalty *= 1.15

    # ── Bonuses ───────────────────────────────
    bonus = 0.0

    if difficulty == "hard" and cls >= 0.9 and pri >= 0.5:
        bonus += BON_AMBIGUOUS_CORRECT
        fb.append("✅ Correctly handled hard/ambiguous email")

    if cls >= 0.7 and 0.70 <= conf <= 0.92:
        bonus += BON_CALIBRATED
        fb.append("✅ Calibrated confidence")

    if task_id == "hard" and rsp >= 0.75:
        wc = len((action.response_text or "").split())
        if 20 <= wc <= 80:
            bonus += BON_EFFICIENT_RESPONSE
            fb.append("✅ Efficient high-quality response")

    # Inbox health positive feedback loop
    if episode_state.inbox_health > 0.85 and base > 0.8:
        bonus += 0.02

    # ── Hard-email penalty scaling ─────────────
    if difficulty == "hard" and penalty < 0:
        penalty *= 1.20

    # ── Total ─────────────────────────────────
    total = max(MIN_R, min(MAX_R, base + penalty + bonus - cost))

    if not fb:
        if total >= 0.85:  fb = ["✅ Excellent triage decision"]
        elif total >= 0.60: fb = ["✅ Reasonable decision"]
        elif total >= 0.35: fb = ["⚠️  Partial credit"]
        else:               fb = ["❌ Poor decision — review logic"]


    cls = max(0.001, min(0.999, cls))
    pri = max(0.001, min(0.999, pri))
    act = max(0.001, min(0.999, act))
    rsp = max(0.001, min(0.999, rsp))

    return {
        "classification_score": cls,
        "priority_score": pri,
        "action_score": act,
        "response_score": rsp
    }
    # cls   = max(0.001, min(0.999, cls))
    # pri   = max(0.001, min(0.999, pri))
    # act   = max(0.001, min(0.999, act))
    # rsp   = max(0.001, min(0.999, rsp))
    # total = max(0.001, min(0.999, total))

    # return StepReward(
    #     classification_score = round(cls, 4),
    #     priority_score       = round(pri, 4),
    #     action_score         = round(act, 4),
    #     response_score       = round(rsp, 4),
    #     total_reward         = round(total, 4),
    #     penalty              = round(penalty, 4),
    #     action_cost          = round(cost, 4),
    #     feedback             = " | ".join(fb),
    # )

    # return StepReward(
    #     classification_score = round(cls, 4),
    #     priority_score       = round(pri, 4),
    #     action_score         = round(act, 4),
    #     response_score       = round(rsp, 4),
    #     total_reward         = round(total, 4),
    #     penalty              = round(penalty, 4),
    #     action_cost          = round(cost, 4),
    #     feedback             = " | ".join(fb),
    # )
