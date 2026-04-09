"""
Grader Engine v3 — Hybrid Deterministic Scoring
-------------------------------------------------
Weights per spec: classification 0.4, priority 0.2, action 0.2, response 0.2
All graders are pure functions (deterministic).

Scoring tiers:
  Classification: exact(1.0) → typo/fuzzy(0.7) → adjacent(0.5) → substring(0.4) → 0.0
  Priority:       exact(1.0) → off-by-1(0.5) → off-by-2(0.25) → 0.0  + urgency-context bonus
  Action:         exact(1.0) → acceptable-alt(0.5) → harmful(-0.25 to -0.5)
  Response:       TF-IDF cosine(0-0.4) + keyword coverage(0-0.4) + length(0-0.2)
  
Extended action space: defer and request_clarification are scored as
"acceptable" for respond targets to avoid false penalties.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, Any, List, Optional

from env.models import AgentAction


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

LABEL_ADJACENCY: Dict[str, List[str]] = {
    "spam":       ["promotions"],
    "promotions": ["spam", "newsletter"],
    "newsletter": ["promotions"],
    "urgent":     ["work", "support", "finance"],
    "work":       ["urgent", "finance", "support"],
    "finance":    ["work", "urgent"],
    "support":    ["work", "urgent"],
    "personal":   ["work"],
}

PRIORITY_ORDER = ["low", "medium", "high", "urgent"]

MIN_RESPONSE_WORDS  = 15
GOOD_RESPONSE_WORDS = 40

URGENCY_SIGNALS = {
    "urgent", "critical", "emergency", "asap", "immediately", "now",
    "today", "deadline", "breach", "subpoena", "outage", "alert",
    "margin call", "wire failed", "market open", "eod", "close of business",
    "within the hour", "right away", "time-sensitive",
}

STOPWORDS = {
    "i","me","my","we","our","you","your","it","its","the","a","an","and","or",
    "but","in","on","at","to","for","of","with","is","are","was","were","be",
    "been","have","has","had","do","does","did","will","would","could","should",
    "may","might","this","that","these","those","so","as","if","then","than",
    "by","from","up","about","into","through","please","can","not","no","yes",
    "hi","hello","hey","dear","thanks","just","also","here","there",
}

# Spec-mandated weights
GRADER_WEIGHTS = {
    "classification": 0.40,
    "priority":       0.20,
    "action":         0.20,
    "response":       0.20,
}


# ─────────────────────────────────────────────
# TF-IDF Cosine Similarity (pure Python)
# ─────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 2]

def _tfidf(tokens: List[str]) -> Dict[str, float]:
    if not tokens: return {}
    cnt = Counter(tokens); n = len(tokens)
    return {t: c/n for t, c in cnt.items()}

def cosine_sim(a: str, b: str) -> float:
    va, vb = _tfidf(_tokenize(a)), _tfidf(_tokenize(b))
    if not va or not vb: return 0.0
    dot  = sum(va.get(t,0)*vb.get(t,0) for t in va)
    ma   = math.sqrt(sum(v**2 for v in va.values()))
    mb   = math.sqrt(sum(v**2 for v in vb.values()))
    return min(1.0, dot/(ma*mb)) if ma and mb else 0.0


# ─────────────────────────────────────────────
# Fuzzy Label Matching
# ─────────────────────────────────────────────

def _lev(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b)+1))
    for i, ca in enumerate(a):
        nd = [i+1]
        for j, cb in enumerate(b):
            nd.append(min(dp[j]+(ca!=cb), dp[j+1]+1, nd[-1]+1))
        dp = nd
    return dp[-1]

def grade_classification(pred: str, true: str) -> float:
    if pred == true: return 1.0
    if _lev(pred, true) <= 1: return 0.7
    if pred in LABEL_ADJACENCY.get(true, []): return 0.5
    if pred in true or true in pred: return 0.4
    return 0.0


# ─────────────────────────────────────────────
# Priority Grader (context-aware)
# ─────────────────────────────────────────────

def grade_priority(pred: str, true: str,
                   body: str = "", subject: str = "") -> float:
    if pred == true: return 1.0
    try:
        pi, ti = PRIORITY_ORDER.index(pred), PRIORITY_ORDER.index(true)
        base   = {1: 0.5, 2: 0.25}.get(abs(pi-ti), 0.0)
    except ValueError:
        return 0.0
    # Urgency context bonus
    if body or subject:
        combined = (body+" "+subject).lower()
        if any(s in combined for s in URGENCY_SIGNALS):
            if pred in ("high","urgent") and ti >= PRIORITY_ORDER.index("high"):
                base = min(1.0, base + 0.15)
    return round(base, 4)


# ─────────────────────────────────────────────
# Action Type Grader (extended action space)
# ─────────────────────────────────────────────

def grade_action_type(pred: str, true_label: str, true_action: str) -> float:
    if pred == true_action: return 1.0
    # Extended acceptable alternatives
    acceptable = {
        ("archive",              "ignore"):   0.5,
        ("ignore",               "archive"):  0.5,
        ("escalate",             "respond"):  0.5,
        ("respond",              "escalate"): 0.5,
        ("defer",                "respond"):  0.4,   # deferral is suboptimal but not harmful
        ("request_clarification","respond"):  0.6,   # asking for info before responding is smart
        ("request_clarification","escalate"): 0.5,
    }
    pair = (pred, true_action)
    if pair in acceptable: return acceptable[pair]

    # Harmful actions
    if true_label == "urgent" and pred in ("ignore","defer"): return -0.5
    if true_label in ("spam","promotions","newsletter") and pred == "escalate": return -0.25
    if true_action == "respond" and pred == "archive": return -0.15
    return 0.0


# ─────────────────────────────────────────────
# Response Quality Grader
# ─────────────────────────────────────────────

_ANCHORS = {
    "respond_work":     "Thank you for your message. I have reviewed the details and will action this before the deadline. I will follow up with an update shortly.",
    "respond_finance":  "Thank you — I have received the invoice and forwarded it to accounts payable for processing. Payment will be made in accordance with our agreed terms. I confirm receipt.",
    "respond_support":  "I have raised a support ticket and escalated your issue to the IT team for immediate resolution. I will keep you updated on progress and the expected resolution time.",
    "respond_personal": "Thank you for reaching out. I will confirm my availability shortly and get back to you before the deadline.",
    "escalate":         "Acknowledged — this has been escalated immediately to senior management and the on-call team. The incident response protocol is now active. I will monitor and provide updates every 30 minutes until resolved.",
    "request_clarification": "Thank you for your message. To ensure I handle this correctly, could you please clarify the following details? I want to make sure I take the right action before proceeding.",
}

_GENERIC = [r"^ok[\.!]?$",r"^noted[\.!]?$",r"^thanks[\.!]?$",
            r"^sure[\.!]?$",r"^will do[\.!]?$",r"^acknowledged[\.!]?$",
            r"^understood[\.!]?$",r"^received[\.!]?$"]
_META    = [r"as an ai",r"i cannot",r"i'm just an",r"i don't have access"]

def grade_response_quality(
    text: Optional[str],
    keywords: List[str],
    action_type: str,
    label: str = "",
) -> float:
    if action_type not in ("respond","escalate","request_clarification"):
        return 1.0
    if not text or not text.strip(): return 0.0
    t = text.strip()
    if any(re.match(p, t.lower()) for p in _GENERIC): return 0.05
    if any(re.search(p, t.lower()) for p in _META):   return 0.10

    wc = len(t.split())

    # Semantic (0–0.40)
    if action_type == "escalate":           anchor = _ANCHORS["escalate"]
    elif action_type == "request_clarification": anchor = _ANCHORS["request_clarification"]
    elif label == "finance":                anchor = _ANCHORS["respond_finance"]
    elif label == "support":                anchor = _ANCHORS["respond_support"]
    elif label == "personal":               anchor = _ANCHORS["respond_personal"]
    else:                                   anchor = _ANCHORS["respond_work"]

    kw_anchor = " ".join(keywords)
    sem = cosine_sim(t, anchor + " " + kw_anchor) * 0.40

    # Keyword coverage (0–0.40)
    if not keywords:
        kw_score = 0.40
    else:
        tl = t.lower()
        matched = sum(1 for k in keywords if k.lower() in tl)
        kw_score = 0.40 * (matched / len(keywords))

    # Length (0–0.20)
    if wc < 5:    ln = 0.0
    elif wc < 15: ln = 0.06
    elif wc < 40: ln = 0.13
    else:         ln = 0.20

    return min(1.0, round(sem + kw_score + ln, 4))


# ─────────────────────────────────────────────
# Task-level Graders
# ─────────────────────────────────────────────

class EasyGrader:
    """Binary spam vs. legitimate — classification weight 1.0."""
    @staticmethod
    def grade(action: AgentAction, gt: Dict[str, Any]) -> Dict[str, float]:
        is_spam_pred = action.label.value == "spam"
        is_spam_true = gt["label"] == "spam"
        if is_spam_pred == is_spam_true:
            cls = 1.0
        else:
            cls = max(0.0, grade_classification(action.label.value, gt["label"]) * 0.6)
        return {"classification_score": round(cls,4),
                "priority_score": 0.0, "action_score": 0.0,
                "response_score": 0.0, "total": round(max(0.001, min(0.999, total)), 4)}


class MediumGrader:
    """Multi-class classification + priority.  Spec weights: cls 0.4+0.2=0.6, pri 0.4."""
    W = {"c": 0.60, "p": 0.40}
    @staticmethod
    def grade(action: AgentAction, gt: Dict[str, Any]) -> Dict[str, float]:
        cls = grade_classification(action.label.value, gt["label"])
        pri = grade_priority(action.priority.value, gt["priority"])
        total = MediumGrader.W["c"]*cls + MediumGrader.W["p"]*pri
        return {"classification_score": round(cls,4), "priority_score": round(pri,4),
                "action_score": 0.0, "response_score": 0.0,
                "total": round(max(0.001, min(0.999, total)), 4)
                }


class HardGrader:
    """Full pipeline using spec weights: cls 0.4, pri 0.2, action 0.2, response 0.2."""
    W = GRADER_WEIGHTS

    @staticmethod
    def grade(action: AgentAction, gt: Dict[str, Any]) -> Dict[str, float]:
        cls = grade_classification(action.label.value, gt["label"])
        pri = grade_priority(action.priority.value, gt["priority"])
        act = grade_action_type(action.action_type.value, gt["label"], gt["action_type"])
        rsp = grade_response_quality(
            action.response_text, gt.get("response_keywords",[]),
            action.action_type.value, gt["label"])

        act_c   = max(0.0, act)
        total   = (HardGrader.W["classification"]*cls + HardGrader.W["priority"]*pri
                   + HardGrader.W["action"]*act_c + HardGrader.W["response"]*rsp)
        penalty = min(0.0, act) * HardGrader.W["action"]
        total   = max(0.0, total + penalty)

        return {"classification_score": round(cls,4), "priority_score": round(pri,4),
                "action_score": round(act,4), "response_score": round(rsp,4),
                "total": round(max(0.001, min(0.999, total)), 4)}


GRADERS = {"easy": EasyGrader, "medium": MediumGrader, "hard": HardGrader}

def get_grader(task_id: str):
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return GRADERS[task_id]


# """
# Grader Engine v3 — Hybrid Deterministic Scoring
# -------------------------------------------------
# Weights per spec: classification 0.4, priority 0.2, action 0.2, response 0.2
# All graders are pure functions (deterministic).

# Scoring tiers:
#   Classification: exact(1.0) → typo/fuzzy(0.7) → adjacent(0.5) → substring(0.4) → 0.0
#   Priority:       exact(1.0) → off-by-1(0.5) → off-by-2(0.25) → 0.0  + urgency-context bonus
#   Action:         exact(1.0) → acceptable-alt(0.5) → harmful(-0.25 to -0.5)
#   Response:       TF-IDF cosine(0-0.4) + keyword coverage(0-0.4) + length(0-0.2)
  
# Extended action space: defer and request_clarification are scored as
# "acceptable" for respond targets to avoid false penalties.
# """

# from __future__ import annotations

# import math
# import re
# import string
# from collections import Counter
# from typing import Dict, Any, List, Optional

# from env.models import AgentAction


# # ─────────────────────────────────────────────
# # Constants
# # ─────────────────────────────────────────────

# LABEL_ADJACENCY: Dict[str, List[str]] = {
#     "spam":       ["promotions"],
#     "promotions": ["spam", "newsletter"],
#     "newsletter": ["promotions"],
#     "urgent":     ["work", "support", "finance"],
#     "work":       ["urgent", "finance", "support"],
#     "finance":    ["work", "urgent"],
#     "support":    ["work", "urgent"],
#     "personal":   ["work"],
# }

# PRIORITY_ORDER = ["low", "medium", "high", "urgent"]

# MIN_RESPONSE_WORDS  = 15
# GOOD_RESPONSE_WORDS = 40

# URGENCY_SIGNALS = {
#     "urgent", "critical", "emergency", "asap", "immediately", "now",
#     "today", "deadline", "breach", "subpoena", "outage", "alert",
#     "margin call", "wire failed", "market open", "eod", "close of business",
#     "within the hour", "right away", "time-sensitive",
# }

# STOPWORDS = {
#     "i","me","my","we","our","you","your","it","its","the","a","an","and","or",
#     "but","in","on","at","to","for","of","with","is","are","was","were","be",
#     "been","have","has","had","do","does","did","will","would","could","should",
#     "may","might","this","that","these","those","so","as","if","then","than",
#     "by","from","up","about","into","through","please","can","not","no","yes",
#     "hi","hello","hey","dear","thanks","just","also","here","there",
# }

# # Spec-mandated weights
# GRADER_WEIGHTS = {
#     "classification": 0.40,
#     "priority":       0.20,
#     "action":         0.20,
#     "response":       0.20,
# }


# # ─────────────────────────────────────────────
# # TF-IDF Cosine Similarity (pure Python)
# # ─────────────────────────────────────────────

# def _tokenize(text: str) -> List[str]:
#     text = text.lower().translate(str.maketrans("", "", string.punctuation))
#     return [w for w in text.split() if w not in STOPWORDS and len(w) > 2]

# def _tfidf(tokens: List[str]) -> Dict[str, float]:
#     if not tokens: return {}
#     cnt = Counter(tokens); n = len(tokens)
#     return {t: c/n for t, c in cnt.items()}

# def cosine_sim(a: str, b: str) -> float:
#     va, vb = _tfidf(_tokenize(a)), _tfidf(_tokenize(b))
#     if not va or not vb: return 0.0
#     dot  = sum(va.get(t,0)*vb.get(t,0) for t in va)
#     ma   = math.sqrt(sum(v**2 for v in va.values()))
#     mb   = math.sqrt(sum(v**2 for v in vb.values()))
#     return min(1.0, dot/(ma*mb)) if ma and mb else 0.0


# # ─────────────────────────────────────────────
# # Fuzzy Label Matching
# # ─────────────────────────────────────────────

# def _lev(a: str, b: str) -> int:
#     if a == b: return 0
#     if not a: return len(b)
#     if not b: return len(a)
#     dp = list(range(len(b)+1))
#     for i, ca in enumerate(a):
#         nd = [i+1]
#         for j, cb in enumerate(b):
#             nd.append(min(dp[j]+(ca!=cb), dp[j+1]+1, nd[-1]+1))
#         dp = nd
#     return dp[-1]

# def grade_classification(pred: str, true: str) -> float:
#     if pred == true: return 1.0
#     if _lev(pred, true) <= 1: return 0.7
#     if pred in LABEL_ADJACENCY.get(true, []): return 0.5
#     if pred in true or true in pred: return 0.4
#     return 0.0


# # ─────────────────────────────────────────────
# # Priority Grader (context-aware)
# # ─────────────────────────────────────────────

# def grade_priority(pred: str, true: str,
#                    body: str = "", subject: str = "") -> float:
#     if pred == true: return 1.0
#     try:
#         pi, ti = PRIORITY_ORDER.index(pred), PRIORITY_ORDER.index(true)
#         base   = {1: 0.5, 2: 0.25}.get(abs(pi-ti), 0.0)
#     except ValueError:
#         return 0.0
#     # Urgency context bonus
#     if body or subject:
#         combined = (body+" "+subject).lower()
#         if any(s in combined for s in URGENCY_SIGNALS):
#             if pred in ("high","urgent") and ti >= PRIORITY_ORDER.index("high"):
#                 base = min(1.0, base + 0.15)
#     return round(base, 4)


# # ─────────────────────────────────────────────
# # Action Type Grader (extended action space)
# # ─────────────────────────────────────────────

# def grade_action_type(pred: str, true_label: str, true_action: str) -> float:
#     if pred == true_action: return 1.0
#     # Extended acceptable alternatives
#     acceptable = {
#         ("archive",              "ignore"):   0.5,
#         ("ignore",               "archive"):  0.5,
#         ("escalate",             "respond"):  0.5,
#         ("respond",              "escalate"): 0.5,
#         ("defer",                "respond"):  0.4,   # deferral is suboptimal but not harmful
#         ("request_clarification","respond"):  0.6,   # asking for info before responding is smart
#         ("request_clarification","escalate"): 0.5,
#     }
#     pair = (pred, true_action)
#     if pair in acceptable: return acceptable[pair]

#     # Harmful actions
#     if true_label == "urgent" and pred in ("ignore","defer"): return -0.5
#     if true_label in ("spam","promotions","newsletter") and pred == "escalate": return -0.25
#     if true_action == "respond" and pred == "archive": return -0.15
#     return 0.0


# # ─────────────────────────────────────────────
# # Response Quality Grader
# # ─────────────────────────────────────────────

# _ANCHORS = {
#     "respond_work":     "Thank you for your message. I have reviewed the details and will action this before the deadline. I will follow up with an update shortly.",
#     "respond_finance":  "Thank you — I have received the invoice and forwarded it to accounts payable for processing. Payment will be made in accordance with our agreed terms. I confirm receipt.",
#     "respond_support":  "I have raised a support ticket and escalated your issue to the IT team for immediate resolution. I will keep you updated on progress and the expected resolution time.",
#     "respond_personal": "Thank you for reaching out. I will confirm my availability shortly and get back to you before the deadline.",
#     "escalate":         "Acknowledged — this has been escalated immediately to senior management and the on-call team. The incident response protocol is now active. I will monitor and provide updates every 30 minutes until resolved.",
#     "request_clarification": "Thank you for your message. To ensure I handle this correctly, could you please clarify the following details? I want to make sure I take the right action before proceeding.",
# }

# _GENERIC = [r"^ok[\.!]?$",r"^noted[\.!]?$",r"^thanks[\.!]?$",
#             r"^sure[\.!]?$",r"^will do[\.!]?$",r"^acknowledged[\.!]?$",
#             r"^understood[\.!]?$",r"^received[\.!]?$"]
# _META    = [r"as an ai",r"i cannot",r"i'm just an",r"i don't have access"]

# def grade_response_quality(
#     text: Optional[str],
#     keywords: List[str],
#     action_type: str,
#     label: str = "",
# ) -> float:
#     if action_type not in ("respond","escalate","request_clarification"):
#         return 1.0
#     if not text or not text.strip(): return 0.0
#     t = text.strip()
#     if any(re.match(p, t.lower()) for p in _GENERIC): return 0.05
#     if any(re.search(p, t.lower()) for p in _META):   return 0.10

#     wc = len(t.split())

#     # Semantic (0–0.40)
#     if action_type == "escalate":           anchor = _ANCHORS["escalate"]
#     elif action_type == "request_clarification": anchor = _ANCHORS["request_clarification"]
#     elif label == "finance":                anchor = _ANCHORS["respond_finance"]
#     elif label == "support":                anchor = _ANCHORS["respond_support"]
#     elif label == "personal":               anchor = _ANCHORS["respond_personal"]
#     else:                                   anchor = _ANCHORS["respond_work"]

#     kw_anchor = " ".join(keywords)
#     sem = cosine_sim(t, anchor + " " + kw_anchor) * 0.40

#     # Keyword coverage (0–0.40)
#     if not keywords:
#         kw_score = 0.40
#     else:
#         tl = t.lower()
#         matched = sum(1 for k in keywords if k.lower() in tl)
#         kw_score = 0.40 * (matched / len(keywords))

#     # Length (0–0.20)
#     if wc < 5:    ln = 0.0
#     elif wc < 15: ln = 0.06
#     elif wc < 40: ln = 0.13
#     else:         ln = 0.20

#     return min(1.0, round(sem + kw_score + ln, 4))


# # ─────────────────────────────────────────────
# # Task-level Graders
# # ─────────────────────────────────────────────

# class EasyGrader:
#     """Binary spam vs. legitimate — classification weight 1.0."""
#     @staticmethod
#     def grade(action: AgentAction, gt: Dict[str, Any]) -> Dict[str, float]:
#         is_spam_pred = action.label.value == "spam"
#         is_spam_true = gt["label"] == "spam"
#         if is_spam_pred == is_spam_true:
#             cls = 1.0
#         else:
#             cls = max(0.0, grade_classification(action.label.value, gt["label"]) * 0.6)
#         return {"classification_score": round(cls,4),
#                 "priority_score": 0.0, "action_score": 0.0,
#                 "response_score": 0.0, "total": round(max(0.001, min(0.999, cls)), 4)}


# class MediumGrader:
#     """Multi-class classification + priority.  Spec weights: cls 0.4+0.2=0.6, pri 0.4."""
#     W = {"c": 0.60, "p": 0.40}
#     @staticmethod
#     def grade(action: AgentAction, gt: Dict[str, Any]) -> Dict[str, float]:
#         cls = grade_classification(action.label.value, gt["label"])
#         pri = grade_priority(action.priority.value, gt["priority"])
#         total = MediumGrader.W["c"]*cls + MediumGrader.W["p"]*pri
#         return {"classification_score": round(cls,4), "priority_score": round(pri,4),
#                 "action_score": 0.0, "response_score": 0.0, "total": round(max(0.001, min(0.999, cls)), 4)}


# class HardGrader:
#     """Full pipeline using spec weights: cls 0.4, pri 0.2, action 0.2, response 0.2."""
#     W = GRADER_WEIGHTS

#     @staticmethod
#     def grade(action: AgentAction, gt: Dict[str, Any]) -> Dict[str, float]:
#         cls = grade_classification(action.label.value, gt["label"])
#         pri = grade_priority(action.priority.value, gt["priority"])
#         act = grade_action_type(action.action_type.value, gt["label"], gt["action_type"])
#         rsp = grade_response_quality(
#             action.response_text, gt.get("response_keywords",[]),
#             action.action_type.value, gt["label"])

#         act_c   = max(0.0, act)
#         total   = (HardGrader.W["classification"]*cls + HardGrader.W["priority"]*pri
#                    + HardGrader.W["action"]*act_c + HardGrader.W["response"]*rsp)
#         penalty = min(0.0, act) * HardGrader.W["action"]
#         total   = max(0.0, total + penalty)

#         return {"classification_score": round(cls,4), "priority_score": round(pri,4),
#                 "action_score": round(act,4), "response_score": round(rsp,4),
#                 "total": round(max(0.001, min(0.999, cls)), 4)}


# GRADERS = {"easy": EasyGrader, "medium": MediumGrader, "hard": HardGrader}

# def get_grader(task_id: str):
#     if task_id not in GRADERS:
#         raise ValueError(f"Unknown task_id: {task_id}")
#     return GRADERS[task_id]
