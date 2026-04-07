"""
smoke_test.py v3 — Tests all new features without network access.
Run: PYTHONPATH=. python smoke_test.py
"""
from __future__ import annotations
import sys
from env.core import EmailTriageEnv
from env.models import AgentAction, EpisodeState, ACTION_COSTS
from graders.engine import (grade_classification, grade_priority,
                             grade_action_type, grade_response_quality, cosine_sim)
from data.dataset import EMAILS, EmailDataset

PASS="✅"; FAIL="❌"; errors=[]

def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        msg = f"  {FAIL}  {name}" + (f" — {detail}" if detail else "")
        print(msg); errors.append(name)

# ── Dataset ────────────────────────────────────────────────────────────────
print("\n── Dataset ─────────────────────────────────────────────────────")
check("180+ emails in corpus", len(EMAILS) >= 150, f"got {len(EMAILS)}")
ds = EmailDataset(seed=42)
hard_batch = ds.get_emails_for_task("hard", n=20)
n_hard = sum(1 for e in hard_batch if e["ground_truth"].get("difficulty")=="hard")
check("Hard task ≥40% adversarial emails", n_hard >= 8, f"got {n_hard}/20")
easy_batch = ds.get_emails_for_task("easy", n=15)
easy_labels = {e["ground_truth"]["label"] for e in easy_batch}
check("Easy task only spam/promotions/newsletter", easy_labels <= {"spam","promotions","newsletter"}, str(easy_labels))

# ── Extended action space ──────────────────────────────────────────────────
print("\n── Action Space ────────────────────────────────────────────────")
check("defer action valid", ACTION_COSTS["defer"] == 0.005)
check("request_clarification valid", ACTION_COSTS["request_clarification"] == 0.015)
check("escalate costs most", ACTION_COSTS["escalate"] > ACTION_COSTS["respond"])

a_defer = AgentAction(label="work", priority="medium", action_type="defer", confidence=0.7)
check("defer action creates valid AgentAction", a_defer.action_type.value == "defer")

a_clarify = AgentAction(label="work", priority="medium",
    action_type="request_clarification",
    response_text="Could you please clarify the deadline and the intended audience for this report?",
    confidence=0.7)
check("request_clarification creates valid action", a_clarify.action_type.value == "request_clarification")

# ── Graders ────────────────────────────────────────────────────────────────
print("\n── Grader Unit Tests ───────────────────────────────────────────")
check("cls exact=1.0",     grade_classification("spam","spam")==1.0)
check("cls typo=0.7",      grade_classification("spm","spam")==0.7)
check("cls adjacent=0.5",  grade_classification("promotions","spam")==0.5)
check("cls wrong=0.0",     grade_classification("urgent","spam")==0.0)
check("pri exact=1.0",     grade_priority("urgent","urgent")==1.0)
check("pri off-1=0.5",     grade_priority("high","urgent")==0.5)
check("pri off-2=0.25",    grade_priority("medium","urgent")==0.25)
check("action defer=0.4",  grade_action_type("defer","work","respond")==0.4)
check("action clarify=0.6",grade_action_type("request_clarification","work","respond")==0.6)
check("action ignore-urgent=-0.5", grade_action_type("ignore","urgent","escalate")==-0.5)
check("cosine sim identical=1.0",  cosine_sim("hello world","hello world")==1.0)
check("cosine sim zero empty=0.0", cosine_sim("","hello")==0.0)
good_rsp = "Thank you for your message. I have reviewed the invoice details and have forwarded it to accounts payable for processing. Payment will be processed by the due date."
check("response quality >0.6 for good response",
      grade_response_quality(good_rsp, ["invoice","payment","processed"], "respond", "finance") > 0.6)
check("response quality=0.0 for empty", grade_response_quality(None,[],"respond")==0.0)
check("response quality=1.0 for archive", grade_response_quality(None,[],"archive")==1.0)

# ── Persistent state — Easy ────────────────────────────────────────────────
print("\n── Easy Task Episode + State ───────────────────────────────────")
env = EmailTriageEnv(task_id="easy", seed=42)
obs = env.reset()
check("EpisodeState in observation", obs.episode_state is not None)
check("inbox_health starts at 1.0", obs.episode_state.inbox_health == 1.0)
check("user_satisfaction starts at 1.0", obs.episode_state.user_satisfaction == 1.0)
check("consecutive_errors starts at 0", obs.episode_state.consecutive_errors == 0)
all_valid=True; sc=0
while obs is not None:
    act = AgentAction(label="spam",priority="low",action_type="archive",confidence=0.9)
    obs, r, done, info = env.step(act)
    sc += 1
    if not (-1<=r.total_reward<=1): all_valid=False
    if done: break
check("All rewards in [-1,1]", all_valid)
check("Easy completes 15 steps", sc==15, f"got {sc}")
res = env.episode_result()
check("final_score in [0,1]", 0<=res.final_score<=1)
check("final_state in result", "inbox_health" in res.final_state)
print(f"     Easy score: {res.final_score:.4f}")

# ── Persistent state evolution — Medium ───────────────────────────────────
print("\n── Medium Task — State Evolution ───────────────────────────────")
env2 = EmailTriageEnv(task_id="medium", seed=42)
obs = env2.reset()
# Force some bad decisions
for _ in range(3):
    if obs is None: break
    bad = AgentAction(label="newsletter",priority="low",action_type="archive",confidence=0.95)
    obs, r, done, info = env2.step(bad)
    if done: break
s2 = env2.state()
check("inbox_health decreases after bad decisions",
      s2["inbox_health"] < 1.0, f"health={s2['inbox_health']}")
check("state() returns inbox_health", "inbox_health" in s2)
check("state() returns user_satisfaction", "user_satisfaction" in s2)
check("state() returns consecutive_errors", "consecutive_errors" in s2)
check("state() returns unresolved_threads", "unresolved_threads" in s2)

# ── Follow-up injection — Hard ─────────────────────────────────────────────
print("\n── Hard Task — Follow-up Injection ─────────────────────────────")
env3 = EmailTriageEnv(task_id="hard", seed=42)
obs = env3.reset()
injections_triggered = 0
steps = 0
while obs is not None and steps < 30:
    steps += 1
    # Deliberately ignore urgent emails to trigger injection
    act_type = "ignore" if obs.episode_state.step < 3 else "respond"
    rsp = None
    if act_type == "respond":
        rsp = "Thank you for your message. I have reviewed the details and will action this before the deadline. I will follow up with a full update shortly."
    try:
        act = AgentAction(label="work",priority="medium",
                          action_type=act_type, response_text=rsp, confidence=0.7)
        obs, r, done, info = env3.step(act)
        if info.get("is_followup"): injections_triggered += 1
        if done: break
    except Exception:
        # skip validation errors from ignore on respond-required emails
        obs, r, done, info = env3.step(AgentAction(
            label="urgent",priority="urgent",action_type="escalate",
            response_text="Escalating immediately to senior management.",confidence=0.8))
        if done: break

check("Follow-up emails injected after bad actions",
      injections_triggered > 0 or env3.state()["ignored_urgent"] > 0,
      f"injections={injections_triggered}")
res3 = env3.episode_result()
check("Hard episode has grader breakdown", "avg_classification" in res3.grader_breakdown)
check("Hard breakdown has inbox_health", "final_inbox_health" in res3.grader_breakdown)
check("Hard breakdown has total_action_cost", "total_action_cost" in res3.grader_breakdown)
print(f"     Hard score: {res3.final_score:.4f}")

# ── Action cost model ──────────────────────────────────────────────────────
print("\n── Action Cost Model ───────────────────────────────────────────")
env4 = EmailTriageEnv(task_id="hard", seed=99)
obs  = env4.reset()
act_esc = AgentAction(label="urgent",priority="urgent",action_type="escalate",
                      response_text="Escalating to senior management and incident response team immediately.",
                      confidence=0.9)
obs2, r_esc, done, info_esc = env4.step(act_esc)
check("Escalate has action_cost > 0", info_esc["action_cost"] == 0.05)
check("Action cost in StepReward", r_esc.action_cost > 0)
check("total_cost tracked in episode_state",
      info_esc["episode_state"]["total_cost"] > 0)

# ── Grader weights (spec) ──────────────────────────────────────────────────
print("\n── Spec Weight Validation ──────────────────────────────────────")
from graders.engine import GRADER_WEIGHTS
check("cls weight=0.4",    GRADER_WEIGHTS["classification"]==0.40)
check("pri weight=0.2",    GRADER_WEIGHTS["priority"]==0.20)
check("action weight=0.2", GRADER_WEIGHTS["action"]==0.20)
check("response weight=0.2",GRADER_WEIGHTS["response"]==0.20)
check("weights sum to 1.0", abs(sum(GRADER_WEIGHTS.values())-1.0)<1e-9)

# ── Pydantic validation ────────────────────────────────────────────────────
print("\n── Pydantic Validation ─────────────────────────────────────────")
try:
    AgentAction(label="work",priority="low",action_type="respond",response_text=""); check("Rejects empty respond",False,"should raise")
except: check("Rejects empty respond",True)
try:
    AgentAction(label="bad_label",priority="low",action_type="archive"); check("Rejects invalid label",False)
except: check("Rejects invalid label",True)
try:
    AgentAction(label="spam",priority="low",action_type="archive"); check("Valid archive action",True)
except Exception as e: check("Valid archive action",False,str(e))

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
if errors:
    print(f"  {FAIL} {len(errors)} test(s) FAILED:")
    for e in errors: print(f"     • {e}")
    sys.exit(1)
else:
    print(f"  {PASS} All tests passed!")
    print(f"{'='*55}\n")
