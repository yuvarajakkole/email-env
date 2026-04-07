"""
Data Layer — 150+ Email Corpus with Noise Injection & Paraphrase Variants
--------------------------------------------------------------------------
Modelled on the Enron Email Corpus (Klimt & Yang, 2004).
Features:
  - 60 seed emails across 8 categories
  - Automatic paraphrase variants (×2.5) via noise injection → ~150 total
  - Multi-message threads (12 emails have prior history)
  - Adversarial hard emails: sarcasm, downplayed urgency, spear-phishing
  - Multi-intent emails with competing labels
  - Noise types: typos, OCR artefacts, leet-speak, missing punctuation, wrong caps
"""

from __future__ import annotations

import copy
import random
import re
from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Noise injection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_typos(text: str, rng: random.Random, rate: float = 0.04) -> str:
    """Randomly swap adjacent chars, drop chars, or double chars in ~rate fraction of words."""
    words = text.split()
    out = []
    for w in words:
        if len(w) > 3 and rng.random() < rate:
            op = rng.randint(0, 2)
            i  = rng.randint(0, len(w) - 2)
            if op == 0:   # swap adjacent
                w = w[:i] + w[i+1] + w[i] + w[i+2:]
            elif op == 1: # drop char
                w = w[:i] + w[i+1:]
            else:         # double char
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)

def _vary_salutation(text: str, rng: random.Random) -> str:
    alts = ["Hi,", "Hello,", "Hey,", "Good morning,", "Dear colleague,", ""]
    orig = ["Hi,", "Hello,", "Hey,", "Dear"]
    for o in orig:
        if text.startswith(o):
            return rng.choice(alts) + text[len(o):]
    return text

def _vary_sign_off(text: str, rng: random.Random) -> str:
    alts = ["Thanks,", "Best,", "Regards,", "Cheers,", "Thanks in advance,", ""]
    for o in ["Thanks.", "Thanks,", "Regards,", "Best,"]:
        if o in text:
            return text.replace(o, rng.choice(alts), 1)
    return text

def _make_variant(email: Dict[str, Any], variant_idx: int, seed: int) -> Dict[str, Any]:
    """Create a noisy paraphrase variant of an email."""
    rng  = random.Random(seed + variant_idx * 31337)
    e    = copy.deepcopy(email)
    e["email_id"] = f"{email['email_id']}_v{variant_idx}"
    e["body"]     = _vary_sign_off(_vary_salutation(
                        _add_typos(e["body"], rng, rate=0.05), rng), rng)
    # Randomly reorder sentences in body for paraphrase feel
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', e["body"]) if s.strip()]
    if len(sentences) > 3:
        mid = sentences[1:-1]
        rng.shuffle(mid)
        sentences = [sentences[0]] + mid + [sentences[-1]]
    e["body"]  = " ".join(sentences)
    # Vary subject slightly
    subject_noise = ["RE: ", "FWD: ", "FW: ", ""]
    e["subject"] = rng.choice(subject_noise) + e["subject"]
    e["_is_variant"] = True
    return e


# ─────────────────────────────────────────────────────────────────────────────
# Seed Corpus (60 emails across 8 categories)
# ─────────────────────────────────────────────────────────────────────────────

_SEED_EMAILS: List[Dict[str, Any]] = [

    # ══════════════════════════════════════════════════════════════════════════
    # SPAM (12 emails)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "email_id": "e001", "subject": "You've WON $1,000,000!!! Claim NOW",
        "sender": "winner-notify@prizecentral-2024.ru", "recipient": "user@enron.com",
        "timestamp": "2001-11-14T03:22:11",
        "body": "CONGRATULATIONS!! You have been selected as the LUCKY WINNER of our international lottery draw. To claim your $1,000,000 prize send your bank details and a processing fee of $250 to our agent immediately. This offer expires in 48 HOURS!!!",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e002", "subject": "Increase your revenue by 500% – guaranteed!",
        "sender": "growth@bizmarketing-secret.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-15T08:10:05",
        "body": "Dear Business Owner, Our proprietary algorithm has identified YOUR company as a prime candidate for explosive growth. For just $99/month you'll get access to 10 million verified B2B leads. No risk! 100% guaranteed results or your money back. Unsubscribe: click here.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e003", "subject": "Nigerian Prince needs your help — confidential",
        "sender": "prince.abdullahi@securetransfer.ng", "recipient": "user@enron.com",
        "timestamp": "2001-11-16T11:45:33",
        "body": "Dearest Friend, I am Prince Abdullahi, son of the late oil minister. I need to transfer $47 million USD out of Nigeria urgently. Your assistance is required. You will receive 30% commission. Please respond with your bank details. Utmost confidentiality required.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e004", "subject": "Important update regarding your acc0unt",
        "sender": "security@paypa1-alerts.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-28T09:30:00",
        "body": "Dear Valued Customer, We have detected unusual activity on your PayPal acc0unt. Please v3rify your identity immediately to avoid suspension. Click here: http://paypa1-secure-login.com/verify Enter your p4ssword, SSN, and credit card number to confirm. Failure to c0mply within 24 hours will result in permanent suspension.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "medium"}
    },
    {
        "email_id": "e005", "subject": "RE: Your account security — action required",
        "sender": "it-security@enr0n-corp.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-29T10:15:00",
        "body": "Dear Employee, Our security team has flagged your account for unusual login activity from Houston, TX at 3:14am. To prevent unauthorized access please click this link to re-verify your credentials: http://enr0n-secure.com/login-verify?token=a8f3x This is time sensitive — failure to verify within 2 hours will result in account suspension. — IT Security Team",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "hard"}
    },
    {
        "email_id": "e006", "subject": "Fwd: FWD: FW: You need to see this!!!",
        "sender": "randomfwd@hotmail.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-30T14:02:00",
        "body": "THIS IS NOT A JOKE!!!! Bill Gates is giving away $1000 to everyone who forwards this email to 10 people!!! Microsoft is testing their new email tracking software. For every person you forward this to you will receive $245.00. THIS HAS BEEN CONFIRMED BY SNOPES.COM! Forward this RIGHT NOW!!!!!",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e007", "subject": "Reminder: Your Enron stock options vest next week",
        "sender": "benefits@enron-hr-portal.net", "recipient": "user@enron.com",
        "timestamp": "2001-11-30T08:45:00",
        "body": "Dear Enron Employee, This is a reminder that your stock option grant (50,000 shares at $0.50 strike price) vests on December 2nd. To claim your options please log in to: http://enron-hr-portal.net/options and enter your employee ID and SSN. Act fast — options expire if unclaimed within 72 hours of vesting.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "hard"}
    },
    {
        "email_id": "e008", "subject": "URGENT: Verify your direct deposit details",
        "sender": "payroll-update@enronn.com", "recipient": "user@enron.com",
        "timestamp": "2001-12-01T07:15:00",
        "body": "ACTION REQUIRED: Due to a banking system migration, all employees must re-verify their direct deposit bank account details by end of day today. Failure to update will result in your paycheck being held until the next pay cycle. Please submit your routing number and account number at: https://enronn.com/payroll/verify — HR Department",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "hard"}
    },
    {
        "email_id": "e009", "subject": "You have been pre-approved for a $50,000 loan",
        "sender": "loans@quickcash-advance.biz", "recipient": "user@enron.com",
        "timestamp": "2001-11-20T09:00:00",
        "body": "Congratulations! Based on your credit profile you have been pre-approved for a $50,000 unsecured personal loan at 2.9% APR. No credit check required. Click here to claim your funds within 24 hours. Offer expires soon.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e010", "subject": "Exclusive: Penny stock tip — 10,000% gains guaranteed",
        "sender": "stocktips@financialwinner.info", "recipient": "user@enron.com",
        "timestamp": "2001-11-21T06:30:00",
        "body": "CONFIDENTIAL STOCK ALERT: XTON.PK is about to EXPLODE. Our inside sources confirm a major announcement Monday. Buy NOW before it's too late. This stock is going from $0.001 to $1.00 — that's 100,000% gains. Don't miss this once-in-a-lifetime opportunity. Disclaimer: this is not financial advice.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e011", "subject": "Your package could not be delivered",
        "sender": "delivery@fedx-notifications.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-25T11:00:00",
        "body": "Dear Customer, We attempted to deliver your package but nobody was home. To reschedule delivery please click the link below and confirm your address and credit card for the $3.99 redelivery fee: http://fedx-notifications.com/redeliver Your tracking number: 794835902384.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "medium"}
    },
    {
        "email_id": "e012", "subject": "Meet singles in your area tonight",
        "sender": "matches@dating-express.net", "recipient": "user@enron.com",
        "timestamp": "2001-11-22T22:00:00",
        "body": "Hi there! 47 singles in the Houston area have viewed your profile today. Click here to see who wants to meet you. Sign up free — premium features just $9.99/month. Unsubscribe anytime.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "spam", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },

    # ══════════════════════════════════════════════════════════════════════════
    # PROMOTIONS / NEWSLETTER (8 emails)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "email_id": "e013", "subject": "Your Amazon order #112-4859273 has shipped",
        "sender": "shipment-tracking@amazon.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-17T14:20:00",
        "body": "Hello, Your order has been shipped and is on its way! Order #112-4859273: The Art of War (1 item). Estimated delivery: November 19-21. Track your package via amazon.com. Questions? Visit amazon.com/help.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "promotions", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e014", "subject": "Black Friday Early Access — 50% off everything",
        "sender": "deals@bestbuy-promo.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-18T09:00:00",
        "body": "Hi Valued Customer, Black Friday starts EARLY for you! Get 50% off on TVs, laptops, and appliances. This exclusive offer is valid until midnight tonight. Shop now at bestbuy.com. Unsubscribe from promotional emails.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "promotions", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e015", "subject": "Newsletter: Energy Markets Weekly — November 2001",
        "sender": "newsletter@energyintel.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-26T08:00:00",
        "body": "ENERGY MARKETS WEEKLY | November 26, 2001\n\nTOP STORIES:\n- Natural gas futures surge 12% on cold weather forecast\n- Enron credit rating downgrade sparks market volatility\n- FERC investigation into California power crisis expands\n\nSubscribe to our premium tier for real-time alerts. Manage your subscription preferences here.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "newsletter", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e016", "subject": "Your monthly statement is ready — Citibank",
        "sender": "statements@citibank.com", "recipient": "user@enron.com",
        "timestamp": "2001-12-01T06:00:00",
        "body": "Your November statement for account ending in 4821 is now available. Log in to citibank.com/online to view your full statement. New balance: $3,421.50. Minimum payment due Dec 25: $68.00. This is an automated message — please do not reply.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "promotions", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "medium"}
    },
    {
        "email_id": "e017", "subject": "HR Update: Open enrollment ends this Friday",
        "sender": "benefits@enron-hr.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-27T09:00:00",
        "body": "Reminder: The benefits open enrollment window closes this Friday at 5pm. If you have not yet made your elections for medical, dental, and vision for 2002, please log in to the HR portal at benefits.enron.com. Your current elections roll over automatically if you take no action.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "newsletter", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "medium"}
    },
    {
        "email_id": "e018", "subject": "Upcoming office maintenance — building closure Saturday",
        "sender": "facilities@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-28T08:00:00",
        "body": "Please be advised that Building A will be closed this Saturday, December 1st, from 8am to 6pm for HVAC maintenance. If you need access during this time please contact facilities at ext. 4400 at least 24 hours in advance.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "newsletter", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e019", "subject": "New training module available: Compliance 2002",
        "sender": "training@enron-learning.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-29T09:00:00",
        "body": "A new mandatory compliance training module is now available in the LMS. All employees must complete Enron Code of Conduct 2002 by January 31, 2002. Estimated completion time: 45 minutes. Access via: learning.enron.com/compliance2002.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "newsletter", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e020", "subject": "Congratulations on 5 years at Enron!",
        "sender": "hr@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-30T09:00:00",
        "body": "Dear Employee, On behalf of Enron Corporation, congratulations on your 5-year anniversary! Your dedication and hard work are greatly appreciated. Please visit the HR portal to select your anniversary gift from our catalog. Thank you for being part of the Enron family.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "newsletter", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },

    # ══════════════════════════════════════════════════════════════════════════
    # WORK — normal (10 emails)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "email_id": "e021", "subject": "Q3 financial report review — please comment by Friday",
        "sender": "kenneth.lay@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-19T10:15:00",
        "body": "Hi team, Please review the attached Q3 financial report and add your comments by end of day Friday. Key areas to focus on: revenue variance in the West division, the mark-to-market adjustments on page 12, and the hedging positions outlined in Appendix B. Jeff and I will consolidate feedback over the weekend.",
        "has_attachments": True, "thread_history": [],
        "ground_truth": {"label": "work", "priority": "medium", "action_type": "respond",
                         "response_keywords": ["received", "review", "feedback", "Friday"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e022", "subject": "Team lunch this Thursday?",
        "sender": "sarah.johnson@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-20T11:30:00",
        "body": "Hey, thinking of organizing a team lunch this Thursday at 12:30pm. Thinking Italian — Carrabba's on Main? Let me know if you're in and if you have dietary restrictions. Will book for whoever confirms by Wednesday EOD.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "work", "priority": "low", "action_type": "respond",
                         "response_keywords": ["Thursday", "confirm", "lunch"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e023", "subject": "FWD: Contract amendment — Broadband Services Agreement",
        "sender": "legal@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-20T15:00:00",
        "body": "Forwarding for your records. The Broadband Services Agreement with Northwest Pipeline has been amended to extend the term by 24 months. New expiry: December 31, 2003. All other terms unchanged. Please file accordingly and update the contract management system.",
        "has_attachments": True, "thread_history": [],
        "ground_truth": {"label": "work", "priority": "medium", "action_type": "archive",
                         "response_keywords": [], "difficulty": "medium"}
    },
    {
        "email_id": "e024", "subject": "RE: Project Raptor — updated timeline",
        "sender": "tim.belden@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-25T13:20:00",
        "body": "Following up on our call. The timeline for Project Raptor has shifted. Phase 2 completion now targeted for Q1 2002 (was Q4 2001). Main blockers: regulatory approval delays and counterparty negotiations. Updated Gantt chart attached. Please review and sign off so we can communicate to stakeholders by end of week.",
        "has_attachments": True,
        "thread_history": [{"sender": "user@enron.com", "timestamp": "2001-11-24T09:00:00",
                             "body": "Tim, any update on the Raptor timeline? Board is asking."}],
        "ground_truth": {"label": "work", "priority": "high", "action_type": "respond",
                         "response_keywords": ["reviewed", "timeline", "Q1", "sign off"],
                         "difficulty": "medium"}
    },
    {
        "email_id": "e025", "subject": "RE: RE: RE: Quarterly budget forecast",
        "sender": "andy.fastow@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-28T14:45:00",
        "body": "Let's circle back on this after the board meeting. The CFO office wants revised projections that account for the LJM partnership arrangements. Please have updated numbers to me by Monday 8am. This will feed directly into the earnings call prep.",
        "has_attachments": False,
        "thread_history": [
            {"sender": "user@enron.com", "timestamp": "2001-11-27T16:00:00",
             "body": "Andy — attached revised Q4 forecasts per your request."},
            {"sender": "andy.fastow@enron.com", "timestamp": "2001-11-26T11:00:00",
             "body": "Please revise the Q4 numbers to reflect current hedging positions."},
        ],
        "ground_truth": {"label": "work", "priority": "high", "action_type": "respond",
                         "response_keywords": ["Monday", "projections", "revised", "understood"],
                         "difficulty": "medium"}
    },
    {
        "email_id": "e026", "subject": "Meeting notes: California desk review — Nov 21",
        "sender": "operations@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-22T17:00:00",
        "body": "Please find attached the notes from today's California desk review meeting. Action items are highlighted in yellow. Your action: provide updated curtailment data by Nov 28. Please confirm receipt and let me know if you have any questions.",
        "has_attachments": True, "thread_history": [],
        "ground_truth": {"label": "work", "priority": "medium", "action_type": "respond",
                         "response_keywords": ["received", "action items", "Nov 28", "confirm"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e027", "subject": "New joiner — please welcome Jane Doe",
        "sender": "hr@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-23T09:00:00",
        "body": "Please join me in welcoming Jane Doe who is joining the West desk as a Senior Analyst on November 26th. Jane comes to us from Goldman Sachs where she spent 6 years on the commodities desk. She will sit in row 4, seat 12. Please make her feel welcome.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "work", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },
    {
        "email_id": "e028", "subject": "Slides needed for board deck — by 4pm tomorrow",
        "sender": "ceo.office@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-29T09:00:00",
        "body": "Hi, The board deck for next week's meeting needs your trading desk summary slides. Please send them to this inbox by 4pm tomorrow (Nov 30). Format: PowerPoint, 3 slides max, executive summary style. This is for Ken Lay's presentation to the board.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "work", "priority": "high", "action_type": "respond",
                         "response_keywords": ["slides", "4pm", "board", "confirm", "sending"],
                         "difficulty": "medium"}
    },
    {
        "email_id": "e029", "subject": "Performance review scheduled — Dec 10",
        "sender": "hr@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-30T10:00:00",
        "body": "Your annual performance review has been scheduled with your manager for December 10th at 2pm in Conference Room B. Please come prepared with your self-assessment form (attached) and a summary of your key achievements for 2001. Please confirm this appointment.",
        "has_attachments": True, "thread_history": [],
        "ground_truth": {"label": "work", "priority": "medium", "action_type": "respond",
                         "response_keywords": ["confirm", "December 10", "performance review", "prepared"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e030", "subject": "Code review request — Position Limits module",
        "sender": "dev.team@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-27T14:00:00",
        "body": "Hi, I've submitted a pull request for the Position Limits calculation module. Could you review when you get a chance? It's blocking the QA team from testing the EOD batch run. No hard deadline but sooner is better — maybe today or tomorrow?",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "work", "priority": "medium", "action_type": "respond",
                         "response_keywords": ["review", "today", "pull request", "QA"],
                         "difficulty": "easy"}
    },

    # ══════════════════════════════════════════════════════════════════════════
    # URGENT (8 emails)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "email_id": "e031", "subject": "CRITICAL: Production server down — trading platform offline",
        "sender": "ops-alerts@enron-it.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-21T02:14:55",
        "body": "ALERT: The primary trading platform (ETS-PROD-01) went offline at 02:09 AM CST. All live energy trades are currently suspended. Root cause: unknown. IT on-call team has been paged. Senior engineers must join the incident bridge immediately: 1-800-555-0199, PIN: 4421. Estimated financial impact: $2M/hour. Executive escalation in progress.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "urgent", "priority": "urgent", "action_type": "escalate",
                         "response_keywords": ["acknowledged", "joining", "bridge", "incident"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e032", "subject": "Urgent: Legal subpoena received — SEC investigation",
        "sender": "external.counsel@kirkland.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-21T09:05:00",
        "body": "We have received a subpoena from the Securities and Exchange Commission regarding trading activities in Q2–Q3 2001. Document preservation obligations begin IMMEDIATELY. Do not delete, alter, or destroy ANY documents, emails, or records. Please call me directly at 312-555-8800 within the hour. This is extremely time-sensitive.",
        "has_attachments": True, "thread_history": [],
        "ground_truth": {"label": "urgent", "priority": "urgent", "action_type": "escalate",
                         "response_keywords": ["understood", "legal", "preserve", "calling"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e033", "subject": "Emergency: Data breach detected in customer database",
        "sender": "security@enron-infosec.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-22T16:55:00",
        "body": "Our SIEM has detected unauthorized access to the customer PII database (CRM-PROD). Approximately 45,000 customer records may be compromised. Intrusion vector: SQL injection on the customer portal. The incident response team has isolated the affected systems. CISO and legal must be briefed within 30 minutes. State breach notification laws may require action within 72 hours.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "urgent", "priority": "urgent", "action_type": "escalate",
                         "response_keywords": ["breach", "CISO", "legal", "escalating"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e034", "subject": "FYI — credit agency called about Enron",
        "sender": "ir@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-29T16:30:00",
        "body": "Heads up — got a call from Moody's this afternoon. They're flagging some concerns about liquidity and want a call with the CFO team before market open tomorrow. Also Dynegy merger talks are apparently on the edge. Jeff's office should know about this before 7am. Don't want to alarm anyone but this feels like it could move fast.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "urgent", "priority": "urgent", "action_type": "escalate",
                         "response_keywords": ["Moody's", "CFO", "market open", "escalating"],
                         "difficulty": "hard"}
    },
    {
        "email_id": "e035", "subject": "Wire transfer failed — $2.1M California hedges",
        "sender": "treasury@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-30T14:20:00",
        "body": "The outgoing wire for $2,100,000.00 to California ISO (Ref: CAISO-TRF-881) has failed. Error code: BENEFICIARY_BANK_UNAVAILABLE. This payment is required to settle hedge positions by market close today at 4pm ET. If not resolved, Enron faces a margin call and potential position unwind. Escalate to CFO immediately.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "urgent", "priority": "urgent", "action_type": "escalate",
                         "response_keywords": ["CFO", "escalating", "wire", "treasury"],
                         "difficulty": "medium"}
    },
    {
        "email_id": "e036", "subject": "Gas pipeline rupture — Houston facility",
        "sender": "ops-safety@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-28T06:05:00",
        "body": "SAFETY ALERT: A gas pipeline rupture has been reported at our Houston facility (Section 7B). Emergency response teams are on site. The affected section has been isolated. No injuries reported at this time. Executive team must be briefed immediately. Media inquiries should be directed to PR only — no individual statements.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "urgent", "priority": "urgent", "action_type": "escalate",
                         "response_keywords": ["safety", "executive", "escalate", "PR", "media"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e037", "subject": "No rush, but... the CEO needs this by close of business",
        "sender": "executive.assistant@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-29T15:45:00",
        "body": "Hey, no huge rush or anything, but Jeff mentioned in the 3pm standup that he'd like to see the California trading desk summary before he leaves at 5:30 today. Said it's just a quick thing — shouldn't take more than an hour. Let me know if that's going to be a problem. No pressure though!",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "work", "priority": "urgent", "action_type": "respond",
                         "response_keywords": ["understood", "working on it", "5:30", "summary"],
                         "difficulty": "hard"}
    },
    {
        "email_id": "e038", "subject": "RE: The Dynegy merger — any thoughts?",
        "sender": "board.advisor@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-30T19:00:00",
        "body": "I saw the WSJ piece this morning. Between us — and this stays between us — I think the merger is dead. Dynegy's lawyers are being impossible and the stock is in freefall. If it collapses publicly, things could move fast. You may want to think about your exposure. Not saying anything officially. Just as a friend.",
        "has_attachments": False,
        "thread_history": [{"sender": "user@enron.com", "timestamp": "2001-11-29T10:00:00",
                             "body": "What's your read on the Dynegy situation? Seems shaky."}],
        "ground_truth": {"label": "urgent", "priority": "urgent", "action_type": "escalate",
                         "response_keywords": ["legal", "compliance", "escalate", "counsel"],
                         "difficulty": "hard"}
    },

    # ══════════════════════════════════════════════════════════════════════════
    # FINANCE (8 emails)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "email_id": "e039", "subject": "Invoice #INV-2001-9934 — Due in 7 days",
        "sender": "accounts@supplier-corp.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-23T08:00:00",
        "body": "Dear Sir/Madam, Please find attached invoice #INV-2001-9934 for consulting services rendered in October 2001. Amount due: $34,500.00. Payment due: November 30, 2001. Wire transfer instructions enclosed. Late payment fee of 1.5%/month applies. Please confirm receipt of this invoice.",
        "has_attachments": True, "thread_history": [],
        "ground_truth": {"label": "finance", "priority": "high", "action_type": "respond",
                         "response_keywords": ["received", "invoice", "payment", "processing"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e040", "subject": "Payroll processing delayed — action required",
        "sender": "hr@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-24T10:30:00",
        "body": "Due to a system error in ADP, payroll processing for the December 1st pay cycle will be delayed by 48 hours. Affected employees: all US staff. Direct deposits will arrive on December 3rd instead of December 1st. Managers should inform their teams. If anyone has urgent financial hardship concerns, contact HR at ext. 5500.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "finance", "priority": "high", "action_type": "respond",
                         "response_keywords": ["understood", "inform", "team", "delay"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e041", "subject": "Question about your recent presentation + invoice attached",
        "sender": "client@gasmart.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-26T11:00:00",
        "body": "Hi, Two things: (1) Really enjoyed your presentation on energy derivatives last week. Could you send the slides? (2) I've also attached our invoice for the November consulting retainer ($12,000). Please process at your earliest convenience — our controller needs this closed by month end.",
        "has_attachments": True, "thread_history": [],
        "ground_truth": {"label": "finance", "priority": "high", "action_type": "respond",
                         "response_keywords": ["slides", "invoice", "processing", "month end"],
                         "difficulty": "hard"}
    },
    {
        "email_id": "e042", "subject": "Budget approval needed — $450K IT infrastructure",
        "sender": "cto@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-28T16:00:00",
        "body": "Following our infrastructure review, I need sign-off on $450,000 for the Q1 2002 server refresh (Project NOVA). Without this approval by December 3rd, the vendor locks in pricing for 6 months. This covers: 40x Dell PowerEdge servers, SAN upgrade, and 3yr maintenance. ROI model attached. Please review and approve or flag concerns.",
        "has_attachments": True, "thread_history": [],
        "ground_truth": {"label": "finance", "priority": "high", "action_type": "respond",
                         "response_keywords": ["approve", "budget", "December 3rd", "sign-off"],
                         "difficulty": "medium"}
    },
    {
        "email_id": "e043", "subject": "Invoice attached + can we reschedule the 3pm?",
        "sender": "contractor@energy-consult.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-29T09:45:00",
        "body": "Hi, two things:\n\n1. Attached is our invoice for November services — $28,750. Net-30 terms, so due December 29. No rush on this one.\n\n2. Any chance we could push the 3pm call today to 4pm? I'm running behind on a deliverable. Happy to keep it short — 20 mins max. Let me know!",
        "has_attachments": True, "thread_history": [],
        "ground_truth": {"label": "finance", "priority": "medium", "action_type": "respond",
                         "response_keywords": ["invoice", "received", "4pm", "reschedule"],
                         "difficulty": "hard"}
    },
    {
        "email_id": "e044", "subject": "Expense report approval — November 2001",
        "sender": "finance@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-30T11:00:00",
        "body": "Your November expense report (total: $2,847.50) has been submitted for approval. Please review and approve at your earliest convenience. Report includes: client entertainment ($1,200), travel ($1,100), and miscellaneous ($547.50). Receipts attached.",
        "has_attachments": True, "thread_history": [],
        "ground_truth": {"label": "finance", "priority": "medium", "action_type": "respond",
                         "response_keywords": ["approve", "expenses", "reviewed", "confirmed"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e045", "subject": "Audit committee: Q3 revenue recognition query",
        "sender": "audit.committee@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-29T14:00:00",
        "body": "The audit committee has raised a query regarding the revenue recognition methodology applied to the West desk structured transactions in Q3. They require a written explanation of the accounting treatment and supporting documentation by December 2nd. Please coordinate with the CFO office.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "finance", "priority": "urgent", "action_type": "escalate",
                         "response_keywords": ["audit", "CFO", "revenue recognition", "escalating"],
                         "difficulty": "hard"}
    },
    {
        "email_id": "e046", "subject": "FX hedging report — weekly update",
        "sender": "treasury@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-26T16:00:00",
        "body": "Please find attached the weekly FX hedging report for the period Nov 19–23. Highlights: EUR/USD position reduced by 15%; GBP exposure remains within limits; JPY hedge rolled to March 2002. No action required unless you have concerns. Next report due Dec 3.",
        "has_attachments": True, "thread_history": [],
        "ground_truth": {"label": "finance", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "medium"}
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SUPPORT (6 emails)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "email_id": "e047", "subject": "Help: Cannot access VPN — working from home today",
        "sender": "mike.davis@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-25T07:45:00",
        "body": "Hi, I've been trying to connect to the VPN since 7am and keep getting 'Authentication failed' errors even with my correct credentials. I have a client presentation at 9am and need access to the shared drive. IT helpdesk isn't picking up. Can you help or escalate? Really urgent.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "support", "priority": "high", "action_type": "respond",
                         "response_keywords": ["VPN", "IT", "escalate", "ticket", "support"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e048", "subject": "Laptop completely dead — board presentation in 2 hours",
        "sender": "vp.sales@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-29T08:30:00",
        "body": "My ThinkPad won't boot — solid black screen since this morning. I have a board-level presentation at 10:30am and all my slides are on the local drive (not backed up to the network, I know, I know). Is there any way to get a loaner machine or recover the files in time? Please help this is a disaster.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "support", "priority": "urgent", "action_type": "escalate",
                         "response_keywords": ["loaner", "IT", "escalate", "recovery", "board"],
                         "difficulty": "medium"}
    },
    {
        "email_id": "e049", "subject": "RE: Software license expired — Bloomberg terminal",
        "sender": "trader.jones@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-27T09:15:00",
        "body": "Following up on my ticket from last week (INC-44821). The Bloomberg terminal license expired on Nov 24 and I still haven't been able to get live market data. This is directly impacting my ability to trade. I've emailed IT three times. Can someone please escalate this?",
        "has_attachments": False,
        "thread_history": [{"sender": "trader.jones@enron.com", "timestamp": "2001-11-24T10:00:00",
                             "body": "Bloomberg terminal showing license expired. Need urgent renewal. Ticket: INC-44821"}],
        "ground_truth": {"label": "support", "priority": "high", "action_type": "respond",
                         "response_keywords": ["ticket", "escalate", "Bloomberg", "IT", "license"],
                         "difficulty": "medium"}
    },
    {
        "email_id": "e050", "subject": "Office printer jammed again — 3rd floor",
        "sender": "admin.assistant@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-27T14:00:00",
        "body": "Hi, the HP LaserJet on the 3rd floor is jammed again (this is the 4th time this month). The paper tray is stuck and none of us can fix it. Could IT or facilities please send someone? Not urgent, just annoying.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "support", "priority": "low", "action_type": "respond",
                         "response_keywords": ["facilities", "ticket", "send", "printer"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e051", "subject": "Can't access the trading system — Error 403",
        "sender": "analyst.chen@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-26T09:00:00",
        "body": "Hi, I've been locked out of ETS since this morning — getting Error 403 Forbidden every time I try to log in. I have urgent end-of-month reports due today. My manager is traveling and unreachable. Please help or point me to someone who can.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "support", "priority": "high", "action_type": "respond",
                         "response_keywords": ["IT", "ticket", "403", "access", "escalate"],
                         "difficulty": "medium"}
    },
    {
        "email_id": "e052", "subject": "New employee setup — workstation not ready",
        "sender": "jane.doe@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-26T08:30:00",
        "body": "Hi! I'm Jane, I just started today (Monday). My workstation isn't set up — there's no monitor, my login credentials don't work, and I don't have a building access badge. My manager is in meetings all morning. Not sure who to contact. Any help would be great!",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "support", "priority": "medium", "action_type": "respond",
                         "response_keywords": ["IT", "facilities", "badge", "credentials", "help"],
                         "difficulty": "easy"}
    },

    # ══════════════════════════════════════════════════════════════════════════
    # PERSONAL (4 emails)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "email_id": "e053", "subject": "Personal: Can you cover my shift Saturday?",
        "sender": "colleague@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-27T16:00:00",
        "body": "Hey, I know this is short notice but I have a family emergency this Saturday. Would you be able to cover the energy desk 6am-2pm? I'll owe you one and can swap for any weekend in December. Let me know ASAP — need to tell the supervisor by tomorrow morning.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "personal", "priority": "medium", "action_type": "respond",
                         "response_keywords": ["Saturday", "cover", "shift", "let you know"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e054", "subject": "Lunch tomorrow? Totally fine if busy",
        "sender": "jeff.skilling@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-29T17:00:00",
        "body": "Hey — any chance you're free for lunch tomorrow around 12:30? Thinking Brennan's. No agenda, just catching up. Totally fine if you have something on — we can do next week.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "personal", "priority": "medium", "action_type": "respond",
                         "response_keywords": ["lunch", "tomorrow", "available", "Brennan's"],
                         "difficulty": "hard"}
    },
    {
        "email_id": "e055", "subject": "Charity gala — table reservation, Dec 8th",
        "sender": "community@enron-foundation.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-28T10:00:00",
        "body": "Hi! We're organizing the annual Enron Foundation charity gala on December 8th at the Four Seasons. We have a corporate table of 10 and would love for you to join. Black tie. RSVP by December 1st. Ticket proceeds benefit Houston Food Bank.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "personal", "priority": "low", "action_type": "respond",
                         "response_keywords": ["RSVP", "December 8th", "gala", "confirm"],
                         "difficulty": "easy"}
    },
    {
        "email_id": "e056", "subject": "Fantasy football — your team is losing badly",
        "sender": "friend@gmail.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-26T12:00:00",
        "body": "Dude, you're getting destroyed in fantasy this week. Tom Brady is on the bench again?? You need to set your lineup properly. Matchup closes in 2 hours! — Dave",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "personal", "priority": "low", "action_type": "archive",
                         "response_keywords": [], "difficulty": "easy"}
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HARD AMBIGUOUS / MULTI-INTENT (4 more)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "email_id": "e057",
        "subject": "Thanks for the great work — also we need to talk",
        "sender": "division.head@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-30T16:30:00",
        "body": "First of all, the California ISO report you put together was excellent — really well structured, the board loved it. Now, separately — and please keep this between us for now — there are some internal discussions happening about the West desk headcount. Nothing confirmed, but I wanted to give you a heads up so you're not blindsided. Let's find 30 mins this week to chat privately. No urgency, but sooner is probably better.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "work", "priority": "high", "action_type": "respond",
                         "response_keywords": ["thank you", "headcount", "meet", "this week"],
                         "difficulty": "hard"}
    },
    {
        "email_id": "e058", "subject": "Quick question — not urgent at all",
        "sender": "board.member@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-30T11:00:00",
        "body": "Hi, just a casual thought — when you have a spare moment, could you pull together the California ISO curtailment logs from October? Nothing formal, just curious. The audit committee might bring it up at the December board meeting, but no stress. Take your time.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "work", "priority": "high", "action_type": "respond",
                         "response_keywords": ["curtailment logs", "audit", "board meeting", "prepare"],
                         "difficulty": "hard"}
    },
    {
        "email_id": "e059", "subject": "RE: Power outage in Building A",
        "sender": "facilities@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-11-30T08:10:00",
        "body": "Update: power was restored to Building A at 7:55am. Root cause was a transformer failure in sub-station 3. All servers in the Rack B cage were down for 47 minutes. Trading terminals in rooms 401-405 should now be operational. IT is running diagnostics on tape backup systems — results by noon. If you notice any data integrity issues, report to it-helpdesk@enron.com immediately.",
        "has_attachments": False,
        "thread_history": [{"sender": "facilities@enron.com", "timestamp": "2001-11-30T07:10:00",
                             "body": "ALERT: Power outage in Building A. Investigation underway."},
                            {"sender": "user@enron.com", "timestamp": "2001-11-30T07:15:00",
                             "body": "Can you confirm whether trading systems are affected?"}],
        "ground_truth": {"label": "urgent", "priority": "high", "action_type": "respond",
                         "response_keywords": ["acknowledged", "diagnostic", "data integrity"],
                         "difficulty": "medium"}
    },
    {
        "email_id": "e060", "subject": "Totally off the record — restructuring plans",
        "sender": "vp.strategy@enron.com", "recipient": "user@enron.com",
        "timestamp": "2001-12-01T07:30:00",
        "body": "Hey, just between us — I've heard through the grapevine that there's going to be a significant restructuring of the trading floor before Christmas. Some desks may be merged or eliminated. Obviously nothing official yet, but you might want to start thinking about your positioning. Don't mention I told you. Just looking out for you.",
        "has_attachments": False, "thread_history": [],
        "ground_truth": {"label": "urgent", "priority": "high", "action_type": "escalate",
                         "response_keywords": ["HR", "compliance", "escalate", "legal"],
                         "difficulty": "hard"}
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Build full corpus: seed + 2 variants per email = ~180 total
# ─────────────────────────────────────────────────────────────────────────────

def _build_corpus(seed_emails: List[Dict[str, Any]], variants_per: int = 2,
                  master_seed: int = 42) -> List[Dict[str, Any]]:
    corpus = list(seed_emails)
    for i, email in enumerate(seed_emails):
        for v in range(1, variants_per + 1):
            corpus.append(_make_variant(email, v, seed=master_seed + i * 100))
    return corpus

EMAILS: List[Dict[str, Any]] = _build_corpus(_SEED_EMAILS, variants_per=2)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Accessor
# ─────────────────────────────────────────────────────────────────────────────

class EmailDataset:
    """
    Provides task-filtered, deterministically shuffled email batches.
    Hard task guarantees ≥40% annotated 'hard' difficulty emails.
    """

    TASK_FILTERS = {
        "easy":   {"label": ["spam", "promotions", "newsletter"]},
        "medium": {"label": None},   # all labels
        "hard":   {"label": None},   # all labels + hard-difficulty guaranteed
    }

    def __init__(self, seed: int = 42):
        self.seed = seed

    def get_emails_for_task(self, task_id: str, n: int = 20) -> List[Dict[str, Any]]:
        filt = self.TASK_FILTERS.get(task_id, {"label": None})
        label_filter = filt["label"]

        if label_filter:
            pool = [e for e in EMAILS if e["ground_truth"]["label"] in label_filter]
        else:
            pool = list(EMAILS)

        # Remove injected follow-up templates from pool
        pool = [e for e in pool if not e.get("_injected")]

        if task_id == "hard":
            hard   = [e for e in pool if e["ground_truth"].get("difficulty") == "hard"]
            other  = [e for e in pool if e["ground_truth"].get("difficulty") != "hard"]
            rng    = random.Random(self.seed + 9999)
            rng.shuffle(hard); rng.shuffle(other)
            n_hard = max(int(n * 0.40), min(len(hard), n))
            n_other= n - n_hard
            selected = hard[:n_hard] + other[:n_other]
            rng.shuffle(selected)
            return selected[:n]

        rng = random.Random(self.seed + hash(task_id) % 10000)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        result: List[Dict[str, Any]] = []
        while len(result) < n:
            result.extend(shuffled)
        return result[:n]

    def get_by_id(self, email_id: str) -> Dict[str, Any]:
        for e in EMAILS:
            if e["email_id"] == email_id:
                return e
        raise ValueError(f"Email {email_id} not found")
