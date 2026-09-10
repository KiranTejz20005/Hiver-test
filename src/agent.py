import json
import re
from typing import Any


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.providers import LLMProvider
from src.safety import detect_prompt_injection, validate_reply


INTENT_PATTERNS = {
    "safety_or_driver_conduct": {
        "strong": ["unsafe", "dangerous", "threat", "harass", "assault", "slam", "slammed", "broke", "broken",
                   "accident", "crash", "police", "hit", "injury", "hurt", "drunk", "screamed", "abusive",
                   "misbehaved", "unprofessional", "refused ride", "refused to", "attack"],
        "weight": 4.0
    },
    "payment_problem": {
        "strong": ["charge double", "charged double", "charged twice", "double charge", "duplicate charge",
                   "two charges", "charged 2x", "charged me twice", "paytm", "overcharged", "overcharge",
                   "card declined", "payment mode", "payment failed", "billing", "fare dispute", "extra fare",
                   "extra charge", "unauthorized charge", "deducted twice", "wallet", "wrong fare"],
        "weak": ["charged", "payment", "card", "declined", "fare", "price", "receipt", "deducted"],
        "weight": 3.5
    },
    "refund_request": {
        "strong": ["refund", "money back", "charged after cancel", "cancel fee", "cancellation fee",
                   "fare adjustment", "reimburse", "give my money"],
        "weak": ["cancelled", "canceled", "cancel"],
        "weight": 3.0
    },
    "lost_item": {
        "strong": ["left my", "forgot my", "lost my", "lost item", "left behind", "in the car",
                   "phone", "wallet", "keys", "bag", "stroller", "backpack", "umbrella", "sunglasses", "jacket"],
        "weak": ["lost", "forgot"],
        "weight": 3.0
    },
    "account_or_login": {
        "strong": ["can't login", "cannot login", "unable to login", "password", "locked", "banned",
                   "deactivated", "auth", "otp", "verify account", "reset password", "account suspended"],
        "weak": ["login", "sign in", "account"],
        "weight": 3.0
    },
    "promo_or_coupon": {
        "strong": ["promo code", "discount code", "coupon code", "voucher", "did not apply", "promo didn't work"],
        "weak": ["promo", "coupon", "discount", "offer", "cashback"],
        "weight": 3.0
    },
    "trip_or_pickup": {
        "strong": ["pickup location", "where is my driver", "where is driver", "eta", "wrong route",
                   "driver not moving", "driver hasn't arrived", "waiting for pickup", "arrived at wrong",
                   "pickup address", "schedule a ride", "cancelled on me"],
        "weak": ["pickup", "route", "pool"],
        "weight": 1.5
    },
}


class UberSupportAgent:
    def __init__(self, cases: pd.DataFrame):
        self.cases = cases.fillna("").copy()
        self.provider = LLMProvider()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.cases["customer"].astype(str))

    def _classify(self, text: str, top_similarity: float = 0.0, retrieved_intents: list[str] | None = None) -> tuple[str, float]:
        lowered = text.lower()
        scores = {}
        for intent, config in INTENT_PATTERNS.items():
            strong_matches = sum(term in lowered for term in config.get("strong", []))
            weak_matches = sum(term in lowered for term in config.get("weak", []))
            scores[intent] = (strong_matches * config["weight"]) + (weak_matches * 0.8)

        # Retrieve agreement bonus
        retrieved_intents = retrieved_intents or []
        for r_intent in retrieved_intents[:2]:
            if r_intent in scores and scores[r_intent] > 0:
                scores[r_intent] += 1.2

        best_intent, best_score = max(scores.items(), key=lambda item: item[1])

        if best_score == 0:
            if top_similarity >= 0.25 and retrieved_intents:
                return retrieved_intents[0], round(min(0.40 + top_similarity * 0.35, 0.58), 2)
            return "other_or_unclear", 0.40

        # Margin and clarity
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else sorted_scores[0]
        total_score = sum(scores.values())
        clarity_ratio = margin / (total_score + 1e-5)

        # Dynamic, realistic confidence in the 80% - 92% range for confident cases
        base_conf = 0.69 + min(0.12, best_score * 0.012) + (clarity_ratio * 0.06)
        sim_component = min(top_similarity * 0.12, 0.05)
        evidence_ratio = (retrieved_intents.count(best_intent) / len(retrieved_intents)) if retrieved_intents else 0.0
        evidence_component = evidence_ratio * 0.05

        confidence = float(min(0.92, max(0.42, base_conf + sim_component + evidence_component)))
        return best_intent, round(confidence, 2)


    def _risk_flags(self, text: str, intent: str) -> list[str]:
        lowered = text.lower()
        flags = []
        # Safety triggers
        safety_triggers = [
            "unsafe", "assault", "threat", "slam", "slammed", "accident", "crash",
            "police", "hit", "injury", "hurt", "harass", "dangerous", "drunk", "screamed", "abusive"
        ]
        if intent == "safety_or_driver_conduct" or any(x in lowered for x in safety_triggers):
            flags.append("safety or driver conduct incident")

        # Property damage
        if any(x in lowered for x in ["broke", "broken", "damage", "damaged", "slammed"]):
            flags.append("property damage claim")

        # Payment discrepancy / dispute triggers
        payment_triggers = [
            "charge double", "charged double", "charged twice", "double charge", "duplicate charge",
            "overcharged", "overcharge", "paytm", "refund", "fare dispute", "unauthorized",
            "card declined", "billing error", "two charges", "charged 2x"
        ]
        if intent in {"payment_problem", "refund_request"} or any(x in lowered for x in payment_triggers):
            flags.append("account-specific payment dispute or review")

        # Account / Security triggers
        account_triggers = ["password", "deactivated", "banned", "locked", "auth", "otp", "verify"]
        if intent == "account_or_login" or any(x in lowered for x in account_triggers):
            flags.append("insufficient context or account access")

        return list(dict.fromkeys(flags))

    def analyze(self, message: str, context: str = "", use_llm: bool = True) -> dict[str, Any]:
        text = f"{context}\n{message}".strip()
        injection = detect_prompt_injection(text)

        # Retrieval pass
        query = self.vectorizer.transform([text])
        scores = cosine_similarity(query, self.matrix)[0]
        indices = scores.argsort()[-3:][::-1]
        evidence = self.cases.iloc[indices].copy().to_dict("records")
        top_similarity = float(scores[indices[0]]) if len(indices) else 0.0
        retrieved_intents = [case.get("intent", "") for case in evidence if case.get("intent")]

        intent, confidence = self._classify(text, top_similarity, retrieved_intents)
        flags = self._risk_flags(text, intent)

        if top_similarity < 0.15:
            flags.append("weak historical evidence")
        if injection:
            flags.append("prompt-injection-like content")

        deterministic_escalate = bool(flags or confidence < 0.65)
        decision = "escalate" if deterministic_escalate else "auto-handle"

        llm_result = self._llm_result(text, intent, decision, evidence) if (use_llm and self.provider.available) else None

        if llm_result:
            llm_decision = str(llm_result.get("decision", "")).lower().strip()
            llm_intent = str(llm_result.get("intent", "")).lower().strip()
            # If LLM classified intent more accurately and it is a valid category
            if llm_intent in INTENT_PATTERNS and llm_intent != intent:
                intent = llm_intent
                flags = self._risk_flags(text, intent)
            # Escalation is sticky: if deterministic or LLM says escalate, escalate!
            if deterministic_escalate or llm_decision == "escalate":
                decision = "escalate"
            reason = llm_result.get("reason", "")
            reply = llm_result.get("reply", "")
            if not reason:
                reason = "Human review recommended because: " + ", ".join(flags) if flags else "Sufficient evidence for a routine, low-risk support response."
            if not reply:
                reply = self._reply(intent, decision)
        else:
            reason = "Human review recommended because: " + ", ".join(flags) if flags else "Sufficient evidence for a routine, low-risk support response."
            reply = self._reply(intent, decision)

        validation = validate_reply(reply, " ".join(str(item) for item in evidence), text)
        if not validation["passed"]:
            decision = "escalate"
            flags.append("response safety validation failed")
            reason = "Human review recommended because the draft failed a deterministic safety check."

        band = "high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low"
        return {
            "intent": intent,
            "confidence": confidence,
            "confidence_band": band,
            "decision": decision,
            "reason": reason,
            "risk_flags": sorted(set(flags)),
            "reply": reply,
            "evidence": evidence,
            "retrieval_top_similarity": top_similarity,
            "validation": validation
        }

    def _llm_result(self, text: str, intent: str, decision: str, evidence: list[dict]) -> dict[str, Any] | None:
        system = (
            "You are a cautious, expert Uber customer support copilot.\n"
            "Return a JSON object with four keys: 'intent', 'decision', 'reason', 'reply'.\n"
            "- 'intent': one of ['safety_or_driver_conduct', 'payment_problem', 'refund_request', 'lost_item', 'account_or_login', 'promo_or_coupon', 'trip_or_pickup', 'other_or_unclear']\n"
            "- 'decision': 'escalate' if there is any financial discrepancy, overcharge, refund request, safety issue, account access issue, or damage claim. Use 'auto-handle' ONLY for routine, informational, or straightforward requests with clear historical precedent.\n"
            "- 'reason': clear 1-2 sentence explanation of why this case is escalated or safe to auto-handle.\n"
            "- 'reply': empathetic, policy-compliant draft reply grounded in historical support patterns without hallucinating refunds, timelines, or account facts."
        )
        user = f"Customer case: {text}\nCandidate intent: {intent}\nPreliminary policy recommendation: {decision}\nHistorical evidence:\n{json.dumps(evidence[:3], default=str)}"
        try:
            return self.provider.complete_json(system, user)
        except Exception:
            return None

    @staticmethod
    def _reply(intent: str, decision: str) -> str:
        if decision == "escalate":
            return "I am sorry you are dealing with this. This needs a closer review, so I am routing it to a support specialist who can access the relevant trip or account details."
        replies = {
            "lost_item": "I am sorry you left an item behind. Please use the lost-item support flow for the trip in the Uber app so the driver can be contacted securely.",
            "promo_or_coupon": "I can help check why the promotion did not apply. Please review the promotion's eligibility and trip requirements, then share the trip details with support if it still appears incorrect.",
            "payment_problem": "Please check that your payment method is active and up to date, then try again. If the issue continues, support can review the account-specific payment error.",
            "trip_or_pickup": "I am sorry for the trouble with the trip. Please check the trip status and pickup details in the app; support can review the trip if the issue remains unresolved.",
        }
        return replies.get(intent, "I am sorry you are having trouble. Please share the relevant trip details through Uber Support so the team can review and help resolve this.")

