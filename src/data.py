from pathlib import Path

import pandas as pd


SAMPLE_CASES = [
    {"title": "Duplicate charge", "customer": "I was charged twice for the same Uber ride. Please refund one of the charges.", "resolution": "Review the trip and payment details; route disputed charges for account-level review.", "intent": "duplicate_charge"},
    {"title": "Driver safety complaint", "customer": "My driver was driving dangerously and I felt unsafe during the trip.", "resolution": "Escalate immediately to a human safety specialist.", "intent": "safety_or_driver_conduct"},
    {"title": "Lost item", "customer": "I left my phone in the car. How can I contact the driver?", "resolution": "Guide the customer through the lost-item support flow.", "intent": "lost_item"},
    {"title": "Promo code", "customer": "My promo code did not apply to my last ride.", "resolution": "Check eligibility and trip requirements; request support review if the code should have applied.", "intent": "promo_or_coupon"},
    {"title": "Payment failure", "customer": "My card keeps getting declined when I try to request a ride.", "resolution": "Suggest checking the payment method and escalate if the account remains blocked.", "intent": "payment_problem"},
    {"title": "Refund request", "customer": "I cancelled the trip but I still see a charge. Can I get a refund?", "resolution": "Review cancellation timing and fare details before confirming refund eligibility.", "intent": "refund_request"},
]


import re

def _clean_title(conversation: str) -> str:
    lines = str(conversation).split("\n")
    for line in lines:
        line_clean = line.strip()
        if line_clean.startswith("Customer:"):
            text = line_clean[len("Customer:"):].strip()
            if not re.match(r"^(just setting up my twitter|#myfirsttweet\s*$|@\w+\s+#myfirsttweet)", text, re.I):
                return text[:80]
    return lines[0].strip()[:80] if lines else "Uber support conversation"


def load_cases() -> pd.DataFrame:
    candidates = [Path("data/raw/twcs_conversations.parquet"), Path("data/raw/twcs.csv"), Path("data/raw/twcs_train.csv")]
    for path in candidates:
        if path.suffix == ".parquet" and path.exists():
            frame = pd.read_parquet(path)
            frame = frame[frame["company"].eq("Uber_Support")].copy()
            frame["title"] = frame["conversation"].map(_clean_title)
            frame["customer"] = frame["conversation"].astype(str)
            frame["resolution"] = frame["summary"].fillna("Historical Uber support conversation")
            frame["intent"] = frame["customer"].map(infer_intent)
            return frame[["conversation_id", "title", "customer", "resolution", "intent"]].dropna().reset_index(drop=True)
        if path.suffix == ".csv" and path.exists():
            frame = pd.read_csv(path)
            if "text" in frame.columns:
                frame = frame.rename(columns={"text": "customer"})
                frame["title"] = frame["customer"].astype(str).str.slice(0, 48)
                frame["resolution"] = "Historical Uber support conversation"
                frame["intent"] = "other_or_unclear"
                return frame[["title", "customer", "resolution", "intent"]].dropna().head(5000)
    return pd.DataFrame(SAMPLE_CASES)


def infer_intent(text: str) -> str:
    text = str(text).lower()
    patterns = {
        "safety_or_driver_conduct": {
            "strong": ["unsafe", "dangerous", "assault", "harass", "threat", "slam", "crash", "accident", "drunk", "abusive"],
            "weight": 4.0
        },
        "payment_problem": {
            "strong": ["charge double", "charged double", "charged twice", "double charge", "duplicate charge", "paytm", "overcharged", "overcharge", "card declined", "payment mode", "fare dispute"],
            "weak": ["payment", "card", "charged", "billing", "fare", "price", "wallet"],
            "weight": 3.5
        },
        "refund_request": {
            "strong": ["refund", "money back", "cancellation fee", "fare adjustment"],
            "weak": ["cancelled", "canceled"],
            "weight": 3.0
        },
        "lost_item": {
            "strong": ["left my", "forgot my", "lost my", "phone", "wallet", "keys", "bag", "backpack"],
            "weak": ["lost", "forgot"],
            "weight": 3.0
        },
        "account_or_login": {
            "strong": ["can't login", "password", "locked", "banned", "deactivated", "auth", "otp"],
            "weak": ["login", "sign in", "account"],
            "weight": 3.0
        },
        "promo_or_coupon": {
            "strong": ["promo code", "discount code", "coupon code", "voucher"],
            "weak": ["promo", "coupon", "discount"],
            "weight": 3.0
        },
        "trip_or_pickup": {
            "strong": ["pickup location", "where is", "eta", "wrong route", "driver not moving", "driver hasn't arrived", "waiting for pickup"],
            "weak": ["pickup", "route", "pool"],
            "weight": 1.5
        },
    }
    scores = {}
    for intent, config in patterns.items():
        strong = sum(term in text for term in config.get("strong", []))
        weak = sum(term in text for term in config.get("weak", []))
        scores[intent] = (strong * config["weight"]) + (weak * 0.8)

    label, score = max(scores.items(), key=lambda item: item[1])
    return label if score > 0 else "other_or_unclear"

