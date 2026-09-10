import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
golden = pd.read_csv("data/golden/uber_golden_set.csv")
golden["message_length"] = golden["customer"].astype(str).str.len()
golden["length_bucket"] = pd.cut(golden["message_length"], [-1, 80, 300, 10_000], labels=["short", "medium", "long"])
profile = {"total": len(golden), "by_intent": golden.expected_intent.value_counts().to_dict(), "by_decision": golden.expected_decision.value_counts().to_dict(), "length_buckets": golden.length_bucket.value_counts().to_dict(), "unknown_or_abstain": int((golden.expected_intent == "other_or_unclear").sum()), "duplicate_customer_text": int(golden.customer.duplicated().sum()), "human_reviewed": False, "human_review_count": 0}
Path("reports/golden_set_profile.json").write_text(json.dumps(profile, indent=2, default=int), encoding="utf-8")
pd.DataFrame([{"dimension": "intent", "value": k, "count": v} for k, v in profile["by_intent"].items()]).to_csv("reports/golden_set_profile.csv", index=False)
print(json.dumps(profile, indent=2, default=int))
