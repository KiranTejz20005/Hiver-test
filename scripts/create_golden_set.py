from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_cases

cases = load_cases()
parts = [group.sample(min(len(group), 25), random_state=42) for _, group in cases.groupby("intent")]
golden = pd.concat(parts, ignore_index=True)
golden["expected_decision"] = golden["intent"].isin(["safety_or_driver_conduct", "payment_problem", "refund_request", "account_or_login", "other_or_unclear"]).map({True: "escalate", False: "auto-handle"})
golden["expected_intent"] = golden["intent"]
golden["human_notes"] = ""
Path("data/golden").mkdir(parents=True, exist_ok=True)
target = Path("data/golden/uber_golden_set.csv")
if target.exists():
    print(f"Golden set already exists; preserved {target}")
else:
    golden.to_csv(target, index=False)
    print(f"Wrote {len(golden)} examples")
