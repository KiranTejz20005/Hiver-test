import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rows = pd.read_csv("reports/evaluation_rows.csv")
failures = rows[rows["expected_intent"] != rows["predicted_intent"]].copy()
failures["failure_mode"] = "intent overlap or insufficient context"
failures.loc[failures["expected_decision"] != failures["predicted_decision"], "failure_mode"] = "escalation calibration"
failures["hypothesis"] = "The message contains overlapping keywords or requires context not present in the incoming turn."
failures.to_csv("reports/failure_analysis.csv", index=False)
print(f"Saved {len(failures)} failures")
