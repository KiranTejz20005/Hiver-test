import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import UberSupportAgent
from src.data import load_cases

golden = pd.read_csv("data/golden/uber_golden_set.csv")
corpus = load_cases()
if "conversation_id" in golden.columns and "conversation_id" in corpus.columns:
    corpus = corpus[~corpus["conversation_id"].isin(golden["conversation_id"])]
agent = UberSupportAgent(corpus)
rows = []
for _, row in golden.iterrows():
    result = agent.analyze(row.customer, use_llm=False)
    rows.append({"confidence": result["confidence"], "correct": int(result["intent"] == row.expected_intent), "decision": result["decision"]})

frame = pd.DataFrame(rows)
frame["bucket"] = pd.cut(frame.confidence, [-0.01, 0.4, 0.6, 0.8, 1.0], labels=["0.0-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
summary = frame.groupby("bucket", observed=False).agg(count=("correct", "size"), accuracy=("correct", "mean"), mean_confidence=("confidence", "mean")).reset_index()
summary = summary[summary["count"] > 0]
ece = sum(len(group) / len(frame) * abs(group.correct.mean() - group.confidence.mean()) for _, group in frame.groupby("bucket", observed=False) if len(group) > 0)
summary.to_csv("reports/calibration.csv", index=False)
Path("reports/calibration.json").write_text(json.dumps({"ece": float(ece), "buckets": summary.to_dict("records")}, indent=2, default=str), encoding="utf-8")
print(json.dumps({"ece": float(ece)}, indent=2))
