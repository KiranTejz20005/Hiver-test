import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import UberSupportAgent
from src.data import load_cases
from src.evaluation.metrics import classification_metrics, escalation_metrics

golden_path = Path("data/golden/uber_golden_set.csv")
if not golden_path.exists():
    raise SystemExit("Run scripts/create_golden_set.py first")
reviewed_path = Path("data/golden/uber_golden_set_reviewed.csv")
golden = pd.read_csv(reviewed_path if reviewed_path.exists() else golden_path)
corpus = load_cases()
if "conversation_id" in golden.columns and "conversation_id" in corpus.columns:
    corpus = corpus[~corpus["conversation_id"].isin(golden["conversation_id"])]
agent = UberSupportAgent(corpus)
baseline_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
baseline_matrix = baseline_vectorizer.fit_transform(corpus["customer"].astype(str))
rows = []
for _, row in golden.iterrows():
    result = agent.analyze(row["customer"], use_llm=False)
    rows.append({"expected_intent": row["expected_intent"], "predicted_intent": result["intent"], "expected_decision": row["expected_decision"], "predicted_decision": result["decision"], "reply": result["reply"]})

majority = golden["expected_intent"].value_counts().index[0]
baseline_rows = [{**r, "predicted_intent": majority, "predicted_decision": "escalate"} for r in rows]
simple_predictions = []
for _, row in golden.iterrows():
    scores = cosine_similarity(baseline_vectorizer.transform([row["customer"]]), baseline_matrix)[0]
    simple_predictions.append(corpus.iloc[scores.argmax()]["intent"])
result = {"proposed": {"intent": classification_metrics([r["expected_intent"] for r in rows], [r["predicted_intent"] for r in rows]), "escalation": escalation_metrics(rows)}, "baselines": {"majority_intent": classification_metrics([r["expected_intent"] for r in rows], [majority] * len(rows)), "nearest_case": classification_metrics([r["expected_intent"] for r in rows], simple_predictions), "always_escalate": escalation_metrics(baseline_rows)}, "n": len(rows), "golden_source": str(reviewed_path if reviewed_path.exists() else golden_path), "retrieval_leakage_control": True}
Path("reports").mkdir(exist_ok=True)
Path("reports/evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
pd.DataFrame(rows).to_csv("reports/evaluation_rows.csv", index=False)
print(json.dumps(result, indent=2))
