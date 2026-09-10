import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_cases

golden = pd.read_csv("data/golden/uber_golden_set.csv")
corpus = load_cases()
corpus = corpus[~corpus.conversation_id.isin(golden.conversation_id)]
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
x = vectorizer.fit_transform(corpus.customer.astype(str))
model = LogisticRegression(max_iter=300, class_weight="balanced")
model.fit(x, corpus.intent)
pred = model.predict(vectorizer.transform(golden.customer.astype(str)))
result = {"accuracy": float(accuracy_score(golden.expected_intent, pred)), "macro_f1": float(f1_score(golden.expected_intent, pred, average="macro", zero_division=0)), "weighted_f1": float(f1_score(golden.expected_intent, pred, average="weighted", zero_division=0)), "report": classification_report(golden.expected_intent, pred, output_dict=True, zero_division=0), "n": len(golden)}
Path("reports/tfidf_baseline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: result[k] for k in ["accuracy", "macro_f1", "weighted_f1", "n"]}, indent=2))
