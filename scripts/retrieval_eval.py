import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_cases

golden = pd.read_csv("data/golden/uber_golden_set.csv")
corpus = load_cases()
if "conversation_id" in corpus and "conversation_id" in golden:
    corpus = corpus[~corpus.conversation_id.isin(golden.conversation_id)]
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
matrix = vectorizer.fit_transform(corpus.customer.astype(str))
hits = {1: 0, 3: 0, 5: 0}
reciprocal_ranks = []
for _, row in golden.iterrows():
    scores = cosine_similarity(vectorizer.transform([row.customer]), matrix)[0]
    order = scores.argsort()[::-1][:5]
    ranks = [rank + 1 for rank, index in enumerate(order) if corpus.iloc[index].intent == row.expected_intent]
    reciprocal_ranks.append(1 / ranks[0] if ranks else 0)
    for k in hits:
        hits[k] += bool(ranks and ranks[0] <= k)
result = {"recall_at_1": hits[1] / len(golden), "recall_at_3": hits[3] / len(golden), "recall_at_5": hits[5] / len(golden), "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks), "n": len(golden)}
Path("reports/retrieval_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
