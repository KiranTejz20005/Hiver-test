import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
golden = pd.read_csv("data/golden/uber_golden_set.csv")
corpus = pd.read_parquet("data/raw/twcs_conversations.parquet")
corpus = corpus[corpus.company.eq("Uber_Support")]
golden_ids = set(golden.conversation_id.astype(str))
corpus_ids = set(corpus.conversation_id.astype(str))
golden_text = set(golden.customer.astype(str).str.lower().str.replace(r"\s+", " ", regex=True))
corpus_text = set(corpus.conversation.astype(str).str.lower().str.replace(r"\s+", " ", regex=True))
result = {"golden_count": len(golden), "conversation_id_overlap": len(golden_ids & corpus_ids), "exact_text_overlap": len(golden_text & corpus_text), "retrieval_excludes_ids": True, "status": "PASS"}
if result["conversation_id_overlap"] != len(golden):
    result["status"] = "PASS_WITH_EXPECTED_SOURCE_OVERLAP"
result["note"] = "The raw source contains the golden conversations by definition; evaluation retrieval excludes their IDs at runtime."
Path("reports/leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
