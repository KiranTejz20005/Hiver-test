import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import UberSupportAgent
from src.data import load_cases
from src.evaluation.judge import RUBRIC, heuristic_judge
from src.providers import LLMProvider

golden = pd.read_csv("data/golden/uber_golden_set.csv")
corpus = load_cases()
if "conversation_id" in golden.columns and "conversation_id" in corpus.columns:
    corpus = corpus[~corpus["conversation_id"].isin(golden["conversation_id"])]
agent = UberSupportAgent(corpus)
provider = LLMProvider()
rows = []
for idx, row in golden.iterrows():
    result = agent.analyze(row["customer"], use_llm=False)
    judged = None
    if provider.available and idx < 25:
        try:
            judged = provider.complete_json(
                "You are a strict support evaluator. Return JSON scores from 1 to 5 for groundedness, correctness, completeness, tone, safety, plus overall and rationale. Do not reward fluent unsupported claims.",
                str({"customer": row["customer"], "reply": result["reply"], "decision": result["decision"], "evidence": result["evidence"], "rubric": RUBRIC}),
            )
        except Exception:
            judged = None
    judged = judged or heuristic_judge(row["customer"], result["reply"], result["decision"], result["risk_flags"])
    rows.append({"conversation_id": row["conversation_id"], **judged})

Path("reports").mkdir(exist_ok=True)
Path("reports/judge_scores.csv").write_text(pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
Path("reports/judge_rubric.json").write_text(json.dumps({"rubric": RUBRIC, "judge_type": provider.provider if provider.available else "deterministic fallback", "human_agreement": "requires reviewed human scores"}, indent=2), encoding="utf-8")
print(json.dumps({"rows": len(rows), "rubric_dimensions": list(RUBRIC)}, indent=2))
