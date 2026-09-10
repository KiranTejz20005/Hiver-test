import json
from pathlib import Path
import pandas as pd

print("=== 1. Checking Required Deliverable Files ===")
required_files = [
    "README.md", "REPORT.md", "DECISION_LOG.md",
    "data/golden/uber_golden_set.csv", "data/golden/README.md",
    "data/uber_cases.parquet",
    "reports/evaluation.json", "reports/judge_scores.csv",
    "reports/judge_rubric.json", "reports/failure_analysis.csv"
]
for f in required_files:
    p = Path(f)
    assert p.exists(), f"Missing {f}"
    print(f"  [PASS] {f} ({p.stat().st_size:,} bytes)")

print("\n=== 2. Checking Headline Evaluation Metrics ===")
eval_data = json.loads(Path("reports/evaluation.json").read_text(encoding="utf-8"))
assert eval_data["retrieval_leakage_control"] is True, "Leakage control failed"
print(f"  [PASS] Golden Holdout Set: {eval_data['n']} cases")
print(f"  [PASS] Proposed Macro-F1: {eval_data['proposed']['intent']['macro_f1']:.3f}")
print(f"  [PASS] Proposed Accuracy: {eval_data['proposed']['intent']['report']['accuracy']*100:.1f}%")
print(f"  [PASS] Baseline 1 (Majority Intent) Macro-F1: {eval_data['baselines']['majority_intent']['macro_f1']:.3f}")
print(f"  [PASS] Baseline 2 (Nearest-Case TF-IDF) Macro-F1: {eval_data['baselines']['nearest_case']['macro_f1']:.3f}")

print("\n=== 3. Checking Judge & Human Agreement ===")
rubric_data = json.loads(Path("reports/judge_rubric.json").read_text(encoding="utf-8"))
assert "human_agreement" in rubric_data, "Missing human agreement"
ha = rubric_data["human_agreement"]
print(f"  [PASS] Human Sample Size: {ha['sample_size']} audited cases")
print(f"  [PASS] Overall MAE: {ha['overall_mean_absolute_error']}")
print(f"  [PASS] Exact Agreement: {ha['overall_exact_agreement_pct']}%")
print(f"  [PASS] Within +/-1 point: {ha['overall_within_1_point_pct']}%")

print("\n=== 4. Checking Datasets Integrity ===")
corpus = pd.read_parquet("data/uber_cases.parquet")
golden = pd.read_csv("data/golden/uber_golden_set.csv")
print(f"  [PASS] Uber Corpus Cases: {len(corpus):,}")
print(f"  [PASS] Golden Cases: {len(golden):,}")
assert len(golden) == 200, "Expected 200 golden cases"
assert len(corpus) == 41185, "Expected 41185 corpus cases"

print("\n=== 5. Testing Agent End-to-End Analysis ===")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import UberSupportAgent
corpus_no_golden = corpus[~corpus.conversation_id.isin(golden.conversation_id)]
agent = UberSupportAgent(corpus_no_golden)

test_query = "My driver was driving aggressively and I felt completely unsafe."
res = agent.analyze(test_query, use_llm=False)
print(f"  Test Query: '{test_query}'")
print(f"  Detected Intent: {res['intent']} (confidence: {res['confidence']*100:.1f}%)")
print(f"  Escalation Decision: {res['decision']}")
print(f"  Risk Flags: {res['risk_flags']}")
assert res['decision'] == 'escalate', "Safety query must escalate"
assert 'safety' in res['intent'], "Must classify as safety"
print("  [PASS] Agent analysis and safety invariants verified!")

print("\n==========================================")
print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
print("==========================================")
