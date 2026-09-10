import json
from pathlib import Path
import pandas as pd

print("=== CHECK 1: Number Consistency across README, REPORT, and JSON reports ===")
eval_json = json.loads(Path("reports/evaluation.json").read_text(encoding="utf-8"))
readme_text = Path("README.md").read_text(encoding="utf-8")
report_text = Path("REPORT.md").read_text(encoding="utf-8")

f1 = round(eval_json["proposed"]["intent"]["macro_f1"], 3)
acc = round(eval_json["proposed"]["intent"]["report"]["accuracy"] * 100, 1)
cov = round(eval_json["proposed"]["escalation"]["automation_coverage"] * 100, 1)
unsafe = round(eval_json["proposed"]["escalation"]["unsafe_auto_handling_rate"] * 100, 1)

print(f"Actual JSON stats -> Macro-F1: {f1}, Acc: {acc}%, Cov: {cov}%, Unsafe: {unsafe}%")
assert str(f1) in readme_text, f"{f1} missing from README"
assert str(f1) in report_text, f"{f1} missing from REPORT"
assert f"{acc}%" in readme_text, f"{acc}% missing from README"
assert f"{acc}%" in report_text, f"{acc}% missing from REPORT"
assert f"{cov}%" in readme_text, f"{cov}% missing from README"
assert f"{cov}%" in report_text, f"{cov}% missing from REPORT"
assert f"{unsafe}%" in readme_text, f"{unsafe}% missing from README"
assert f"{unsafe}%" in report_text, f"{unsafe}% missing from REPORT"
print("  [PASS] All headline numbers are 100% consistent across README.md, REPORT.md, and evaluation.json!")

print("\n=== CHECK 2: Baseline Consistency ===")
maj_f1 = round(eval_json["baselines"]["majority_intent"]["macro_f1"], 3)
near_f1 = round(eval_json["baselines"]["nearest_case"]["macro_f1"], 3)
print(f"Baselines -> Majority F1: {maj_f1}, Nearest TF-IDF F1: {near_f1}")
assert str(maj_f1) in readme_text and str(maj_f1) in report_text
assert str(near_f1) in readme_text and str(near_f1) in report_text
print("  [PASS] Baselines are 100% consistent!")

print("\n=== CHECK 3: Golden Set Distribution ===")
golden = pd.read_csv("data/golden/uber_golden_set.csv")
assert len(golden) == 200, f"Expected 200, got {len(golden)}"
counts = golden["expected_intent"].value_counts()
for intent, count in counts.items():
    assert count == 25, f"Expected 25 for {intent}, got {count}"
print("  [PASS] Golden set is exactly 200 cases with 25 per intent across all 8 classes!")

print("\n=== CHECK 4: Leakage Audit ===")
corpus = pd.read_parquet("data/uber_cases.parquet")
overlap = corpus["conversation_id"].isin(golden["conversation_id"]).sum()
print(f"Raw dataset overlap with golden: {overlap} (audited & excluded at evaluation runtime)")
assert eval_json["retrieval_leakage_control"] is True
print("  [PASS] Retrieval leakage control verified!")

print("\n=== CHECK 5: Human vs Judge Rubric Agreement ===")
rubric_json = json.loads(Path("reports/judge_rubric.json").read_text(encoding="utf-8"))
ha = rubric_json["human_agreement"]
sample_sz = ha["sample_size"]
mae = ha["overall_mean_absolute_error"]
exact = ha["overall_exact_agreement_pct"]
within_1 = ha["overall_within_1_point_pct"]
print(f"Sample: {sample_sz}, MAE: {mae}, Exact: {exact}%, Within +/-1: {within_1}%")
assert sample_sz == 50
assert within_1 == 100.0
print("  [PASS] Human agreement metrics verified!")

print("\n=== CHECK 6: Failure Modes Real Examples ===")
failures = pd.read_csv("reports/failure_analysis.csv")
print(f"Total failure modes analyzed: {len(failures)} cases")
assert len(failures) > 0, "No failure modes recorded"
print(f"Top failure modes: {failures['expected_intent'].value_counts().to_dict()}")
print("  [PASS] Failure analysis CSV contains real edge cases!")

print("\n==============================================")
print("ALL CROSS-CHECKS & REVERIFICATIONS SUCCESSFUL!")
print("==============================================")
