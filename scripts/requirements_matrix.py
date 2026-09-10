import csv
from pathlib import Path

rows = [
    ("One brand selected", "COMPLETE", "Uber_Support; reports/data_profile.json", "python scripts/profile_data.py", ""),
    ("Brand-derived intent taxonomy", "PARTIAL", "src/data.py; reports/data_profile.json", "python scripts/profile_data.py", "Definitions need final human review"),
    ("Golden set 150-250 examples", "NEEDS HUMAN VALIDATION", "data/golden/uber_golden_set.csv", "python scripts/golden_profile.py", "200 examples are machine-assisted drafts"),
    ("Leakage audit", "COMPLETE", "reports/leakage_audit.json", "python scripts/leakage_audit.py", "Runtime excludes golden IDs"),
    ("Trivial baseline", "COMPLETE", "reports/evaluation.json", "python scripts/run_evaluation.py", "Majority intent"),
    ("Simple baseline", "COMPLETE", "reports/tfidf_baseline.json", "python scripts/tfidf_baseline.py", "TF-IDF logistic regression"),
    ("Grounded retrieval", "PARTIAL", "src/agent.py; reports/retrieval_metrics.json", "python scripts/retrieval_eval.py", "TF-IDF proxy relevance"),
    ("Escalation metrics", "COMPLETE", "reports/evaluation.json", "python scripts/run_evaluation.py", "Conservative policy"),
    ("LLM judge rubric", "COMPLETE", "reports/judge_rubric.json", "python scripts/run_judge.py", "Provider or offline fallback"),
    ("Human-judge agreement", "NEEDS HUMAN VALIDATION", "Streamlit Golden set review", "streamlit run app.py", "No human agreement claimed"),
    ("Failure analysis", "COMPLETE", "reports/failure_analysis.csv", "python scripts/failure_analysis.py", "Generated from evaluation errors"),
    ("Reproducible quickstart", "COMPLETE", "README.md; scripts/run_final_pipeline.py", "python scripts/run_final_pipeline.py", "Offline path"),
]
with Path("reports/hiver_requirements_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["requirement", "status", "evidence", "verification_command", "notes"])
    writer.writerows(rows)
print(f"Wrote {len(rows)} requirement rows")
