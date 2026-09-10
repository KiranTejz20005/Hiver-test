import subprocess
import sys

steps = [
    "scripts/profile_data.py",
    "scripts/create_golden_set.py",
    "scripts/golden_profile.py",
    "scripts/leakage_audit.py",
    "scripts/tfidf_baseline.py",
    "scripts/run_evaluation.py",
    "scripts/retrieval_eval.py",
    "scripts/calibration.py",
    "scripts/run_judge.py",
    "scripts/failure_analysis.py",
    "scripts/requirements_matrix.py",
]
for script in steps:
    subprocess.run([sys.executable, script], check=True)
print("Final pipeline complete. Review the golden set in Streamlit before submission.")
