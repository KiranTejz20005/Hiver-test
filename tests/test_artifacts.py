import json
from pathlib import Path


def test_required_submission_artifacts_exist():
    required = [
        Path("README.md"), Path("REPORT.md"), Path("DECISION_LOG.md"),
        Path("reports/evaluation.json"), Path("reports/judge_scores.csv"),
        Path("reports/failure_analysis.csv"), Path("data/golden/uber_golden_set.csv"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, missing
    report = json.loads(Path("reports/evaluation.json").read_text(encoding="utf-8"))
    assert report["retrieval_leakage_control"] is True
