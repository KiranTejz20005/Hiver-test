from src.evaluation.metrics import escalation_metrics


def test_escalation_metrics_are_safe_and_bounded():
    rows = [
        {"expected_decision": "escalate", "predicted_decision": "escalate"},
        {"expected_decision": "auto-handle", "predicted_decision": "auto-handle"},
        {"expected_decision": "escalate", "predicted_decision": "auto-handle"},
    ]
    metrics = escalation_metrics(rows)
    assert metrics["unsafe_auto_handling_rate"] == 0.5
    assert 0 <= metrics["automation_coverage"] <= 1
