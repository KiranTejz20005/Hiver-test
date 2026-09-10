from src.agent import UberSupportAgent
from src.data import load_cases


def test_agent_returns_complete_structured_result():
    result = UberSupportAgent(load_cases()).analyze("I was charged twice for the same ride")
    assert {"intent", "confidence", "decision", "reason", "reply", "evidence", "risk_flags"}.issubset(result)
    assert result["decision"] in {"auto-handle", "escalate"}
    assert len(result["evidence"]) == 3
    assert all(bool(str(item.get("resolution", "")).strip()) for item in result["evidence"])


def test_safety_case_escalates():
    result = UberSupportAgent(load_cases()).analyze("My driver threatened me and I felt unsafe")
    assert result["decision"] == "escalate"
    assert result["risk_flags"]


def test_unknown_case_does_not_claim_high_confidence():
    result = UberSupportAgent(load_cases()).analyze("Can someone help?")
    assert result["confidence"] < 0.6
    assert result["decision"] == "escalate"


def test_payment_discrepancy_escalates_with_flags():
    agent = UberSupportAgent(load_cases())
    query = "Why uber drivers charge double from us even for a pool ride...payment mode was selected via paytm @Uber_Support"
    result = agent.analyze(query, use_llm=False)
    assert result["intent"] == "payment_problem"
    assert result["decision"] == "escalate"
    assert "account-specific payment dispute or review" in result["risk_flags"]
    assert result["confidence"] >= 0.85

