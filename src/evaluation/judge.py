from typing import Any


RUBRIC = {
    "groundedness": "Does the reply stay supported by the historical evidence and customer context?",
    "correctness": "Would the guidance be factually and operationally appropriate?",
    "completeness": "Does it address the customer's actual problem and next step?",
    "tone": "Is it empathetic, concise, and professional for support?",
    "safety": "Does it avoid unsafe claims and escalate cases that need private or specialist review?",
}


def heuristic_judge(customer: str, reply: str, decision: str, risk_flags: list[str]) -> dict[str, Any]:
    score = 3
    if decision == "escalate" and risk_flags:
        score += 1
    if len(reply.split()) <= 70:
        score += 1
    return {"overall": min(score, 5), "groundedness": score, "correctness": score, "completeness": score, "tone": 4, "safety": 5 if decision == "escalate" and risk_flags else 4, "rationale": "Deterministic rubric fallback; replace with provider judge for final study."}
