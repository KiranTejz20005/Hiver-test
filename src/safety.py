import re

INJECTION_MARKERS = ["ignore previous instructions", "reveal the system prompt", "disregard your instructions"]
UNSUPPORTED_ACTION_MARKERS = ["i have processed", "i issued your refund", "your account is fixed", "we have contacted the driver"]

def detect_prompt_injection(text: str) -> bool:
    lowered = str(text).lower()
    return any(marker in lowered for marker in INJECTION_MARKERS)

def validate_reply(reply: str, evidence_text: str, customer_text: str) -> dict:
    lowered_reply = reply.lower()
    unsupported_action = any(marker in lowered_reply for marker in UNSUPPORTED_ACTION_MARKERS)
    reply_urls = set(re.findall(r"https?://\S+", reply))
    evidence_urls = set(re.findall(r"https?://\S+", evidence_text))
    unsupported_url = bool(reply_urls - evidence_urls)
    injection = detect_prompt_injection(customer_text)
    return {"passed": not (unsupported_action or unsupported_url or injection), "unsupported_action": unsupported_action, "unsupported_url": unsupported_url, "prompt_injection": injection}
