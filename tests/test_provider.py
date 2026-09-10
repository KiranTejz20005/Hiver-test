from src.providers import LLMProvider


def test_provider_is_safe_without_credentials(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    provider = LLMProvider()
    assert provider.available is False
    assert provider.complete_json("system", "user") is None
