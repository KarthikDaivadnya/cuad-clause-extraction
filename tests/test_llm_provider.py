import pytest

from src.llm_provider import LLMProvider, LLMError
from src import config


class FlakyProvider(LLMProvider):
    """Fails `fail_times` times, then succeeds. Used to test retry logic
    without sleeping for real or hitting a real API."""

    def __init__(self, fail_times):
        super().__init__(model="flaky-test-model")
        self.fail_times = fail_times
        self.attempts = 0

    def _call(self, system, prompt, max_tokens, temperature):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError("simulated transient failure")
        return "success"


class AlwaysFailsProvider(LLMProvider):
    def __init__(self):
        super().__init__(model="always-fails")
        self.attempts = 0

    def _call(self, system, prompt, max_tokens, temperature):
        self.attempts += 1
        raise RuntimeError("simulated permanent failure")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Don't actually wait during backoff in tests."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


def test_succeeds_after_transient_failures():
    provider = FlakyProvider(fail_times=2)
    assert provider.complete("prompt") == "success"
    assert provider.attempts == 3


def test_raises_llm_error_after_max_retries():
    provider = AlwaysFailsProvider()
    with pytest.raises(LLMError):
        provider.complete("prompt")
    assert provider.attempts == config.MAX_RETRIES
