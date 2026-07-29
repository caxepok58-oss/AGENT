class ShortsAgentError(Exception):
    """Base class for all shorts-agent errors."""


class ConfigError(ShortsAgentError):
    """Invalid or missing configuration."""


class ProviderError(ShortsAgentError):
    """A pluggable provider (LLM/TTS/visuals/trends) failed."""


class PolicyViolation(ShortsAgentError):
    """Generated content failed a policy/safety check."""

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("Content blocked by policy guardrails: " + "; ".join(reasons))


class RateLimitExceeded(ShortsAgentError):
    """A configured rate limit (e.g. max uploads/day) would be exceeded."""


class UploadError(ShortsAgentError):
    """YouTube upload failed after retries."""
