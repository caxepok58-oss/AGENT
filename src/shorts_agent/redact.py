"""Scrub secrets out of text before it is logged.

Some upstream APIs (the YouTube Data API, Pixabay) take their API key as a URL
query parameter rather than an auth header. requests' connection-level
exceptions (``ConnectionError``, ``Timeout``, ``ProxyError``) embed the full
request URL — key included — in their string representation, so logging
``str(exc)`` directly on a network failure would write the raw key straight
into the application's logs.
"""

from __future__ import annotations


def redact(text: str, *secrets: str | None) -> str:
    """Replace any occurrence of a known secret value in ``text`` with a placeholder."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text
