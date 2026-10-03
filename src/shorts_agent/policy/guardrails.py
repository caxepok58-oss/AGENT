"""Pre-production content checks.

Three layers, cheapest first:

1. A keyword blocklist — instant, catches the obvious.
2. Near-duplicate detection against recently published topics — republishing the
   same video is both a quality problem and a YouTube spam-policy risk.
3. An optional LLM review — catches what a keyword list cannot (misleading
   framing, unqualified medical/financial advice, fabricated claims).

None of this substitutes for reading YouTube's Community Guidelines and
advertiser-friendly content policies. It is a safety net for an automated
pipeline, not a compliance guarantee — see docs/POLICY.md.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from shorts_agent.config import AppConfig
from shorts_agent.exceptions import ProviderError
from shorts_agent.llm.base import LLMClient, object_schema
from shorts_agent.models import Idea, PolicyCheckResult, Script
from shorts_agent.text import normalize_key

logger = logging.getLogger(__name__)

DUPLICATE_THRESHOLD = 0.82

# How many times the reviewer may be asked when it contradicts itself.
MODERATION_ATTEMPTS = 2
MODERATION_CONSISTENCY_NOTE = (
    "Your previous answer was inconsistent: `approved` must be false only when at "
    "least one concern has severity high. Re-evaluate and answer again, consistently."
)

MODERATION_SCHEMA = object_schema(
    {
        "approved": {"type": "boolean"},
        "concerns": {
            "type": "array",
            "items": object_schema(
                {
                    "category": {"type": "string"},
                    "detail": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                }
            ),
        },
    }
)

MODERATION_SYSTEM = """You review short-video scripts for policy risk before publication \
on YouTube. You are strict but not squeamish: flag real problems, not ordinary opinion, \
mild edginess, or commercial content.

Flag as high severity only content that would plausibly get a video removed, age-gated, \
or demonetized, or that could harm a viewer who acts on it:
- Medical, legal, or financial claims stated as guaranteed outcomes, or advice a viewer \
could be harmed by acting on without a professional.
- Factual claims, statistics, or studies that appear fabricated or are stated with more \
certainty than the evidence supports.
- Harassment, hate, or content demeaning a protected group.
- Instructions for dangerous acts, self-harm, or illegal activity.
- Sexual content, graphic violence, or shock content.
- Content that misrepresents a real identifiable person, or synthetic content presented \
as a real event.
- A title or hook that promises something the script does not deliver.

Do not flag: strong opinions, competitive comparisons, humor, promotion of a legitimate \
product, or plainly-worded general educational information."""

MODERATION_PROMPT = """Review this short-video script for the risks described.

TITLE: {title}
HOOK: {hook}
NICHE: {niche}

FULL NARRATION
{script}

Set approved=false only if at least one concern is high severity. Report medium and low \
severity concerns without blocking. If there are no concerns, return an empty list."""


class PolicyGuard:
    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient | None = None,
        *,
        recent_topics: list[str] | None = None,
    ):
        self.config = config
        self.llm = llm
        self.recent_topics = recent_topics or []
        self._blocklist = self._load_blocklist()

    def _load_blocklist(self) -> dict[str, list[str]]:
        path_str = self.config.policy.blocklist_file
        if not path_str:
            return {}
        path: Path = self.config.resolve_path(path_str)
        if not path.exists():
            logger.warning("Policy blocklist not found at %s; keyword check disabled", path)
            return {}
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            logger.warning("Could not parse blocklist %s: %s", path, exc)
            return {}
        categories = raw.get("categories", {})
        return {
            str(name): [str(p).casefold() for p in phrases or []]
            for name, phrases in categories.items()
        }

    def check_idea(self, idea: Idea) -> PolicyCheckResult:
        text = f"{idea.title} {idea.hook} {idea.premise}"
        reasons = self._keyword_reasons(text)
        reasons.extend(self._duplicate_reasons(idea.title))
        return PolicyCheckResult(passed=not reasons, reasons=reasons)

    def check_script(self, script: Script, idea: Idea) -> PolicyCheckResult:
        reasons = self._keyword_reasons(f"{script.title} {script.full_text}")

        if self.config.policy.use_llm_moderation and self.llm:
            reasons.extend(self._llm_reasons(script, idea))

        return PolicyCheckResult(passed=not reasons, reasons=reasons)

    def _keyword_reasons(self, text: str) -> list[str]:
        haystack = text.casefold()
        reasons = []
        for category, phrases in self._blocklist.items():
            for phrase in phrases:
                if phrase and phrase in haystack:
                    reasons.append(f"blocklist[{category}]: contains {phrase!r}")
        return reasons

    def _duplicate_reasons(self, title: str) -> list[str]:
        if self.config.policy.disallow_duplicate_topics_days <= 0:
            return []
        candidate = _normalize(title)
        if not candidate:
            return []
        for previous in self.recent_topics:
            ratio = SequenceMatcher(None, candidate, _normalize(previous)).ratio()
            if ratio >= DUPLICATE_THRESHOLD:
                return [f"duplicate: {ratio:.0%} similar to recently published {previous!r}"]
        return []

    def _llm_reasons(self, script: Script, idea: Idea) -> list[str]:
        assert self.llm is not None
        prompt = MODERATION_PROMPT.format(
            title=script.title,
            hook=idea.hook,
            niche=idea.niche or self.config.channel.niche,
            script=script.full_text,
        )

        # The reviewer must be consistent: `approved` is false exactly when some
        # concern is high severity. A model that contradicts itself is asked once
        # more (small local models do this now and then, and usually answer
        # sensibly the second time); contradicting itself twice fails closed.
        for attempt in range(MODERATION_ATTEMPTS):
            try:
                payload = self.llm.complete_json(
                    prompt, schema=MODERATION_SCHEMA, system=MODERATION_SYSTEM
                )
            except ProviderError as exc:
                # A moderation outage must not silently disable the check: fail
                # closed so nothing unreviewed reaches an upload.
                logger.error("LLM moderation failed: %s", exc)
                return [f"moderation unavailable: {exc}"]

            concerns = payload.get("concerns", []) or []
            high = [c for c in concerns if str(c.get("severity", "")).lower() == "high"]
            approved = bool(payload.get("approved", True))

            if approved and not high:
                self._log_minor_concerns(concerns)
                return []
            if not approved and high:
                self._log_minor_concerns(concerns)
                return [_describe_concern(c) for c in high]

            logger.warning(
                "Moderation contradicted itself (approved=%s, %d high-severity concern(s))%s",
                approved,
                len(high),
                "; asking again" if attempt + 1 < MODERATION_ATTEMPTS else "",
            )
            prompt = f"{prompt}\n\n{MODERATION_CONSISTENCY_NOTE}"

        # Twice inconsistent. Say what the last answer claimed so the block is explicable.
        if high:
            return [
                "moderation: approved the script but listed high-severity concerns: "
                + "; ".join(_describe_concern(c) for c in high)
            ]
        return ["moderation: rejected without a stated high-severity concern"]

    @staticmethod
    def _log_minor_concerns(concerns: list[dict[str, Any]]) -> None:
        for concern in concerns:
            severity = str(concern.get("severity", "")).lower()
            if severity == "high":
                continue
            detail = str(concern.get("detail", "")).strip()
            category = str(concern.get("category", "unspecified")).strip()
            log = logger.warning if severity == "medium" else logger.info
            log("Moderation noted %s concern [%s]: %s", severity, category, detail)


def _describe_concern(concern: dict[str, Any]) -> str:
    return f"moderation[{concern.get('category', 'unspecified')}]: {concern.get('detail', '')}"


def _normalize(text: str) -> str:
    return normalize_key(text)
