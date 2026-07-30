"""Title, description, tags and hashtags for a Short.

The hard limits enforced here come from the YouTube Data API, and violating them
fails the upload rather than degrading it:

* title — 100 characters, and no ``<`` or ``>``
* description — 5000 characters, same character restriction
* tags — 500 characters total across all tags

Titles are also kept well under the limit on purpose: the Shorts feed truncates
long titles, so a title that only reads correctly in full is a wasted hook.
"""

from __future__ import annotations

import logging
import re

from shorts_agent.config import AppConfig
from shorts_agent.llm.base import LLMClient, object_schema
from shorts_agent.models import Idea, Script, VideoMetadata
from shorts_agent.text import strip_non_word

logger = logging.getLogger(__name__)

TITLE_LIMIT = 100
TITLE_TARGET = 70
DESCRIPTION_LIMIT = 5000
TAGS_TOTAL_LIMIT = 500
MAX_HASHTAGS = 4  # YouTube ignores hashtags past the third in the title area

METADATA_SCHEMA = object_schema(
    {
        "title": {
            "type": "string",
            "description": f"Under {TITLE_TARGET} characters, front-loaded with the hook.",
        },
        "description": {
            "type": "string",
            "description": "2-4 sentences describing the video, then hashtags on the last line.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "8-15 lowercase search keywords, no '#'.",
        },
        "hashtags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-4 hashtags including #Shorts, each starting with '#'.",
        },
    }
)

SYSTEM = """You write metadata for YouTube Shorts. You optimise for how viewers and search \
actually behave on short-form video:

- The title is a hook, not a summary. Front-load the most compelling words — the feed \
truncates long titles, and anything after the cut may as well not exist.
- Titles must deliver what they promise. Withholding information to create curiosity is \
fine; implying something the video does not contain is not, and it costs the channel \
viewer trust and long-term reach.
- The description's first line is the only part most viewers see. Put the value there.
- Tags are for search intent: the phrases someone would actually type. Not adjectives, \
not restatements of the title.
- Include #Shorts. Keep hashtag counts low; a wall of hashtags reads as spam and \
YouTube ignores most of them anyway.
- No ALL CAPS words, no clickbait punctuation ("!!!", "???"), no emoji spam. One emoji \
at most, only if it genuinely fits."""

PROMPT = """Write YouTube metadata for this Short.

TITLE IDEA: {title}
HOOK: {hook}
NICHE: {niche}
CHANNEL: {channel_name}
LANGUAGE: {language}
TREND KEYWORDS THIS VIDEO TARGETS: {trend_keywords}

FULL NARRATION
{script}

Constraints:
- Title under {title_target} characters.
- Description: 2-4 sentences of real value, then a final line with the hashtags.
- Tags: 8-15 lowercase search phrases someone would type. No '#'. Total length across \
all tags must stay under {tags_limit} characters.
- Hashtags: 3-4, including #Shorts.{extra_hashtags}"""


def _sanitize_line(text: str) -> str:
    """Strip characters the Data API rejects and collapse whitespace."""
    return re.sub(r"\s+", " ", text.replace("<", "").replace(">", "")).strip()


class MetadataGenerator:
    def __init__(self, llm: LLMClient, config: AppConfig):
        self.llm = llm
        self.config = config

    def generate(self, idea: Idea, script: Script) -> VideoMetadata:
        channel = self.config.channel
        defaults = channel.default_hashtags or []
        extra = (
            f"\n- Prefer including these channel hashtags where they fit: {', '.join(defaults)}"
            if defaults
            else ""
        )

        prompt = PROMPT.format(
            title=idea.title,
            hook=idea.hook,
            niche=channel.niche,
            channel_name=channel.name,
            language=channel.language,
            trend_keywords=", ".join(idea.trend_keywords) or "(none)",
            script=script.full_text,
            title_target=TITLE_TARGET,
            tags_limit=TAGS_TOTAL_LIMIT,
            extra_hashtags=extra,
        )

        payload = self.llm.complete_json(prompt, schema=METADATA_SCHEMA, system=SYSTEM)

        hashtags = _clean_hashtags(payload.get("hashtags", []), defaults)
        metadata = VideoMetadata(
            title=_sanitize_line(str(payload.get("title") or idea.title)),
            description=_build_description(
                str(payload.get("description") or idea.premise), hashtags
            ),
            tags=_clean_tags(payload.get("tags", [])),
            hashtags=hashtags,
            category_id=channel.category_id,
            privacy_status=self.config.publishing.privacy_status,
            made_for_kids=channel.made_for_kids,
            contains_synthetic_media=self.config.content.ai_disclosure,
            playlist_id=self.config.publishing.playlist_id or None,
        )

        return validate_metadata(metadata)


def _clean_tags(raw: list) -> list[str]:
    """Normalise tags and trim the list to fit the API's total length budget."""
    seen: set[str] = set()
    tags: list[str] = []
    used = 0

    for item in raw:
        tag = _sanitize_line(str(item)).lstrip("#").lower()
        if not tag or tag in seen:
            continue
        # The API counts the combined length; a comma per tag is the separator.
        projected = used + len(tag) + (1 if tags else 0)
        if projected > TAGS_TOTAL_LIMIT:
            continue
        seen.add(tag)
        tags.append(tag)
        used = projected

    return tags


def _clean_hashtags(raw: list, defaults: list[str]) -> list[str]:
    seen: set[str] = set()
    hashtags: list[str] = []

    for item in list(raw) + list(defaults):
        tag = _sanitize_line(str(item))
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = f"#{tag}"
        tag = strip_non_word(tag, allow="#")
        key = tag.casefold()
        if len(tag) < 2 or key in seen:
            continue
        seen.add(key)
        hashtags.append(tag)

    if not any(h.lower() == "#shorts" for h in hashtags):
        hashtags.insert(0, "#Shorts")

    return hashtags[:MAX_HASHTAGS]


def _build_description(body: str, hashtags: list[str]) -> str:
    """Assemble the description, ensuring hashtags appear exactly once."""
    text = body.replace("<", "").replace(">", "").strip()

    # Drop any hashtag line the model already appended so we don't duplicate it.
    lines = [line for line in text.splitlines() if line.strip()]
    while lines and lines[-1].strip().startswith("#"):
        lines.pop()

    description = "\n".join(lines).strip()
    if hashtags:
        description = f"{description}\n\n{' '.join(hashtags)}".strip()
    return description[:DESCRIPTION_LIMIT]


def validate_metadata(metadata: VideoMetadata) -> VideoMetadata:
    """Clamp metadata to the Data API's limits, warning where content was cut."""
    metadata.title = metadata.title.strip()

    if len(metadata.title) > TITLE_LIMIT:
        logger.warning("Title exceeded %d characters and was truncated", TITLE_LIMIT)
        metadata.title = metadata.title[: TITLE_LIMIT - 1].rstrip() + "…"

    # A whitespace-only title passes a plain truthiness check but is rejected by
    # the Data API, so normalise before validating.
    if not metadata.title:
        raise ValueError("Generated metadata has an empty title")

    if len(metadata.description) > DESCRIPTION_LIMIT:
        logger.warning("Description exceeded %d characters and was truncated", DESCRIPTION_LIMIT)
        metadata.description = metadata.description[:DESCRIPTION_LIMIT]

    total_tag_length = sum(len(t) for t in metadata.tags) + max(len(metadata.tags) - 1, 0)
    if total_tag_length > TAGS_TOTAL_LIMIT:
        metadata.tags = _clean_tags(metadata.tags)
        logger.warning("Tags exceeded %d characters total and were trimmed", TAGS_TOTAL_LIMIT)

    return metadata
