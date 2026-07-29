from __future__ import annotations

import logging

from shorts_agent.config import AppConfig
from shorts_agent.ideation import prompts
from shorts_agent.llm.base import LLMClient, object_schema
from shorts_agent.models import Idea, TrendTopic

logger = logging.getLogger(__name__)

IDEA_SCHEMA = object_schema(
    {
        "ideas": {
            "type": "array",
            "items": object_schema(
                {
                    "title": {
                        "type": "string",
                        "description": "Working title of the video, under 80 characters.",
                    },
                    "hook": {
                        "type": "string",
                        "description": "The literal first spoken sentence, under 12 words.",
                    },
                    "premise": {
                        "type": "string",
                        "description": "One or two sentences describing what the video covers.",
                    },
                    "target_emotion": {
                        "type": "string",
                        "description": "The dominant feeling the viewer should have "
                        "(e.g. curiosity, surprise, relief, indignation).",
                    },
                    "trend_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Which of the supplied trend signals this idea uses.",
                    },
                    "virality_reasoning": {
                        "type": "string",
                        "description": "A concrete claim about why this retains viewers.",
                    },
                }
            ),
        }
    }
)


class IdeaGenerator:
    def __init__(self, llm: LLMClient, config: AppConfig):
        self.llm = llm
        self.config = config

    def generate(
        self,
        topics: list[TrendTopic],
        *,
        count: int | None = None,
        recent_topics: list[str] | None = None,
        top_performers: list[dict] | None = None,
    ) -> list[Idea]:
        count = count or self.config.content.ideas_per_run
        channel = self.config.channel

        prompt = prompts.IDEA_PROMPT.format(
            count=count,
            channel_name=channel.name,
            niche=channel.niche,
            audience=channel.audience,
            persona=channel.persona.strip(),
            language=channel.language,
            trends=self._format_trends(topics),
            duration=int(self.config.content.target_duration_seconds),
            performance_section=prompts.performance_section(top_performers or []),
            avoid_section=prompts.avoid_section(recent_topics or []),
        )

        payload = self.llm.complete_json(prompt, schema=IDEA_SCHEMA, system=prompts.SYSTEM_IDEATION)

        ideas: list[Idea] = []
        for raw in payload.get("ideas", [])[:count]:
            ideas.append(
                Idea(
                    title=str(raw.get("title", "")).strip(),
                    hook=str(raw.get("hook", "")).strip(),
                    premise=str(raw.get("premise", "")).strip(),
                    target_emotion=str(raw.get("target_emotion", "")).strip(),
                    trend_keywords=[str(k) for k in raw.get("trend_keywords", [])],
                    virality_reasoning=str(raw.get("virality_reasoning", "")).strip(),
                    niche=channel.niche,
                )
            )

        usable = [i for i in ideas if i.title and i.hook]
        logger.info("Generated %d usable ideas from %d trend topics", len(usable), len(topics))
        return usable

    @staticmethod
    def _format_trends(topics: list[TrendTopic]) -> str:
        if not topics:
            return "(no live trend signals available — rely on the niche itself)"
        lines = []
        for i, topic in enumerate(topics, start=1):
            detail = f"score {topic.score:.1f}, source {topic.source}"
            lines.append(f"{i}. {topic.keyword} ({detail})")
        return "\n".join(lines)
