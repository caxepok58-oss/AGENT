from __future__ import annotations

import logging

from shorts_agent.config import AppConfig
from shorts_agent.exceptions import ProviderError
from shorts_agent.ideation import prompts
from shorts_agent.llm.base import LLMClient, object_schema
from shorts_agent.models import Idea, Scene, Script

logger = logging.getLogger(__name__)

SCRIPT_SCHEMA = object_schema(
    {
        "scenes": {
            "type": "array",
            "items": object_schema(
                {
                    "text": {
                        "type": "string",
                        "description": "Exactly what the narrator says in this scene.",
                    },
                    "visual_keyword": {
                        "type": "string",
                        "description": "Concrete, literal, searchable stock-footage subject.",
                    },
                    "on_screen_text": {
                        "type": "string",
                        "description": "Optional short overlay for this scene; "
                        "empty string for none.",
                    },
                }
            ),
        },
        "cta": {
            "type": "string",
            "description": "The call to action, or an empty string if none.",
        },
    }
)

SECONDS_PER_SCENE = 5.5
MAX_SCENES = 12
MIN_SCENES = 3


def scene_count_for(duration_seconds: float) -> int:
    return max(MIN_SCENES, min(MAX_SCENES, round(duration_seconds / SECONDS_PER_SCENE)))


class ScriptWriter:
    def __init__(self, llm: LLMClient, config: AppConfig):
        self.llm = llm
        self.config = config

    def write(self, idea: Idea) -> Script:
        content = self.config.content
        channel = self.config.channel
        word_budget = int(content.target_duration_seconds * content.words_per_second)
        scenes = scene_count_for(content.target_duration_seconds)

        prompt = prompts.SCRIPT_PROMPT.format(
            title=idea.title,
            hook=idea.hook,
            premise=idea.premise,
            niche=channel.niche,
            persona=channel.persona.strip(),
            language=channel.language,
            word_budget=word_budget,
            wps=content.words_per_second,
            duration=int(content.target_duration_seconds),
            scene_count=scenes,
            cta_line=prompts.cta_line(content.outro_text if content.add_outro_cta else None),
        )

        script = self._request(prompt, idea)
        estimated = self.estimated_duration(script)

        # The model reliably respects the scene count but drifts on total length,
        # so give it one corrective pass with the measured overshoot rather than
        # shipping a video that gets cut off mid-sentence.
        if estimated > content.max_duration_seconds:
            over_by = estimated - content.target_duration_seconds
            logger.info(
                "Script ran long (%.1fs vs %.1fs max); requesting a tightened pass",
                estimated,
                content.max_duration_seconds,
            )
            retry_prompt = (
                f"{prompt}\n\nYour previous script was {script.word_count} words, which "
                f"voices to about {estimated:.0f} seconds — roughly {over_by:.0f} seconds "
                f"too long. Rewrite it at no more than {word_budget} words total, keeping "
                "the same structure and the strongest lines. Cut whole sentences rather "
                "than trimming every sentence into fragments."
            )
            tightened = self._request(retry_prompt, idea)
            if self.estimated_duration(tightened) < estimated:
                script = tightened

        final = self.estimated_duration(script)
        if final < content.min_duration_seconds:
            logger.warning(
                "Script voices to only %.1fs, below the %.1fs minimum",
                final,
                content.min_duration_seconds,
            )

        return script

    def _request(self, prompt: str, idea: Idea) -> Script:
        payload = self.llm.complete_json(prompt, schema=SCRIPT_SCHEMA, system=prompts.SYSTEM_SCRIPT)

        scenes: list[Scene] = []
        for index, raw in enumerate(payload.get("scenes", [])):
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            overlay = str(raw.get("on_screen_text", "")).strip()
            scenes.append(
                Scene(
                    index=index,
                    text=text,
                    visual_keyword=str(raw.get("visual_keyword", "")).strip() or idea.title,
                    on_screen_text=overlay or None,
                )
            )

        if not scenes:
            raise ProviderError("The model returned a script with no usable scenes")

        cta = str(payload.get("cta", "")).strip()
        return Script(idea_id=idea.id, title=idea.title, scenes=scenes, cta=cta or None)

    def estimated_duration(self, script: Script) -> float:
        return script.word_count / self.config.content.words_per_second
