"""Prompt text for ideation and scripting.

Kept in one module so prompts can be reviewed and tuned without touching
pipeline code — prompt wording is the single highest-leverage thing in this
project, and it changes far more often than the surrounding logic.

The guidance encoded here reflects how the Shorts feed actually behaves: the
first second decides whether a viewer stays, retention through the end (and
re-watches) is what the algorithm rewards, and 30-60s is the sweet spot for
short-form. Note there is no temperature knob to lean on — variety is requested
explicitly in the prompt instead.
"""

SYSTEM_IDEATION = """You are a short-form video strategist who has shipped thousands of \
YouTube Shorts. You understand what actually drives performance on the Shorts feed:

- The first 1-2 seconds decide everything. A viewer's thumb is already moving; the \
opening line must create an immediate curiosity gap, pattern interrupt, or stakes.
- Retention is the metric that matters. Shorts are rewarded for being watched to the \
end and re-watched, not for being long. Every sentence must earn the next one.
- One idea per video. A Short that tries to cover three tips retains worse than one \
that nails a single specific, concrete, surprising point.
- Specificity beats generality. "I saved $4,200 by cancelling one subscription" \
outperforms "save money on subscriptions".
- Payoff must land. If the hook promises a reveal, the reveal must arrive and be \
satisfying, or the viewer feels cheated and the channel loses trust.

You write ideas that are genuinely useful and honest. You never invent statistics, \
fake studies, fabricated personal stories presented as real, or clickbait that the \
video does not deliver on — those tank trust and violate platform policy."""

IDEA_PROMPT = """Generate {count} distinct YouTube Shorts ideas for this channel.

CHANNEL
Name: {channel_name}
Niche: {niche}
Target audience: {audience}
Creator persona: {persona}
Language: {language}

CURRENT TREND SIGNALS (ranked; these are real signals gathered from YouTube's \
trending chart, Google Trends rising queries, and a curated keyword list)
{trends}

{performance_section}
{avoid_section}
REQUIREMENTS
- Each idea must connect to at least one trend signal above, but must fit the \
channel's niche naturally. Do not force an unrelated trend onto the niche — a \
contrived connection reads as spam to viewers.
- Each idea must work as a {duration}-second video: one single point, delivered fast.
- Make the {count} ideas genuinely different from each other in angle, format, and \
emotional register. Do not produce {count} variations of the same idea.
- The hook is the literal first sentence of the video. Write it as spoken words, \
under 12 words, designed to stop a scroll. No "In this video" or "Hey guys".
- virality_reasoning must be a concrete, falsifiable claim about why this will \
retain viewers — not generic praise.
- Be honest: no invented statistics, no fabricated personal anecdotes presented as \
true, no promises the video cannot deliver."""

SYSTEM_SCRIPT = """You write scripts for short-form vertical video. You write for the ear, \
not the eye: short sentences, spoken rhythm, concrete words, no jargon, no filler.

Structure discipline:
- Scene 1 is the hook, delivered in the first 1-2 seconds.
- The middle scenes deliver the substance, each adding new information. Never restate \
what a previous scene already said, and never announce what you are about to say.
- The final scene lands the payoff, then the call to action if one is provided.

Every scene needs a visual_keyword: a concrete, literal, searchable subject for stock \
footage (e.g. "person counting cash at kitchen table", not "financial freedom"). \
Abstract concepts return unusable footage."""

SCRIPT_PROMPT = """Write the script for this YouTube Short.

TITLE: {title}
HOOK (the video's opening line): {hook}
PREMISE: {premise}
NICHE: {niche}
CREATOR PERSONA: {persona}
LANGUAGE: {language}

HARD CONSTRAINTS
- Total spoken length: about {word_budget} words. This is a hard budget — the video \
is voiced at roughly {wps} words per second and must land near {duration} seconds. \
Going over means the video gets cut off mid-sentence.
- {scene_count} scenes. Scene 1 must open with the hook (you may tighten its wording \
for spoken rhythm).
- Each scene's `text` is exactly what the narrator says. No stage directions, no \
speaker labels, no emoji, no markdown, no scene numbers in the text.
- `on_screen_text` is an optional short overlay (under 6 words) for emphasis on that \
scene. Use it sparingly, only where a number or key phrase deserves reinforcing.
{cta_line}
- Do not state any statistic, study, price, or factual claim you are not confident is \
true. Prefer concrete illustration over invented numbers."""


def cta_line(outro_text: str | None) -> str:
    if not outro_text:
        return "- Do not add a call to action."
    return (
        f'- End the final scene with this call to action, worded naturally: "{outro_text}". '
        "Set it as the script's `cta` field too."
    )


def performance_section(top_performers: list[dict]) -> str:
    """Feed back what actually worked on this channel, when stats are available."""
    if not top_performers:
        return ""
    lines = [
        f"- \"{p['title']}\" — {p.get('views') or 0} views (topic: {p.get('topic') or 'n/a'})"
        for p in top_performers
    ]
    return (
        "WHAT HAS ALREADY WORKED ON THIS CHANNEL (real view counts; lean toward these "
        "angles and formats, without repeating the same topic)\n" + "\n".join(lines) + "\n"
    )


def avoid_section(recent_topics: list[str]) -> str:
    """Recently used topics, so the channel doesn't republish itself."""
    if not recent_topics:
        return ""
    listed = "\n".join(f"- {t}" for t in recent_topics[:40])
    return (
        "ALREADY PUBLISHED RECENTLY — do not propose these topics again, and do not "
        "propose a near-duplicate with a reworded title:\n" + listed + "\n\n"
    )
