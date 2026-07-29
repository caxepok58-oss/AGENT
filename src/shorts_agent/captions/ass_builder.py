"""Build ASS subtitles with word-by-word highlighting.

ASS (rather than SRT or drawing text with moviepy) because it is the only option
that gives all three of: per-word colour changes, a heavy outline that stays
readable over arbitrary footage, and rendering by ffmpeg's libass in a single
pass — which is far faster and sharper than compositing a text clip per word.

The visual convention implemented here is the short-form standard: a few words
on screen at a time, the currently spoken word picked out in an accent colour.
"""

from __future__ import annotations

from pathlib import Path

from shorts_agent.config import CaptionsConfig
from shorts_agent.models import WordTiming

# ASS alignment uses a numpad layout; these are the centered positions.
_ALIGNMENT = {"bottom": 2, "middle": 5, "top": 8}

WORDS_PER_CHUNK = 4
MAX_CHARS_PER_LINE = 22
MIN_EVENT_DURATION = 0.08


def _timestamp(seconds: float) -> str:
    """Format seconds as ASS ``H:MM:SS.cc``."""
    seconds = max(seconds, 0.0)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    if centiseconds == 100:  # rounding can carry
        centiseconds = 0
        secs += 1
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _escape(text: str) -> str:
    """Neutralize ASS markup characters in caption text.

    Braces open override blocks and backslashes start tags, so unescaped text
    from a script would silently disappear or corrupt styling.
    """
    return (
        text.replace("\\", "/")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def _color_tag(color: str) -> str:
    """Format a colour for an inline override tag.

    Style lines take a bare ``&HAABBGGRR``, but inline ``\\c`` overrides need a
    closing ``&``. libass tolerates its absence; other renderers do not.
    """
    value = color.strip()
    if not value.endswith("&"):
        value += "&"
    return f"{{\\c{value}}}"


def _wrap(words: list[str]) -> list[list[str]]:
    """Split a chunk's words into display lines that fit the frame width."""
    lines: list[list[str]] = [[]]
    for word in words:
        current = lines[-1]
        projected = len(" ".join(current + [word]))
        if current and projected > MAX_CHARS_PER_LINE:
            lines.append([word])
        else:
            current.append(word)
    return lines


def chunk_timings(
    timings: list[WordTiming], words_per_chunk: int = WORDS_PER_CHUNK
) -> list[list[WordTiming]]:
    """Group word timings into on-screen phrases.

    A pause longer than half a second reads as a sentence boundary, so a new
    chunk starts there even if the current one is not full — otherwise captions
    join the end of one thought to the start of the next.
    """
    chunks: list[list[WordTiming]] = []
    current: list[WordTiming] = []

    for timing in timings:
        if current:
            gap = timing.start - current[-1].end
            if len(current) >= words_per_chunk or gap > 0.5:
                chunks.append(current)
                current = []
        current.append(timing)

    if current:
        chunks.append(current)
    return chunks


def build_ass(
    timings: list[WordTiming],
    config: CaptionsConfig,
    *,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Render an ASS subtitle document for ``timings``."""
    alignment = _ALIGNMENT.get(config.position, 5)
    # Outline and shadow scale with the font so captions stay legible when the
    # font size is tuned, and a generous vertical margin keeps text clear of
    # the Shorts UI overlays at the bottom of the frame.
    outline = max(2, round(config.font_size * 0.07))
    shadow = max(1, round(config.font_size * 0.03))
    margin_v = round(height * 0.14)
    margin_h = round(width * 0.06)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{config.font},{config.font_size},{config.base_color},{config.highlight_color},{config.outline_color},&H64000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    for chunk in chunk_timings(timings):
        for active_index, active in enumerate(chunk):
            end = active.end
            # Hold each word's highlight until the next word begins so the
            # caption never blinks off during the gap between words.
            if active_index + 1 < len(chunk):
                end = max(end, chunk[active_index + 1].start)
            if end - active.start < MIN_EVENT_DURATION:
                end = active.start + MIN_EVENT_DURATION

            text = _render_chunk(chunk, active_index, config)
            events.append(
                f"Dialogue: 0,{_timestamp(active.start)},{_timestamp(end)},Default,,0,0,0,,{text}"
            )

    return header + "\n".join(events) + "\n"


def _render_chunk(chunk: list[WordTiming], active_index: int, config: CaptionsConfig) -> str:
    """Render one chunk with the active word in the highlight colour."""
    words = [_escape(t.word) for t in chunk]
    lines = _wrap(words)

    highlight = _color_tag(config.highlight_color)
    base = _color_tag(config.base_color)

    rendered_lines: list[str] = []
    position = 0
    for line in lines:
        parts: list[str] = []
        for word in line:
            if position == active_index:
                parts.append(f"{highlight}{word}{base}")
            else:
                parts.append(word)
            position += 1
        rendered_lines.append(" ".join(parts))

    return "\\N".join(rendered_lines)


def write_ass(
    timings: list[WordTiming],
    output_path: Path,
    config: CaptionsConfig,
    *,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_ass(timings, config, width=width, height=height), encoding="utf-8")
    return output_path
