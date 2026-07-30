"""Build ASS subtitles with word-by-word highlighting.

ASS (rather than SRT or drawing text with moviepy) because it is the only option
that gives all three of: per-word colour changes, a heavy outline that stays
readable over arbitrary footage, and rendering by ffmpeg's libass in a single
pass — which is far faster and sharper than compositing a text clip per word.

The visual convention implemented here is the short-form standard: a few words
on screen at a time, the currently spoken word picked out in an accent colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shorts_agent.config import CaptionsConfig
from shorts_agent.models import WordTiming

# ASS alignment uses a numpad layout; these are the centered positions.
_ALIGNMENT = {"bottom": 2, "middle": 5, "top": 8}

WORDS_PER_CHUNK = 4
MIN_EVENT_DURATION = 0.08
# Rough advance width of a bold sans-serif glyph as a fraction of font size.
# Used only to pre-wrap chunks into pleasing lines; libass enforces the real
# margins via WrapStyle, so an imprecise estimate cannot cause overflow.
CHAR_WIDTH_RATIO = 0.58
MIN_CHARS_PER_LINE = 8
# How long a finished phrase stays on screen during a pause before it clears.
MAX_PHRASE_HOLD = 0.6
# Overlays are secondary to the narration captions, so they render smaller.
OVERLAY_SIZE_RATIO = 0.62


@dataclass(frozen=True)
class Overlay:
    """A short emphasis line shown for one scene, opposite the captions."""

    text: str
    start: float
    end: float


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


def chars_per_line(width: int, font_size: int, margin_h: int) -> int:
    """Estimate how many characters fit across the frame at this font size.

    A fixed character budget cannot work: 22 characters fits comfortably at font
    size 60 and runs off both edges at 90. Deriving it from the usable width
    keeps pre-wrapping sensible whatever the configured size.
    """
    usable = max(width - 2 * margin_h, 1)
    return max(MIN_CHARS_PER_LINE, int(usable / (font_size * CHAR_WIDTH_RATIO)))


def _wrap(words: list[str], max_chars: int) -> list[list[str]]:
    """Split a chunk's words into display lines that fit the frame width."""
    lines: list[list[str]] = [[]]
    for word in words:
        current = lines[-1]
        projected = len(" ".join(current + [word]))
        if current and projected > max_chars:
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
    overlays: list[Overlay] | None = None,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Render an ASS subtitle document for ``timings``.

    ``overlays`` are per-scene emphasis lines shown opposite the captions, so the
    two never collide.
    """
    alignment = _ALIGNMENT.get(config.position, 5)
    # Outline and shadow scale with the font so captions stay legible when the
    # font size is tuned, and a generous vertical margin keeps text clear of
    # the Shorts UI overlays at the bottom of the frame.
    outline = max(2, round(config.font_size * 0.07))
    shadow = max(1, round(config.font_size * 0.03))
    margin_v = round(height * 0.14)
    margin_h = round(width * 0.06)

    overlay_size = round(config.font_size * OVERLAY_SIZE_RATIO)
    overlay_outline = max(2, round(overlay_size * 0.07))
    overlay_alignment, overlay_margin_v = _overlay_placement(config.position, height)

    max_chars = chars_per_line(width, config.font_size, margin_h)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{config.font},{config.font_size},{config.base_color},{config.highlight_color},{config.outline_color},&H64000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},{margin_h},{margin_h},{margin_v},1
Style: Overlay,{config.font},{overlay_size},{config.highlight_color},{config.highlight_color},{config.outline_color},&H64000000,-1,0,0,0,100,100,0,0,1,{overlay_outline},{shadow},{overlay_alignment},{margin_h},{margin_h},{overlay_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    chunks = chunk_timings(timings)

    for chunk_index, chunk in enumerate(chunks):
        next_chunk_start = (
            chunks[chunk_index + 1][0].start if chunk_index + 1 < len(chunks) else None
        )

        for active_index, active in enumerate(chunk):
            end = active.end
            if active_index + 1 < len(chunk):
                # Hold the highlight until the next word begins so the caption
                # never blinks off during the gap between words.
                end = max(end, chunk[active_index + 1].start)
            elif next_chunk_start is not None:
                # Last word of a phrase: keep it on screen into the pause rather
                # than cutting to a bare frame, but cap the hold so a long
                # silence doesn't leave stale text sitting there.
                end = max(end, min(next_chunk_start, active.end + MAX_PHRASE_HOLD))

            if end - active.start < MIN_EVENT_DURATION:
                end = active.start + MIN_EVENT_DURATION

            text = _render_chunk(chunk, active_index, config, max_chars)
            events.append(
                f"Dialogue: 0,{_timestamp(active.start)},{_timestamp(end)},Default,,0,0,0,,{text}"
            )

    events.extend(_overlay_events(overlays or []))
    return header + "\n".join(events) + "\n"


def _overlay_placement(caption_position: str, height: int) -> tuple[int, int]:
    """Put overlays on the opposite side of the frame from the captions.

    Captions move around via config; overlays must not land on top of them. The
    margin also keeps text clear of the Shorts UI, which occupies the bottom of
    the frame and the top-right corner.
    """
    if caption_position == "top":
        return _ALIGNMENT["bottom"], round(height * 0.28)
    return _ALIGNMENT["top"], round(height * 0.10)


def _overlay_events(overlays: list[Overlay]) -> list[str]:
    events: list[str] = []
    for overlay in overlays:
        text = _escape(overlay.text)
        if not text or overlay.end <= overlay.start:
            continue
        events.append(
            f"Dialogue: 1,{_timestamp(overlay.start)},{_timestamp(overlay.end)},"
            f"Overlay,,0,0,0,,{text}"
        )
    return events


def _render_chunk(
    chunk: list[WordTiming], active_index: int, config: CaptionsConfig, max_chars: int
) -> str:
    """Render one chunk with the active word in the highlight colour."""
    words = [_escape(t.word) for t in chunk]
    lines = _wrap(words, max_chars)

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
    overlays: list[Overlay] | None = None,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_ass(timings, config, overlays=overlays, width=width, height=height),
        encoding="utf-8",
    )
    return output_path
