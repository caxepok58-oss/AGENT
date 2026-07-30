# Architecture

## Shape of the system

Every stage is a small module behind an interface, wired together by one
orchestrator. Nothing in the pipeline knows which LLM, voice, or footage source
is in use — that is decided once, by a factory, from config.

```
                          ┌──────────────────┐
                          │  config + .env   │
                          └────────┬─────────┘
                                   │
  ┌────────────────────────────────▼─────────────────────────────────┐
  │                     pipeline/orchestrator.py                     │
  └──┬──────┬──────────┬──────────┬─────────┬──────────┬─────────┬───┘
     │      │          │          │         │          │         │
     ▼      ▼          ▼          ▼         ▼          ▼         ▼
  trends  ideation   policy      tts    visuals    captions  metadata
     │      │          │          │         │          │         │
     │      └──────────┴──────────┴─────────┴──────────┘         │
     │                            │                             │
     ▼                            ▼                             ▼
  YouTube API                video/assembler.py            youtube/uploader.py
  Google Trends              (moviepy + ffmpeg)             (resumable upload)
  curated YAML                        │                            │
                                      ▼                            ▼
                                  output/<run>/video.mp4      storage.py (SQLite)
```

## Stage order, and why it is that order

```
trends → ideas → policy → script → policy → audio → visuals → captions
       → video → metadata → publish
```

Two decisions in that ordering are load-bearing:

**Policy runs twice, both times before rendering.** Screening an idea costs one
LLM call. Screening a finished video costs a full render — and if it passes when
it should not, it costs an upload. So ideas are screened on selection, and the
script is screened again once written, before any audio is synthesized.

**Audio comes before visuals, not after.** Scene duration is determined by how
long the narration actually takes, not by a guess. Visuals are then cut to fit
that measured duration, so nothing drifts out of sync.

## Modules

| Module | Responsibility | Extension point |
|---|---|---|
| `config.py` | Two-layer config: `Settings` (secrets, from `.env`) and `AppConfig` (behavior, from YAML) | Add a field to the relevant sub-model |
| `models.py` | Pydantic models passed between stages | — |
| `storage.py` | SQLite: run history, upload rate limiting, topic dedup, view stats | — |
| `llm/` | Provider-agnostic `complete` / `complete_json` | Subclass `LLMClient`, register in `factory.py` |
| `trends/` | Signal providers plus a merging aggregator | Subclass `TrendProvider` |
| `ideation/` | Idea and script generation; all prompt text in `prompts.py` | Edit `prompts.py` |
| `policy/` | Blocklist, duplicate detection, LLM moderation | Extend the blocklist YAML or the moderation prompt |
| `tts/` | Voice synthesis **and per-word timings** | Subclass `TTSProvider` |
| `visuals/` | Stock footage and generated fallback cards | Subclass `VisualProvider` |
| `captions/` | ASS subtitle document with word highlighting | — |
| `video/` | moviepy composition, then ffmpeg caption burn | — |
| `metadata/` | Title, description, tags, hashtags, with API limits enforced | Edit the `SYSTEM` prompt |
| `youtube/` | OAuth, resumable upload, stats | — |
| `pipeline/` | Orchestration, rate limiting, scheduling | — |
| `cli.py` | Typer commands | Add a `@app.command()` |

## Design decisions worth knowing

**Word-level caption timing drives the TTS interface.** `TTSProvider.synthesize`
returns timings, not just a file path, because word-by-word highlighted captions
are the visual signature of modern short-form video and they need to know when
each word is actually spoken.

edge-tts is the default because it reports boundary events — but *which* events
is the service's choice, not ours, and it has changed for the same voice over
time. The provider therefore resolves timings in three tiers: `WordBoundary`
events used verbatim; failing that, `SentenceBoundary` events, whose real start
and duration anchor a length-weighted distribution of that sentence's words;
failing that, the rendered audio's measured duration across the whole scene.
Captions stay anchored to real audio in every tier, and the active tier is
logged at debug level rather than hidden.

**Captions are burned by ffmpeg, not moviepy.** libass renders text far more
crisply than compositing a text clip per word, and it costs one extra encode
instead of hundreds of per-frame composites. This is why the assembler is two
stages: moviepy writes a master, ffmpeg burns captions onto it.

**Scene emphasis text rides the same ASS file.** A scene's `on_screen_text` is
emitted as a second ASS style rather than a separate render pass, so it costs
nothing extra. It is placed on the opposite side of the frame from the captions —
if captions move to the top via config, overlays move to the bottom — because
the two would otherwise collide.

**Footage is cropped, never letterboxed.** Source clips are scaled to cover
1080×1920 and centre-cropped. Black bars read as low-effort in the feed; losing
the edges of a shot does not.

**Stills get a slow zoom.** A frozen frame in a Short looks like a broken video,
so images drift inward at ~1.8%/second. Short clips loop rather than freezing on
their last frame, for the same reason.

**Providers degrade; they do not fail the run.** A missing API key, a
rate-limited upstream, or an empty search returns empty (trends) or falls back to
generated cards (visuals). The one exception is LLM moderation, which fails
*closed* — an unreachable reviewer stops the run rather than letting unreviewed
content through.

**No sampling parameters on the LLM.** Current Anthropic models reject
`temperature`/`top_p`, so output variety is requested in the prompt instead
("make the ideas genuinely different in angle, format, and emotional register").
Structured output is requested via a JSON schema, so parsing cannot fail on a
successful response.

**The storage layer is a feedback loop, not just a log.** `top_performers()`
feeds real view counts back into the ideation prompt, and `recent_topics()` keeps
the channel from republishing itself.

## Extending it

### A new visual source

```python
# src/shorts_agent/visuals/my_source.py
from shorts_agent.visuals.base import VisualProvider
from shorts_agent.visuals.generated import GeneratedVisualProvider
from shorts_agent.models import VisualAsset

class MyVisualProvider(VisualProvider):
    name = "my_source"

    def __init__(self, api_key, *, width=1080, height=1920):
        self.api_key = api_key
        # Always keep a fallback: a scene with no visual cannot be recovered later.
        self._fallback = GeneratedVisualProvider(width=width, height=height)

    def fetch(self, keyword, output_dir, scene_index, *, text=None) -> VisualAsset:
        ...
```

Then add a branch to `visuals/factory.py` and the `Literal` in
`ProvidersConfig.visuals`.

### A new trend signal

Subclass `TrendProvider`, return `[]` on any failure, and add it to
`build_trend_providers()` plus the `TrendProviderName` literal. The aggregator
handles deduplication and cross-source scoring for you.

### Changing what the videos sound like

Nearly all of it is in `src/shorts_agent/ideation/prompts.py`. `SYSTEM_IDEATION`
encodes what makes a Short retain viewers; `SYSTEM_SCRIPT` encodes writing for
the ear. Those two strings shape output more than any code change.

## Data on disk

```
output/<run-id>/
  audio/scene_00.mp3      per-scene narration
  visuals/scene_00.*      per-scene footage or card
  captions.ass            word-timed subtitles
  work/master.mp4         pre-caption master (deleted after burning)
  video.mp4               final deliverable
data/shorts_agent.db      run history, uploads, view stats
```

## Testing

124 tests, no network access. `tests/conftest.py` provides `FakeLLM`, which
scripts a sequence of JSON responses so a multi-call stage (ideas → script →
moderation → metadata) can be driven deterministically. Providers are stubbed by
monkeypatching the factory functions in `pipeline.orchestrator`, which is why
those are module-level imports rather than inline ones.
