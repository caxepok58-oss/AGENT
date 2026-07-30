# shorts-agent

A trend-aware AI agent that researches what is currently working on YouTube,
writes a short-form script, produces a vertical video with voiceover and
word-by-word captions, and publishes it to YouTube with SEO metadata.

> **Read [docs/POLICY.md](docs/POLICY.md) before publishing anything.** Automated
> uploading touches YouTube's spam, authenticity and synthetic-media policies, and
> getting them wrong can cost you a channel. This project defaults to rendering
> **without uploading**, and uploads as **private** when you do enable it.

## What it actually does

```
trends → ideas → policy → script → policy → voiceover → visuals → captions
       → video → metadata → (review) → upload
```

1. **Research** — merges signals from YouTube's trending chart, Google Trends
   rising queries, and your own curated keyword list. Topics that several
   independent sources agree on rank highest.
2. **Ideate** — an LLM proposes video concepts tied to those signals, biased by
   which of your past videos actually got views, and excluding topics you
   published recently.
3. **Screen** — a keyword blocklist, near-duplicate detection, and an optional
   LLM review run *before* anything expensive is rendered.
4. **Script** — scene-by-scene narration sized to your target duration, with one
   corrective pass if it runs long.
5. **Produce** — text-to-speech with per-word timings, stock or generated
   visuals, ASS captions burned in by ffmpeg, composed to 1080×1920 H.264.
6. **Publish** — resumable upload with SEO metadata, AI-content disclosure, and
   optional scheduling. Off by default.

## Quickstart

```bash
# 1. Install. The 'all' extra pulls every integration; see "Install options" below.
pip install -e ".[all]"

# 2. Create config files in the current directory.
shorts-agent init

# 3. Add your keys to .env, then set your niche and persona in config/config.yaml.
#    The niche and persona drive everything the agent writes — be specific.

# 4. Check your setup: what works, what will fall back, what will fail.
shorts-agent doctor

# 5. See what it would make, without producing anything.
shorts-agent ideate

# 6. Produce a video locally. Nothing is uploaded.
shorts-agent run

# 7. Review it quickly: frames as one image, plus script and metadata.
shorts-agent preview <run-id>

# 8. Once you have watched the file and read docs/POLICY.md:
shorts-agent auth              # one-time browser authorization
shorts-agent publish <run-id>  # upload the video you just reviewed
```

The only required key is an LLM key. Everything else degrades: without a stock
footage key it renders generated backgrounds, without a YouTube API key it uses
your curated keyword list, without Google Trends it uses the other sources.

## Commands

| Command | What it does |
|---|---|
| `init` | Copy example config and `.env` into place |
| `doctor` | Check ffmpeg, fonts, keys and config; report what will silently degrade |
| `trends` | Show current trend signals for your niche |
| `ideate` | Generate and screen ideas without producing a video |
| `run` | Full pipeline. Renders locally; add `--publish` to upload |
| `preview <run-id>` | Contact sheet of frames plus the script and metadata, for fast review |
| `publish <run-id>` | Upload a previously rendered video after review |
| `auth` | Authorize this machine to upload to your channel |
| `report` | Recent runs and how published videos are performing (`--refresh` for fresh view counts) |
| `daemon` | Run on a schedule (`--every 12`). The daily upload cap still applies |

## Install options

Each integration is a separate extra, so you only install what you use:

```bash
pip install -e ".[llm-anthropic,tts,video]"   # minimum for local production
pip install -e ".[all]"                        # everything
pip install -e ".[all,dev]"                    # plus pytest, ruff, mypy
```

| Extra | Provides |
|---|---|
| `llm-anthropic` / `llm-openai` | Script, idea and metadata generation |
| `tts` | edge-tts voiceover (no API key, reports timing boundaries) |
| `tts-elevenlabs` | ElevenLabs voices (higher quality, needs a key) |
| `video` | moviepy, Pillow, numpy and a bundled ffmpeg |
| `youtube` | Upload and analytics |
| `trends` | Google Trends signals |
| `scheduler` | The `daemon` command |

## Configuration

Two files, deliberately separate:

- **`.env`** — secrets and machine paths. Never commit it.
- **`config/config.yaml`** — niche, persona, durations, provider choices,
  publishing limits. Safe to commit, and worth keeping one per channel.

The settings that matter most:

| Setting | Why it matters |
|---|---|
| `channel.niche` / `channel.persona` | Drives every generated idea and script. Vague values produce generic videos |
| `content.target_duration_seconds` | 30–60s is the sweet spot for Shorts; the platform cap is 180s |
| `content.ai_disclosure` | Sets the API's synthetic-media flag. Keep it `true` for AI voice or visuals |
| `publishing.privacy_status` | Defaults to `private` so a human sees each video first |
| `publishing.max_uploads_per_day` | Hard stop against a runaway loop tripping spam policies |
| `providers.visuals` | `generated` needs no key; `pexels`/`pixabay` need free keys |

## Documentation

- **[docs/SETUP.md](docs/SETUP.md)** — obtaining each API key and OAuth credential, step by step
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — how the modules fit together and where to extend
- **[docs/POLICY.md](docs/POLICY.md)** — YouTube policy, disclosure, copyright and rate limits

## Development

```bash
pytest              # 233 tests, no network access required
pytest --cov=shorts_agent --cov-report=term-missing  # coverage report
ruff check src/     # lint
ruff format src/    # format
mypy src/shorts_agent
```

## Limitations

Worth knowing before you rely on this:

- **It does not guarantee views.** It applies current best practice for hooks,
  pacing, captions and metadata. Reach depends on your niche, your channel's
  history, and luck.
- **Trend providers are unofficial or rate-limited.** Google Trends via pytrends
  scrapes an undocumented endpoint and breaks periodically; YouTube's API has a
  daily quota. The curated keyword list exists as the reliable fallback.
- **Generated visuals are plain by design.** Gradient cards keep the pipeline
  key-free and keep captions readable, but stock footage performs better. Add a
  Pexels or Pixabay key for real footage.
- **Stock footage is not automatically credited.** Pexels and Pixabay licences
  do not require attribution, but check the current terms yourself.
- **Caption timing precision varies by provider and voice.** edge-tts reports
  boundary events, but which kind is up to the service: some voices emit
  per-word boundaries (exact highlighting), others only per-sentence boundaries,
  in which case words are distributed inside each sentence's real span. Captions
  stay anchored to the audio either way; individual word highlights can drift by
  a fraction of a second in the sentence-level case. ElevenLabs timings are
  derived from the rendered audio's measured duration.
- **Music is not included.** Drop your own licensed tracks in `config/music/`.
  Do not use commercial music you have not licensed.

## Licence

MIT. You are responsible for the content you publish with it.
