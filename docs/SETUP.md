# Setup

Every integration is optional except an LLM key. Set up what you need and skip
the rest — the pipeline degrades rather than failing.

| You want to… | You need |
|---|---|
| Generate ideas, scripts, metadata | An LLM key (**required**) |
| Voiceover with timed captions | Nothing — edge-tts is keyless |
| Real stock footage | A free Pexels **or** Pixabay key |
| Live YouTube trend signals | A YouTube Data API key |
| Upload to your channel | YouTube OAuth credentials |
| Higher-quality voices | An ElevenLabs key |

---

## 1. Install

```bash
git clone <your-fork> && cd shorts-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
shorts-agent init
```

`init` creates `.env`, `config/config.yaml`, `config/manual_trends.yaml` and
`config/policy_blocklist.yaml` in the current directory.

ffmpeg comes bundled via `imageio-ffmpeg`, so you do not need to install it
separately. If you prefer a system ffmpeg, make sure it is built with `libass`
(`ffmpeg -filters | grep ass`) or captions cannot be burned in.

**Fonts:** captions are rendered by name through fontconfig, so the font in
`config.yaml` must exist on the machine doing the render. The default,
`DejaVu Sans`, ships with virtually every Linux distribution. On macOS or
Windows use `Arial` or `Helvetica`. Check what is available with `fc-list : family`.

---

## 2. LLM (required: an API key, or a local model)

### Anthropic (default)

1. Sign in at <https://console.anthropic.com>.
2. **API keys → Create key**, copy it.
3. In `.env`:

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-5-5
ANTHROPIC_API_KEY=sk-ant-...
```

### OpenAI

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

`LLM_PROVIDER` in `.env` overrides `providers.llm` in `config/config.yaml`,
because the provider is a per-machine choice. Leave it unset to use the YAML
value.

### Local model (Ollama) — no API key, no account, no payment

Use this when a cloud API is not an option — for example when Anthropic and
OpenAI are unavailable in your country — or simply to spend nothing. Everything
runs on your own machine.

1. Install Ollama from <https://ollama.com/download> (Windows, macOS, Linux).
2. Pull a model that fits your RAM/VRAM:

   ```bash
   ollama pull qwen2.5:7b
   ```

   A 7B model (about 5 GB) runs on most laptops; 12–14B models (`gemma3:12b`,
   `qwen2.5:14b`, about 10 GB) write noticeably better scripts if you have the
   memory. Prefer plain instruct models: "reasoning" models that print their
   thinking can confuse the JSON parsing.
3. In `.env`:

   ```bash
   LLM_PROVIDER=ollama
   LLM_MODEL=qwen2.5:7b
   ```

   No key is needed. `LLM_MODEL` must name a model you have pulled
   (`ollama list` shows them). Ollama is expected at
   `http://localhost:11434/v1`; set `OPENAI_BASE_URL` only if it runs elsewhere,
   such as `http://192.168.1.50:11434/v1` on another machine.
4. Run `shorts-agent doctor`. For Ollama it checks that the server is running and
   that your model is pulled, and tells you the exact command if not.

Ollama support uses the `llm-openai` extra (included in `all`).

What to expect: a local model is slower than a cloud API (on a CPU a single call
can take a minute or more) and less consistent, so an occasional retry is
normal — the client re-prompts when a model returns badly formatted JSON. For
languages other than English, check the quality of the generated script before
committing to a model. If long prompts seem to be cut off, raise Ollama's context
window (for example `PARAMETER num_ctx 8192` in a Modelfile).

### Other OpenAI-compatible servers

LM Studio, vLLM, a llama.cpp server, or a hosted gateway work through the
`openai` provider with a custom endpoint:

```bash
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:1234/v1     # LM Studio's default
LLM_MODEL=<the model name your server uses>
# OPENAI_API_KEY only if the service requires one
```

Verify with `shorts-agent ideate` — it makes exactly one LLM call and produces
no video.

---

## 3. Stock footage (optional but recommended)

Without a key the agent renders gradient cards. Real footage performs better.

### Pexels (free)

1. <https://www.pexels.com/api/> → **Get Started**, sign up.
2. Copy the API key from your dashboard.
3. In `.env`: `PEXELS_API_KEY=...`
4. In `config/config.yaml`: `providers.visuals: "pexels"`

### Pixabay (free)

1. Create an account at <https://pixabay.com>.
2. <https://pixabay.com/api/docs/> shows your key while signed in.
3. In `.env`: `PIXABAY_API_KEY=...`
4. In `config/config.yaml`: `providers.visuals: "pixabay"`

Both are rate-limited (roughly 200 requests/hour on Pexels' free tier). One video
uses one request per scene, so a 8-scene video costs 8 requests.

---

## 4. YouTube Data API key (optional, for trend signals)

Read-only trend research needs only an API key — no OAuth.

1. Open <https://console.cloud.google.com> and create a project.
2. **APIs & Services → Library → YouTube Data API v3 → Enable**.
3. **APIs & Services → Credentials → Create credentials → API key**.
4. Restrict the key to the YouTube Data API (recommended).
5. In `.env`: `YOUTUBE_API_KEY=...`

**Quota:** the default allowance is 10,000 units/day. This project's trend
research uses the `mostPopular` chart, which costs **1 unit** per call. Keyword
search costs **100 units** per call and is therefore off by default — a handful
of searches can consume a whole day's quota.

Verify with `shorts-agent trends`; you should see live video titles alongside
your curated keywords.

---

## 5. YouTube OAuth (only for uploading)

Uploading acts on behalf of a channel, so an API key is not enough.

1. In the same Google Cloud project: **APIs & Services → OAuth consent screen**.
   - User type: **External**.
   - Fill in the app name and your email; you can leave most fields blank.
   - Under **Test users**, add the Google account that owns your channel.
   - Leave the app in **Testing** — you do not need Google verification for your
     own channel. Note that refresh tokens for apps in Testing expire after 7
     days, so you will re-run `shorts-agent auth` periodically. Publishing the
     app removes that expiry but may trigger a verification review.
2. **Credentials → Create credentials → OAuth client ID**.
   - Application type: **Desktop app**.
3. Download the JSON and save it as `client_secret.json` in your project root.
4. In `.env`:

```bash
YOUTUBE_CLIENT_SECRETS_FILE=./client_secret.json
YOUTUBE_TOKEN_FILE=./youtube_token.json
```

5. Authorize once:

```bash
shorts-agent auth
```

A browser opens; grant access to the channel you want to post to. The token is
cached in `youtube_token.json` with owner-only permissions and refreshed
automatically.

> **`client_secret.json` and `youtube_token.json` are already in `.gitignore`.**
> The token grants write access to your channel — treat it like a password.

**Quota:** an upload costs roughly 100 units and draws on a dedicated daily
allocation of about 100 uploads/day. The `publishing.max_uploads_per_day`
setting is a much lower cap on top of that, and exists to protect your channel
from spam-policy problems, not just to protect quota.

### Verifying an upload before trusting the pipeline

Run once with the safest settings, which are also the defaults:

```yaml
publishing:
  privacy_status: "private"
  auto_publish: false
  max_uploads_per_day: 1
```

```bash
shorts-agent run              # renders only
shorts-agent publish <run-id> # uploads as private
```

Then open YouTube Studio and confirm: the video is private, the title and
description are what you expected, and **"Altered or synthetic content" is
declared** (see [POLICY.md](POLICY.md)).

---

## 6. ElevenLabs (optional)

```bash
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM   # optional; defaults to "Rachel"
```

Trade-off: better voices, but timings are derived from the rendered audio's
measured duration rather than reported by the service, so individual caption
highlights can drift within a scene. Keep `edge` if caption precision matters
more — it reports boundary events, per word where the voice supports it and
otherwise per sentence.

---

## 7. Background music (optional)

Drop licensed tracks into `config/music/`. The first file alphabetically is used
and mixed at `visuals.music_volume_db` (default −18 dB) under the narration.

Only use music you have the right to use. YouTube's Content ID will flag
commercial music, and a claim can block or demonetize the video. The YouTube
Audio Library is a safe starting point.

---

## 8. Non-English channels

The pipeline works in any language. Two settings must agree, and nothing
downstream can detect it if they do not — an English voice reading Russian text
produces confident-sounding nonsense:

```yaml
channel:
  language: "ru"
  niche: "личные финансы для начинающих"
  persona: >
    Спокойный финансовый коуч, объясняет один практический совет за видео.

providers:
  edge_tts_voice: "ru-RU-DmitryNeural"   # must match channel.language
```

A mismatch logs a warning at the start of production. List available voices with
`edge-tts --list-voices` (there are voices for 70+ locales, all keyless).

Two things to check for a non-Latin script:

- **Captions.** The font must cover your script. `DejaVu Sans` covers Cyrillic,
  Greek and most European scripts; for CJK, Arabic, Devanagari or Thai, set
  `captions.font` to a font that covers it (e.g. `Noto Sans CJK`, `Noto Sans
  Arabic`) and confirm it is installed with `fc-list : family`.
- **Line length.** `MAX_CHARS_PER_LINE` in `captions/ass_builder.py` is tuned for
  Latin text. CJK characters are wider, so lower it if captions overflow.

Trend signals also need the right region: set `trends.youtube_region_code` to
your audience's country (e.g. `RU`, `DE`, `BR`) so the trending chart and Google
Trends reflect it.

---

## 9. Configure your channel

The single highest-leverage edit in the whole project is `config/config.yaml`:

```yaml
channel:
  niche: "personal finance tips for young adults"
  persona: >
    An upbeat, concise finance coach who explains one practical money tip per
    video in plain language, no jargon.
```

Vague values ("make videos about business") produce generic scripts. Specific
values ("one under-60-second explanation of a single tax rule for freelancers in
their first year") produce sharp ones.

Tune these next:

```yaml
content:
  target_duration_seconds: 45   # 30-60s performs best on Shorts
  ideas_per_run: 5              # more ideas = more to choose from, more tokens
  words_per_second: 2.3         # raise if your voice reads fast

publishing:
  privacy_status: "private"     # keep a human in the loop
  max_uploads_per_day: 2
  default_publish_delay_hours: 4
```

---

## Troubleshooting

**Start with `shorts-agent doctor`.** It checks ffmpeg and libass, the caption
font, every configured key, OAuth state, and the config values that quietly
degrade output — and tells you the fix for each. It makes no API calls, so it is
free to run at any time.

**`Config file not found`** — run `shorts-agent init`, or point
`SHORTS_AGENT_CONFIG` at your file.

**`ANTHROPIC_API_KEY is not set`** — `.env` must be in the directory you run
from. Confirm with `python -c "from shorts_agent.config import get_settings; print(bool(get_settings().anthropic_api_key))"`.

**No trend signals** — expected without `YOUTUBE_API_KEY`. Add keywords to
`config/manual_trends.yaml`; that provider never fails.

**Captions missing from the output** — your ffmpeg lacks libass, or the font in
`config.yaml` is not installed. Check `ffmpeg -filters | grep ass` and
`fc-list : family`.

**`certificate verify failed` during TTS** — you are behind a TLS-inspecting
proxy. Point the standard CA variables at your corporate bundle; `edge-tts`
reads certifi's bundle specifically, so that bundle needs your CA appended.

**Upload fails with `publish_at requires privacy_status='private'`** — working
as intended. YouTube silently ignores a scheduled time on a public video, so the
uploader refuses rather than publishing immediately by accident.

**Refresh token expired after a week** — your OAuth app is in Testing mode. Re-run
`shorts-agent auth`, or publish the app.
