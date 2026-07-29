# Setup

Every integration is optional except an LLM key. Set up what you need and skip
the rest — the pipeline degrades rather than failing.

| You want to… | You need |
|---|---|
| Generate ideas, scripts, metadata | An LLM key (**required**) |
| Voiceover with accurate captions | Nothing — edge-tts is keyless |
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

## 2. LLM key (required)

### Anthropic (default)

1. Sign in at <https://console.anthropic.com>.
2. **API keys → Create key**, copy it.
3. In `.env`:

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-5
ANTHROPIC_API_KEY=sk-ant-...
```

### OpenAI

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
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

Trade-off: better voices, but timings are estimated from the rendered audio
rather than reported per word, so individual caption highlights can drift
slightly within a scene. Keep `edge` if caption precision matters more.

---

## 7. Background music (optional)

Drop licensed tracks into `config/music/`. The first file alphabetically is used
and mixed at `visuals.music_volume_db` (default −18 dB) under the narration.

Only use music you have the right to use. YouTube's Content ID will flag
commercial music, and a claim can block or demonetize the video. The YouTube
Audio Library is a safe starting point.

---

## 8. Configure your channel

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
