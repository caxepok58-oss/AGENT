# Policy, disclosure and safe operation

Automating uploads to a platform you do not control is the risky part of this
project — not the code. This document explains what the defaults protect you
from and what remains your responsibility.

**Nothing here is legal advice, and platform policies change.** Read YouTube's
current [Community Guidelines](https://www.youtube.com/howyoutubeworks/policies/community-guidelines/),
[spam and deceptive practices policy](https://support.google.com/youtube/answer/2801973),
and [advertiser-friendly content guidelines](https://support.google.com/youtube/answer/6162278)
yourself before publishing at any volume.

---

## Why the defaults are conservative

| Default | Reason |
|---|---|
| `run` renders but does not upload | An unreviewed upload cannot be un-published from a viewer's memory. The first videos from a new config are the ones most likely to be wrong |
| `privacy_status: private` | You see each video in Studio before anyone else does |
| `max_uploads_per_day: 2` | A loop that uploads dozens of near-identical videos is the fastest way to trigger the spam policy |
| Policy screening runs before rendering | Catches problems while they cost one LLM call instead of a full render and an upload |
| `ai_disclosure: true` | Synthetic voice and visuals require disclosure. See below |
| Moderation fails closed | If the LLM reviewer is unreachable, the run stops rather than publishing unreviewed content |

Raising any of these is a decision you should make deliberately, after you have
watched several finished videos end to end.

---

## Disclosing AI-generated content

YouTube requires creators to disclose **meaningfully altered or synthetically
generated content that could be mistaken for a real person, place, or event**.
The disclosure appears in the video's description, and for sensitive topics
(health, news, elections, finance) YouTube may show a more prominent label on
the player itself.

This project sets the Data API's `status.containsSyntheticMedia` field from
`content.ai_disclosure` in your config, which is the API equivalent of the
"Altered or synthetic content" question in YouTube Studio.

**Keep `ai_disclosure: true`** if your videos use a synthetic voice, AI-generated
visuals, or AI-written narration presented as a person speaking — which is the
normal case for this pipeline.

What disclosure does **not** cover: it is not a licence to fabricate. A disclosed
video that invents statistics, impersonates a real person, or presents a
made-up event as real still violates policy. The generation prompts in
`src/shorts_agent/ideation/prompts.py` explicitly instruct against invented
statistics and fabricated personal anecdotes, and the moderation pass checks for
them — but prompts are not guarantees. Read what you publish.

Reference: [Disclosing use of altered or synthetic content](https://support.google.com/youtube/answer/14328491).

---

## Spam, repetition and "inauthentic content"

YouTube's spam policy targets, among other things, **mass-produced or repetitive
content** — and monetization policy distinguishes "inauthentic" content from
content with genuine original value. An automated pipeline can drift into that
territory without anyone intending it.

What the project does about it:

- **Near-duplicate detection.** Ideas within 82% similarity of a topic published
  in the last `disallow_duplicate_topics_days` (default 30) are rejected.
- **A daily upload cap**, checked against real upload history before a run
  starts spending anything.
- **Publish spacing.** Scheduled videos are queued at least
  `default_publish_delay_hours` apart, so a burst of runs does not release
  several Shorts at once.
- **Performance feedback.** Ideation is biased toward topics that actually earned
  views, which pushes away from formulaic repetition.

What remains on you:

- **Add something the algorithm cannot.** Your own experience, your own examples,
  your own voice in the persona. A channel of purely synthesized general advice
  is exactly what the inauthentic-content policy is aimed at.
- **Watch your own videos.** If they feel interchangeable to you, they will to
  viewers.
- **Do not run several channels off one config** to multiply output. That is the
  pattern enforcement looks for.

---

## Copyright

- **Stock footage.** Pexels and Pixabay licences permit commercial use without
  attribution, but both prohibit some uses (e.g. redistributing the footage
  itself, or implying endorsement by identifiable people). Read the current
  licence — they change.
- **Music.** Nothing is bundled. Anything you place in `config/music/` is your
  responsibility. Commercial music will be caught by Content ID and can block or
  demonetize the video. Use the YouTube Audio Library or a licensed library.
- **Voices.** Synthetic voices from edge-tts and ElevenLabs are subject to those
  providers' terms. Do not clone a real person's voice without permission.
- **Facts and quotes.** If a script quotes or paraphrases a specific source,
  credit it in the description.

---

## API quotas and rate limits

Exceeding a quota does not just fail a run — sustained abuse can get API access
revoked.

| API | Cost | Notes |
|---|---|---|
| YouTube `videos.list` (trending chart) | 1 unit | The default trend path |
| YouTube `search.list` | 100 units | Off by default; a few calls can exhaust a day |
| YouTube `videos.insert` (upload) | ~100 units, dedicated daily allocation | Roughly 100 uploads/day at the API level, far above this project's cap |
| Google Trends (pytrends) | No official quota | An undocumented endpoint that rate-limits aggressively. Failures are swallowed |
| Pexels | ~200 requests/hour (free) | One request per scene |

The default YouTube Data API allowance is 10,000 units/day per project.

---

## What this project deliberately does not do

- **No engagement manipulation.** No comment bots, no view inflation, no
  sub4sub. That is a straight route to termination.
- **No scraping YouTube's site.** Trend data comes from the official API.
- **No misleading metadata.** The metadata prompt forbids promising something the
  video does not deliver, because it costs viewer trust and, at scale, triggers
  policy action.
- **No impersonation.** Nothing here generates content in a real person's
  likeness or voice.
- **No auto-publish by default.** You have to ask for it, twice.

---

## A minimum responsible workflow

1. Run `shorts-agent run` (no upload) several times. Use
   `shorts-agent preview <run-id>` to see every frame, the script and the
   metadata at a glance — then watch each video fully, because a contact sheet
   cannot show pacing, audio or caption timing.
2. Fix your persona and niche until the output is something you would post under
   your own name.
3. Read this document and the linked YouTube policies.
4. `shorts-agent auth`, then `shorts-agent publish <run-id>` for a single private
   video. Confirm the synthetic-content disclosure appears in Studio.
5. Only then consider `--publish`, and keep `max_uploads_per_day` low.
6. Check `shorts-agent report --refresh` regularly. Falling views on
   near-identical topics is your early warning that the channel is drifting into
   repetitive content.

If you would be uncomfortable telling a viewer how the video was made, do not
publish it.
