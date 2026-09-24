# Recitation Auto-Checker

Automated tajweed/recitation error detection, built on Tarteel AI's
open-source `tarteel-ai/whisper-base-ar-quran` model (MIT-licensed
fine-tune of Whisper for Quranic Arabic). Transcribes a submitted
recitation, diffs it word-for-word against the expected ayah text, and
classifies mismatches as skipped/wrong/extra words — writing results
into the same `is_recitation_errors` table the app's manual word-pinning
already uses, tagged `source: 'auto'`.

**What this can and can't do**: it catches *content* errors — skipped
words, wrong words, extra words, i.e. memorisation/recitation accuracy.
It does **not** detect acoustic tajweed rule violations (madd duration,
ghunnah nasalization, etc.) — Whisper outputs words, not fine-grained
pronunciation analysis. Those still need a human ear, via the manual
pinning tool already in the app.

## Setup

```bash
py -3.11 -m pip install -r requirements.txt
```

ffmpeg is also required (already installed on this machine) — the app
records audio as `webm/opus`, and librosa needs ffmpeg to decode that
reliably.

Copy `.env.example` to `.env` and fill in your Supabase project's URL
and **service role key** (Project Settings → API — the "service_role"
secret key, not the anon key). This key bypasses RLS, so it must never
be committed, shared, or put in the browser-facing app — it only
belongs in this local `.env` file, which is gitignored.

## Usage

Standalone, on any audio file:

```bash
py -3.11 recitation_checker.py path/to/audio.mp3 112 1 1
# args: audio_file  surah_number  ayah_start  ayah_end
```

As a service, wired to a real submitted recitation:

```bash
py -3.11 -m uvicorn service:app --reload --port 8090
```

```bash
curl -X POST http://localhost:8090/check-recitation \
  -H "Content-Type: application/json" \
  -d "{\"recitation_id\": \"<uuid from is_recitations>\"}"
```

This downloads the student's submitted audio from Supabase Storage,
runs detection, and inserts the results into `is_recitation_errors` —
they'll then show up automatically in the app's Review Recitations
panel and in exported reports, right alongside any manually pinned
errors.

## What's validated so far

Tested against real reciter audio (Alafasy, Surah Al-Ikhlas):
- A correct recitation → zero false errors.
- A deliberately different verse → correctly flagged the actual
  skipped/wrong words, while still recognising the one word that
  genuinely matched.

Two real bugs were found and fixed during testing — both matter for
any Whisper + Quran-text-diff approach, not just this implementation:
1. This fine-tune's checkpoint leaks its own special tokens
   (`<|ar|><|transcribe|>...`) into decoded text — `skip_special_tokens`
   doesn't catch them since they're not marked "special" in its
   tokenizer config. Stripped via regex.
2. Uthmani/Indo-Pak Quran text uses different Unicode forms than
   Whisper's output (e.g. alef *wasla* ٱ vs plain ا), plus the
   Indo-Pak source embeds invisible formatting characters and
   standalone pause marks as fake "words". All of this looks identical
   on screen but breaks literal string comparison — normalized before
   diffing.

## Not yet done

- **Not wired into the app's submit flow.** Right now this runs
  on-demand (`POST /check-recitation` with a specific `recitation_id`).
  Making it fire automatically when a student submits a recitation
  needs this service to be deployed somewhere reachable (it can't run
  inside the static-HTML app or a browser) — a real hosting decision
  for you to make (a small VM, a container host, etc.), not something
  built yet.
- **Real-time/live feedback while reciting** was in an earlier draft
  script but has an architectural issue (comparing short rolling audio
  windows against the *whole* expected text causes false "skipped
  word" errors) — deliberately not built for that reason. The
  submit-then-check flow above avoids the problem entirely.
