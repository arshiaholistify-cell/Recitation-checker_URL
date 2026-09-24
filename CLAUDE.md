# Recitation-checker_URL

This repository is **empty** — it has no source of its own, and this file is its
first commit. It exists as a placeholder for the recitation auto-checker.

## Where the code actually lives

The working service is in the TarBee repo, at
`Holistify-Islamic-Studies-App/recitation-checker/`:

- `service.py` — FastAPI wrapper; `POST /check-recitation`
- `recitation_checker.py` — transcription and word-level diffing
- `tajweed_checker.py` — acoustic pass
- `Dockerfile` — deployed to Railway, which injects `PORT`

Work on it there, not here, unless and until someone decides to split it out.

## What it does

Built on Tarteel AI's `tarteel-ai/whisper-base-ar-quran` (an MIT-licensed
Whisper fine-tune for Quranic Arabic). It transcribes a submitted recitation,
diffs it word by word against the expected ayah text, and classifies mismatches
as skipped, wrong or extra words. Results are written to `is_recitation_errors`
with `source='auto'` — the same table TarBee's manual word-pinning UI reads.

Scope limits worth knowing before promising anything to a user:

- It catches **content** errors (skipped, wrong, extra words), not acoustic
  tajweed violations like madd duration or ghunnah. Whisper emits words, not
  fine-grained pronunciation analysis; those still need a human ear.
- It checks against **Hafs 'an Asim only**. Other riwayat (Warsh, Qaloon) give
  unreliable results.

A full check takes 2–3+ minutes, too long to hold an HTTP request open, so
`POST /check-recitation` starts background work and returns immediately; the app
polls the `is_auto_check_jobs` table through Supabase directly. Don't turn that
back into a synchronous request.

## Security

**This repository is public.** The service uses a Supabase **service-role key**,
which bypasses RLS. It belongs only in a local `.env` or in Railway's service
variables — never committed here or anywhere else.

## If you're setting this repo up properly

Two options, and the choice hasn't been made yet:

1. **Move** `recitation-checker/` out of the TarBee repo into this one, so the
   Python service versions independently of the web app. The Railway deploy
   would point here instead.
2. **Retire** this repo, and leave the service where it is.

Leaving it empty is the one option that doesn't help anyone — an unexplained
empty repo is a trap for the next person.
