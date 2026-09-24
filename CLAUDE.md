# Recitation Auto-Checker

Python service that detects recitation errors in submitted Quran audio and
writes them back into TarBee's database. Moved here from the TarBee repo
(`Holistify-Islamic-Studies-App/recitation-checker/`) so it versions
independently of the web app it serves.

Deployed to **Railway**, separately from TarBee. TarBee reaches it through the
`RECITATION_CHECKER_URL` build placeholder.

## What it does, and doesn't

Built on Tarteel AI's `tarteel-ai/whisper-base-ar-quran`, an MIT-licensed
Whisper fine-tune for Quranic Arabic. It transcribes a recitation, diffs it word
by word against the expected ayah text, and classifies mismatches as skipped,
wrong or extra words. Results go into `is_recitation_errors` with
`source='auto'` — the same table TarBee's manual word-pinning UI reads and
displays.

Two limits that matter before promising anything to a user:

- It catches **content** errors — skipped, wrong, extra words, i.e. memorisation
  and recitation accuracy. It does **not** detect acoustic tajweed violations
  such as madd duration or ghunnah nasalization; Whisper emits words, not
  fine-grained pronunciation analysis. Those still need a human ear through the
  manual pinning tool.
- It checks against **Hafs 'an Asim only**. The model and the reference text are
  both Hafs; other riwayat (Warsh, Qaloon) produce unreliable results.

## Layout

- `service.py` — FastAPI wrapper, `POST /check-recitation`; reads `PORT`
  (Railway injects it, defaults to 8090 locally)
- `recitation_checker.py` — transcription and word-level diffing
- `tajweed_checker.py` — acoustic pass
- `Dockerfile` — the Railway image
- `finetune/` — dataset building, training and evaluation scripts for the model
  fine-tune; not part of the deployed service
- `*.mp3`, `*.webm` — real recitation samples kept as test fixtures

## The async design — don't undo it

A full check (Whisper word-accuracy pass plus acoustic tajweed pass) takes 2–3+
minutes on real audio. That is too long to hold an HTTP request open: something
in the network path — proxy, mobile network or browser — was observed silently
killing the connection well before completion, even though the server kept
computing the whole time.

So `POST /check-recitation` starts the work on a background thread and returns
almost immediately. The frontend polls the `is_auto_check_jobs` table **through
Supabase directly, not through this service**. See
`islamic_studies_auto_check_jobs_migration.sql` in the TarBee repo.

Don't "simplify" this back into a synchronous request.

## Running it

```bash
py -3.11 -m pip install -r requirements.txt
py -3.11 -m uvicorn service:app --reload --port 8090
```

ffmpeg must be on PATH — the app records audio as webm/opus and librosa needs
ffmpeg to decode it reliably. The Dockerfile installs it, and also installs the
CPU-only torch build first, because a plain `pip install torch` on Linux pulls
the multi-gigabyte CUDA wheel this container has no use for.

`transformers` is pinned to `4.57.6`: `quran-muaalem` depends on a private
internal (`_HIDDEN_STATES_START_POSITION`) that the 5.x rewrite removed, so an
unpinned install breaks the import. Don't bump it without testing that.

Standalone use on any audio file:

```bash
py -3.11 recitation_checker.py path/to/audio.mp3 112 1 1
# args: audio_file  surah_number  ayah_start  ayah_end
```

## Security

**This repository is public.** The service authenticates to Supabase with the
**service-role key**, which bypasses RLS entirely. It belongs only in a local
`.env` (gitignored) or in Railway's service variables — never committed here,
never pasted into chat, never shipped to the browser. `.env.example` is the
template and must keep its key field empty.

## Cross-repo note

The web app that consumes this lives in `Holistify-Islamic-Studies-App`
(TarBee). Changes to the response shape, the `is_recitation_errors` columns or
the `is_auto_check_jobs` contract affect both repos — the migrations for those
tables live on the TarBee side.

Commit history before the move stays in the TarBee repo; the shallow clone this
move was made from couldn't carry it across.
