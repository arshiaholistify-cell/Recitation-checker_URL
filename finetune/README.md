# Quran Whisper fine-tuning (Hugging Face Space, one-off GPU job)

Continues fine-tuning `tarteel-ai/whisper-base-ar-quran` (the model the
deployed auto-checker already uses) on a curated set of public reciter
audio, aimed at the specific weakness found through empirical testing:
worse accuracy on longer/less-common passages (Al-Baqarah, Al-Kahf,
Al-Ahzab, Yaseen). No manual transcription needed — the Quran's text is
fixed and already known, so `build_dataset.py` pairs public reciter
audio directly with the matching verse text.

This whole pipeline was tested locally end-to-end (data download,
alignment, feature extraction, one training step, eval, save) before
being handed off here — see the project's session notes. What's
**not** yet verified is a real multi-epoch training run on a GPU, or
the actual resulting model's accuracy, since that requires the paid
GPU step below.

## What this costs

- **Hugging Face PRO**: $9/month, required just to create a Docker
  Space at all (the CPU-only "Basic" hardware tier is then free).
- **GPU time during training only**: ~$1–10 on the recommended
  bounded first-pass dataset scope (T4-small at $0.40/hr, or
  A10G-small at $1/hr — see conversation for the fuller estimate).
  You switch the Space to a GPU tier only for the training run itself,
  then back to free CPU Basic (or pause it) once done — GPU tiers
  bill per minute while the hardware is set to that tier, not per
  month.

## One-time setup

1. **Enable HF PRO**: on huggingface.co, go to your account settings →
   Billing, and subscribe to PRO. (Required before step 2 will work.)

2. **Create a write-access token**: go to
   https://huggingface.co/settings/tokens → "New token" → give it
   **write** access → copy it somewhere safe. You'll paste this into
   the Space's secrets in step 5, not anywhere in code.

3. **Create the Space**: on huggingface.co, click your profile →
   "New Space". Fill in:
   - Space name: e.g. `quran-whisper-finetune`
   - SDK: **Docker**
   - Hardware: leave as the default free **CPU Basic** for now (you
     only switch to GPU right before starting training, in step 7)
   - Visibility: **Private** (this only needs to run once for you)

4. **Upload these 5 files** to the new Space: go to the Space's
   **Files** tab → "Add file" → "Upload files", and upload all of:
   `Dockerfile`, `requirements.txt`, `build_dataset.py`, `train.py`,
   `app.py` (everything in this `finetune/` folder except this
   README). This triggers a build automatically — wait for it to show
   "Running" before continuing (it should run fine on CPU Basic since
   nothing tries to train yet at this point).

5. **Set secrets and variables**: go to the Space's **Settings** tab:
   - Under **Variables and secrets** → **New secret**: name
     `HF_TOKEN`, value = the write token from step 2.
   - **New variable**: name `HUB_MODEL_ID`, value =
     `<your-hf-username>/whisper-base-ar-quran-finetuned-v1` (or
     whatever name you'd like the resulting model repo to have — it
     will be created automatically, and is pushed **private**).

## Running the actual training

6. Go to the Space's **Settings** tab → find the hardware section
   (Space hardware / GPU upgrade) → switch to a GPU tier — **T4** is
   the cheapest that should work fine for a "base"-size model.

7. The Space restarts automatically on a hardware change. Once it
   shows "Running" again, training starts on its own (see `app.py`) —
   no button to click. Watch progress two ways:
   - **Space logs** (there's a "Logs" or similar tab) — the same kind
     of live log output used throughout this project's Railway
     debugging.
   - The Space's own URL, root path (`/`) — returns a small JSON
     status: `building_dataset` → `training` → `complete` (or
     `failed`, with a `detail` field — check logs for the full error).

8. **Once status shows `complete`**: go back to Settings → hardware →
   switch back to **CPU Basic** (or pause the Space entirely) so GPU
   billing stops. The trained model is now live at
   `https://huggingface.co/<HUB_MODEL_ID>` (private, visible only to
   your account).

## Using the fine-tuned model in the deployed app

Once you've actually evaluated the new model and decided it's worth
switching to (compare its `eval_wer` in the training logs against the
current model's known behavior on the test recordings from this
session — a lower WER on held-out data is a good sign, but the real
test is trying it against those same hard recordings):

1. In `recitation-checker/recitation_checker.py`, change:
   ```python
   MODEL_NAME = "tarteel-ai/whisper-base-ar-quran"
   ```
   to your new model's id, e.g. `"<your-username>/whisper-base-ar-quran-finetuned-v1"`.

2. Since the model repo is private, the Railway service also needs an
   `HF_TOKEN` (a **read**-scope token is enough) added as a Railway
   service variable, so `from_pretrained()` can authenticate to
   download it.

3. Commit, push, let Railway redeploy, and re-run the same empirical
   tests done earlier in this project (the Yaseen/Kahf/Baqarah
   recordings) to confirm it's actually better before trusting it.
