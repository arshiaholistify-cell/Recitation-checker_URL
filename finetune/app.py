"""
Space entrypoint: runs the pipeline as a background job on container
start, and serves a small status page in the foreground so the Space
stays in a clean "Running" state instead of crash-looping once the job
finishes (Docker Spaces expect a long-lived process).

The pipeline is two phases across two hardware tiers, not one:
  1. Dataset build + preprocessing (audio decode, feature extraction,
     tokenization) — entirely CPU-bound, never touches torch.cuda, but
     slow (~4s/example single-threaded, ~a day across the full
     dataset). Run this on CPU Basic (free) or another cheap CPU tier.
     Result gets pushed to the Hub as a private dataset repo.
  2. Training — the only phase that actually needs a GPU. Run this on
     a GPU hardware tier.

This file auto-detects which phase to run: on start, it checks whether
HUB_DATASET_ID already exists on the Hub. If not, it does phase 1 and
then STOPS (status: "ready_for_gpu") instead of continuing straight to
training — switch the Space's hardware to a GPU tier and restart it to
run phase 2, which will find the dataset already there and skip
straight to training. This means phase 1 never runs on billed GPU time,
and phase 2 never repeats the slow CPU work if the Space restarts.

Progress is visible in the Space's own Deploy/Build logs (same as the
Railway service's logs during earlier development) — this page is just
a quick way to check status without digging through logs.

Required Space secrets/variables (set in Settings, not in code):
  HF_TOKEN        — write-access token, used to push the dataset and model
  HUB_DATASET_ID  — e.g. "your-username/quran-whisper-finetune-processed"
  HUB_MODEL_ID    — e.g. "your-username/whisper-base-ar-quran-finetuned-v1"

Optional Space variable for one-off dataset repair:
  REPAIR_DATASET  — set to "true" to re-consolidate the processed
                    dataset from its checkpoint chunks (fixes a corrupt
                    shard from an interrupted push) instead of running
                    phase 1 or 2. Remove it and restart again afterward
                    to resume normal operation.

Once the status page shows "complete" (training done) or "ready_for_gpu"
(phase 1 done, waiting on you to switch hardware), or "failed" — check
logs — switch the Space's hardware back to CPU Basic (free) or pause it
if nothing further is queued, since GPU tiers bill per minute while
running.
"""

import os
import subprocess
import sys
import threading

from fastapi import FastAPI
from huggingface_hub import HfApi

app = FastAPI()
STATE = {"status": "starting", "detail": ""}


def run_streamed(cmd):
    """Run a subprocess and print its output line-by-line as it happens,
    instead of buffering everything until the process exits — otherwise
    the Space's Logs tab shows nothing for the entire (potentially
    multi-hour) duration of a step, making it impossible to tell
    progress from a stall. Also captures the tail of that output so a
    failure can be surfaced directly in the status JSON's "detail"
    field — the Container log view gets buried within seconds by the
    browser's own repeated polling of this same status endpoint,
    making it painful to dig a traceback back out of it."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
    )
    tail = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip("\n"))
        tail = tail[-60:]
    proc.wait()
    return proc.returncode, "\n".join(tail)


def dataset_already_processed(hub_dataset_id):
    try:
        HfApi().dataset_info(hub_dataset_id)
        return True
    except Exception:
        # Covers "doesn't exist yet" (the expected case on a fresh
        # phase-1 run) and any transient lookup error alike — either
        # way, the safe thing is to (re)do phase 1 rather than assume
        # a training phase has something valid to load.
        return False


def run_job():
    hub_model_id = os.environ.get("HUB_MODEL_ID")
    hub_dataset_id = os.environ.get("HUB_DATASET_ID")
    if not hub_model_id or not hub_dataset_id:
        STATE["status"] = "failed"
        STATE["detail"] = "HUB_MODEL_ID and/or HUB_DATASET_ID environment variable is not set."
        return

    # One-off repair path: train.py's load_dataset() can hit a corrupt
    # parquet shard in the processed dataset repo if the final
    # push_to_hub() at the end of a prior preprocess.py run got cut off
    # mid-upload by a container restart (see preprocess.py's docstring
    # for that restart pattern) — that push has no resumability of its
    # own, unlike the per-chunk checkpointing that feeds it. Re-running
    # just the consolidation (re-download every checkpoint chunk,
    # re-push a fresh combined dataset) fixes it without redoing any
    # audio decoding. Set REPAIR_DATASET=true as a Space variable,
    # restart, then remove it and restart again to resume training.
    if os.environ.get("REPAIR_DATASET", "").lower() in ("1", "true", "yes"):
        STATE["status"] = "repairing_dataset"
        code, tail = run_streamed(
            [sys.executable, "preprocess.py", "--dataset-dir", "./quran_dataset", "--hub-dataset-id", hub_dataset_id, "--consolidate-only"]
        )
        if code != 0:
            STATE["status"] = "failed"
            STATE["detail"] = "Dataset repair failed:\n" + tail
            return
        STATE["status"] = "repair_complete"
        STATE["detail"] = "Re-consolidated the processed dataset. Remove the REPAIR_DATASET variable and restart to resume training."
        return

    if not dataset_already_processed(hub_dataset_id):
        # Phase 1 — CPU-only, meant to run on cheap/free CPU hardware.
        # Stops here rather than continuing to training: the point is
        # to never let the slow (~a day) preprocessing step run on a
        # billed GPU tier.
        STATE["status"] = "building_dataset"
        code, tail = run_streamed([sys.executable, "build_dataset.py", "--out", "./quran_dataset"])
        if code != 0:
            STATE["status"] = "failed"
            STATE["detail"] = "Dataset build failed:\n" + tail
            return

        STATE["status"] = "preprocessing"
        code, tail = run_streamed(
            [sys.executable, "preprocess.py", "--dataset-dir", "./quran_dataset", "--hub-dataset-id", hub_dataset_id]
        )
        if code != 0:
            STATE["status"] = "failed"
            STATE["detail"] = "Preprocessing failed:\n" + tail
            return

        STATE["status"] = "ready_for_gpu"
        STATE["detail"] = (
            f"Preprocessing done — pushed to https://huggingface.co/datasets/{hub_dataset_id}. "
            "Switch this Space's hardware to a GPU tier and restart it to run training."
        )
        return

    # Phase 2 — the dataset already exists on the Hub (either from an
    # earlier phase-1 run on this same Space before a hardware switch,
    # or reused as-is), so skip straight to the GPU-only training step.
    STATE["status"] = "training"
    code, tail = run_streamed(
        [sys.executable, "train.py", "--hub-dataset-id", hub_dataset_id, "--hub-model-id", hub_model_id]
    )
    if code != 0:
        STATE["status"] = "failed"
        STATE["detail"] = "Training failed:\n" + tail
        return

    STATE["status"] = "complete"
    STATE["detail"] = f"Model pushed to https://huggingface.co/{hub_model_id}"


threading.Thread(target=run_job, daemon=True).start()


@app.get("/")
def status():
    return STATE
