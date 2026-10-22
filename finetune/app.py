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

Before training, every run verifies the checkpoints on persistent
storage and quarantines any that cannot be loaded — see
verify_checkpoints(). No variable needed; the report lands on the status
page and in the logs either way. That report now also carries the
surviving checkpoint's step count and its full eval WER history, so
"did this training help, and at which step was it best" is answerable
from the status page alone.

If that checkpoint has already completed the epochs being asked for,
the status goes to "already_complete" and training is SKIPPED, because
resuming a finished run trains nothing while still billing GPU time.
Raise the EPOCHS variable to continue it, clear the checkpoints to start
over, or set FORCE_TRAIN=true to run regardless.

Once the status page shows "complete" (training done) or "ready_for_gpu"
(phase 1 done, waiting on you to switch hardware), or "failed" — check
logs — switch the Space's hardware back to CPU Basic (free) or pause it
if nothing further is queued, since GPU tiers bill per minute while
running.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import traceback

from fastapi import FastAPI
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

app = FastAPI()
STATE = {"status": "starting", "detail": ""}

OUTPUT_DIR = "/data/whisper-finetune-out" if os.path.isdir("/data") else "./whisper-finetune-out"

# How many epochs this Space is asking for. Passed to train.py explicitly
# on every run rather than left to its default, so that this is the only
# place the number lives: the completion check below compares a
# checkpoint's progress against it, and a check reading one number while
# training reads another is worse than no check at all. Override with an
# EPOCHS Space variable to extend a finished run.
EPOCHS = int(os.environ.get("EPOCHS", "2"))

# Keep the memory-mapped Arrow cache on the container's OWN disk, not
# on /data.
#
# Spaces points HF_HOME at persistent storage when it is attached, which
# puts both the downloaded parquet and the Arrow files datasets converts
# them into on /data. /data is a network-backed FUSE mount: statvfs on it
# reports a fictional 42 PiB, so a quota ceiling there is invisible, and
# datasets memory-maps those Arrow files rather than reading them. When a
# page fault on an mmap'd region cannot be served — a storage hiccup, a
# quota wall, a file truncated by an earlier interrupted write — the
# kernel delivers SIGBUS and the process dies instantly with no
# traceback. That is exactly how training kept dying, immediately after
# the dataloader started touching feature pages.
#
# Local disk has hundreds of GiB free and mmaps normally, so the Arrow
# cache lives there. Where the downloaded parquet goes is a separate
# decision, taken just below: that one goes ON /data, because it is not
# memory-mapped and losing it to a container recycle costs a ~15 GB
# re-download. Checkpoints stay on /data too — they are the thing that
# actually has to survive.
def probe_storage(path):
    """Write and read back a byte, because /data lies in both directions.

    statvfs on that FUSE mount reports a fictional 42 PiB whether it is
    healthy or dead, and os.path.isdir() still answers True for a mount
    whose daemon has disconnected. Only touching it tells the truth: a
    dead mount raises OSError 107, ENOTCONN ("Transport endpoint is not
    connected"), which is the readable half of the same fault that
    reaches memory-mapped reads as an unsurvivable SIGBUS.
    """
    if not os.path.isdir(path):
        return "absent"
    probe = os.path.join(path, ".storage-probe")
    try:
        with open(probe, "w") as handle:
            handle.write("ok")
        with open(probe) as handle:
            got = handle.read()
        os.remove(probe)
    except OSError as exc:
        return f"UNUSABLE — {type(exc).__name__}: {exc}"
    return "writable" if got == "ok" else "UNUSABLE — read-back mismatch"


# The two caches are split deliberately, because they fail differently.
#
# The platform recycles this container mid-run — the logs carry the
# literal "an error occurred when try to find container ...: not found"
# that train.py's comments describe. Everything on /tmp dies with it, so
# putting the ~15 GB of downloaded parquet there means re-downloading the
# whole dataset on every recycle, and training never reaches its first
# checkpoint. That download therefore goes on /data, which survives.
#
# The Arrow files datasets converts parquet into are a different matter:
# those are memory-mapped, and a page fault that cannot be served — which
# is what a teardown does to the /data FUSE mount while this process is
# still alive — delivers SIGBUS, killing training outright with no
# traceback. They stay on local disk, where a mount cannot vanish
# underneath them. Re-converting after a recycle costs a couple of
# minutes; re-downloading costs the whole run.
#
# HF_HOME is set too, not just the individual caches: several paths are
# DERIVED from it, HF_XET_CACHE ($HF_HOME/xet) among them, and the
# Space's own HF_HOME variable points at /data.
LOCAL_HF_CACHE = "/tmp/hf-cache"
PERSISTENT_HF_CACHE = "/data/hf-cache"

os.environ["HF_HOME"] = LOCAL_HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(LOCAL_HF_CACHE, "datasets")
os.environ["HF_XET_CACHE"] = os.path.join(LOCAL_HF_CACHE, "xet")
# Only trust /data if it answers a real write; a dead mount still passes
# isdir(), and falling back to local disk costs a re-download but keeps
# the run alive, which is the better failure.
if probe_storage("/data") == "writable":
    os.environ["HF_HUB_CACHE"] = os.path.join(PERSISTENT_HF_CACHE, "hub")
else:
    os.environ["HF_HUB_CACHE"] = os.path.join(LOCAL_HF_CACHE, "hub")
for _cache in ("HF_DATASETS_CACHE", "HF_HUB_CACHE", "HF_XET_CACHE"):
    os.makedirs(os.environ[_cache], exist_ok=True)

# Run the integrity check in a CHILD process, never in this one. A
# truncated memory-mapped file does not raise a Python exception when
# read — it delivers SIGBUS, which kills the process outright. In-process
# that would take the whole Space down with no report; in a child, the
# death shows up as a return code that describe_exit() names, which is
# exactly the evidence we are after.
_SAFETENSORS_CHECK = (
    "import os, sys\n"
    "from safetensors.torch import load_file\n"
    "path = sys.argv[1]\n"
    "print('file size: %d bytes' % os.path.getsize(path), flush=True)\n"
    "tensors = load_file(path)\n"
    "print('loaded %d tensors, SAFETENSORS OK' % len(tensors), flush=True)\n"
)


def describe_exit(code):
    """Render a subprocess return code so a crash is distinguishable
    from a kill. A process killed by a signal prints nothing of its own
    — no traceback, no last words — so without this the logs look
    identical to a step that stopped mid-sentence. SIGKILL means the
    kernel's OOM killer; SIGBUS means a memory-mapped file could not be
    served (truncated file, or the filesystem behind it out of space)."""
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        hints = {"SIGKILL": " (kernel OOM killer)", "SIGBUS": " (bad mmap: truncated file or full disk)"}
        return f"killed by {name}{hints.get(name, '')}"
    return f"exit code {code}"


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
        # Kept generous: tqdm redraws can fill this buffer on their own
        # and push a real traceback out of it.
        tail = tail[-200:]
    proc.wait()
    tail.append(f"[process exited: {describe_exit(proc.returncode)}]")
    return proc.returncode, "\n".join(tail)


def _disk_report():
    lines = ["caches:"]
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        lines.append(f"  {var}={os.environ.get(var, '(unset)')}")
    # /data is a FUSE mount whose statvfs numbers are fictional, so its
    # row below says nothing about the real quota — read it as "present",
    # not as free space.
    lines.append(f"persistent storage (/data): {probe_storage('/data')}")
    lines.append("disk:")
    for path in ("/", "/data", "/tmp", LOCAL_HF_CACHE, os.path.expanduser("~/.cache/huggingface"), "/dev/shm"):
        if not os.path.exists(path):
            lines.append(f"  {path}: (absent)")
            continue
        usage = shutil.disk_usage(path)
        pct = 100.0 * usage.used / usage.total if usage.total else 0.0
        lines.append(f"  {path}: {usage.free / 1024 ** 3:.1f} GiB free of {usage.total / 1024 ** 3:.1f} GiB ({pct:.0f}% used)")
    return lines


def _dir_size(path):
    total = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            full = os.path.join(root, name)
            if os.path.isfile(full):
                total += os.path.getsize(full)
    return total


def _tree_report(path):
    lines = [f"contents of {path}:"]
    if not os.path.isdir(path):
        lines.append("  MISSING — no local checkpoints on persistent storage")
        return lines
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            lines.append(f"  {name}/  {_dir_size(full) / 1024 ** 2:.0f} MiB")
        else:
            lines.append(f"  {name}  {os.path.getsize(full) / 1024 ** 2:.1f} MiB")
    return lines


def _checkpoint_dirs(path):
    """Checkpoint directories, oldest first, by step number."""
    if not os.path.isdir(path):
        return []
    found = []
    for name in os.listdir(path):
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        if match and os.path.isdir(os.path.join(path, name)):
            found.append((int(match.group(1)), os.path.join(path, name)))
    return [full for _step, full in sorted(found)]


def _quarantine(path):
    """Rename a checkpoint out of the way instead of deleting it.

    get_last_checkpoint() in train.py only recognises directories named
    checkpoint-<step>, so a renamed one is invisible to the resume logic
    while staying on disk for inspection. Renaming also cannot lose work
    the way rm -rf can, which matters when the thing being judged
    "corrupt" is the only record of 2,450 steps of training."""
    target = path.replace("checkpoint-", "corrupt-checkpoint-")
    suffix = 1
    while os.path.exists(target):
        suffix += 1
        target = path.replace("checkpoint-", "corrupt-checkpoint-") + f".{suffix}"
    os.rename(path, target)
    return target


def verify_checkpoints():
    """Find the newest checkpoint that actually loads, newest first.

    A checkpoint whose model.safetensors was truncated — by a container
    kill landing mid-save, or by the filesystem filling up — does not
    fail cleanly. safetensors memory-maps the file, and reading past a
    truncated end delivers SIGBUS, killing the process with no traceback
    at all. That is indistinguishable from a hang in the logs, and it
    repeats on every restart, so the run can never get past its own
    resume.

    Each candidate is therefore loaded in a CHILD process: a SIGBUS
    there is contained and reported as a return code rather than taking
    this Space down. Anything that fails is renamed aside so the next
    restart resumes from the newest checkpoint that works.

    Returns (report_lines, usable_checkpoint_or_None).
    """
    report = _disk_report() + [""] + _tree_report(OUTPUT_DIR) + [""]
    candidates = list(reversed(_checkpoint_dirs(OUTPUT_DIR)))
    if not candidates:
        report.append("no local checkpoints — train.py will try the Hub, or start fresh")
        return report, None

    for path in candidates:
        weights = os.path.join(path, "model.safetensors")
        if not os.path.isfile(weights):
            reason = "no model.safetensors"
        else:
            code, tail = run_streamed([sys.executable, "-c", _SAFETENSORS_CHECK, weights])
            if code == 0:
                report.append(f"{os.path.basename(path)}: OK — resuming from this one")
                return report, path
            reason = describe_exit(code)
            report.extend("  " + line for line in tail.splitlines())
        try:
            moved = _quarantine(path)
            report.append(f"{os.path.basename(path)}: UNUSABLE ({reason}) — renamed to {os.path.basename(moved)}")
        except OSError as exc:
            report.append(f"{os.path.basename(path)}: UNUSABLE ({reason}) — could NOT rename it ({exc})")
            return report, None

    report.append("every local checkpoint failed to load — train.py will try the Hub, or start fresh")
    return report, None


def training_progress(checkpoint):
    """What a checkpoint says about how far its run actually got.

    trainer_state.json sits in every checkpoint and carries the whole
    history — global_step, how many epochs were asked for and how many
    completed, and every eval logged along the way. It costs nothing to
    read and answers the question the logs never did: is there any work
    left to do?

    Returns the parsed state, or None if it cannot be read (an old
    checkpoint predating the file, or one truncated by the same kind of
    interrupted write verify_checkpoints() exists to catch).
    """
    path = os.path.join(checkpoint, "trainer_state.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"could not read {path}: {type(exc).__name__}: {exc}", flush=True)
        return None


def describe_progress(state):
    """Summarise a trainer_state.json, eval history included.

    The WER history is the whole point of printing this. It is recorded
    in every checkpoint, so the question this project has actually been
    asking — did fine-tuning help, and at which step was the model at
    its best — is answerable from disk, with no GPU and no inference
    run. Before this it was only ever reachable by loading the model and
    re-evaluating it.
    """
    lines = []
    step = state.get("global_step")
    total = state.get("max_steps")
    epoch = state.get("epoch")
    lines.append(f"progress: step {step} of {total}" + (f", {epoch:.2f} epochs done" if epoch is not None else ""))

    evals = [(entry.get("step"), entry["eval_wer"]) for entry in state.get("log_history", []) if "eval_wer" in entry]
    if not evals:
        lines.append("no eval recorded in this checkpoint")
        return lines
    best_step, best_wer = min(evals, key=lambda pair: pair[1])
    last_step, last_wer = evals[-1]
    lines.append(f"best eval WER {best_wer:.2f}% at step {best_step} ({len(evals)} evals recorded)")
    lines.append(f"last eval WER {last_wer:.2f}% at step {last_step}")
    if best_step != last_step and last_wer > best_wer:
        lines.append(
            f"  NOTE: WER got worse after step {best_step} (+{last_wer - best_wer:.2f} points by step {last_step}). "
            "The final-step weights are not the best ones this run produced."
        )
    return lines


def dataset_already_processed(hub_dataset_id):
    """True if the processed dataset is already on the Hub.

    Only a genuine "not found" counts as "needs phase 1". Anything else
    — an expired HF_TOKEN, a Hub 5xx — is allowed to propagate: on a GPU
    tier, mistaking a transient lookup failure for a fresh run silently
    starts a ~day-long CPU preprocessing job on billed GPU hardware.
    Note that the Hub also answers 404 for a private repo the current
    token cannot see, so a token problem can still surface here as
    RepositoryNotFoundError — if this reports "building_dataset" on a
    Space you know has already been preprocessed, suspect the token
    before suspecting the dataset."""
    HfApi().dataset_info(hub_dataset_id)
    return True


def run_job():
    """Thread entrypoint — keeps STATE truthful if _run_job raises.

    Without this, an exception in the job thread leaves the status page
    reporting "starting" forever while nothing is running, which reads
    exactly like a hung step."""
    try:
        _run_job()
    except Exception:
        STATE["status"] = "failed"
        STATE["detail"] = "Job crashed:\n" + traceback.format_exc()


def _run_job():
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

    try:
        processed = dataset_already_processed(hub_dataset_id)
    except RepositoryNotFoundError:
        processed = False

    if not processed:
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
    storage = probe_storage("/data")
    if storage.startswith("UNUSABLE"):
        # Training would write every checkpoint to this mount, so there
        # is nothing to be gained by starting: it dies minutes later with
        # a SIGBUS that looks like a hang. Say so plainly instead.
        STATE["status"] = "failed"
        STATE["detail"] = (
            f"Persistent storage /data is not usable: {storage}\n\n"
            "This is the Space's storage mount, not this code. A dropped FUSE mount does not "
            "recover in place — RESTART the Space to get a fresh container. If it keeps "
            "happening, move training output off /data and rely on the Hub checkpoint pushes."
        )
        return

    STATE["status"] = "checking_checkpoints"
    report, usable = verify_checkpoints()

    # train.py resumes from whichever checkpoint survived that check, so
    # that is the one to ask how far the run got — and the answer decides
    # whether there is anything to run at all.
    #
    # A checkpoint that already reached the epoch budget makes train.py a
    # no-op: Trainer resumes, finds global_step at max_steps, iterates an
    # empty range of epochs, returns without taking a single step, and
    # pushes back the model it just loaded. Nothing about that is visible
    # from outside — the status page goes "training" then "complete"
    # exactly as a real run does — and on a GPU tier it bills by the
    # minute to do it. Worse, it hides config changes: an edit to the
    # eval schedule or the learning rate looks applied and silently never
    # runs. Say so and stop instead.
    epochs_done = None
    state = training_progress(usable) if usable else None
    if state:
        report.extend(describe_progress(state))
        epochs_done = state.get("epoch")
    checkpoint_report = "\n".join(report)
    print(checkpoint_report, flush=True)
    STATE["detail"] = checkpoint_report

    force_train = os.environ.get("FORCE_TRAIN", "").lower() in ("1", "true", "yes")
    if epochs_done is not None and epochs_done >= EPOCHS and not force_train:
        STATE["status"] = "already_complete"
        STATE["detail"] = (
            f"{os.path.basename(usable)} has already completed {epochs_done:.2f} of the {EPOCHS} epochs "
            "requested, so training would run 0 steps and re-push the same model.\n\n"
            "Nothing to do here — switch this Space's hardware back to CPU Basic, or pause it.\n\n"
            f"To train further, set an EPOCHS Space variable higher than {EPOCHS} and restart. To start a "
            f"fresh run under changed settings, move the checkpoint-* directories out of {OUTPUT_DIR} "
            "first. To run anyway, set FORCE_TRAIN=true.\n\n"
            + checkpoint_report
        )
        return

    STATE["status"] = "training"
    code, tail = run_streamed(
        [
            sys.executable, "train.py",
            "--hub-dataset-id", hub_dataset_id,
            "--hub-model-id", hub_model_id,
            # Passed explicitly so EPOCHS above is the only place this
            # number lives; train.py's own default is never relied on.
            "--epochs", str(EPOCHS),
        ]
    )
    if code != 0:
        STATE["status"] = "failed"
        STATE["detail"] = "Training failed:\n" + tail + "\n\n--- checkpoint check before this run ---\n" + checkpoint_report
        return

    STATE["status"] = "complete"
    STATE["detail"] = f"Model pushed to https://huggingface.co/{hub_model_id}"


threading.Thread(target=run_job, daemon=True).start()


@app.get("/")
def status():
    return STATE
