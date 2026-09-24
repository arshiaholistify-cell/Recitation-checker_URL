"""
Splits the CPU-bound preprocessing (audio decode + feature extraction +
tokenization) out of train.py so it can run on cheap/free CPU Space
hardware instead of a billed GPU tier. This step never touches
torch.cuda — it's pure librosa + WhisperProcessor work — but it's slow:
~4s/example single-threaded (num_proc>1 OOM'd on a live run, see the
history in train.py's comments), which across a 31,180-example dataset
is on the order of a day of wall-clock time. Running that on GPU
hardware, which bills per minute regardless of whether the GPU is
doing anything, was pure waste.

Every one of this Space's runs has died around the same ~64% mark —
outlasting a RAM upgrade (16GB to 32GB, no change), subprocess
isolation of the decode (see _get_pool(), no change), and a guard
against oversized/corrupt audio files (no change) — which points at
something restarting the whole container on a fixed time budget,
independent of the code running inside it. No amount of per-example
error handling can survive that, so instead of retrying to fix the
crash, this processes and checkpoints in small chunks: each finished
chunk is uploaded straight to the Hub as soon as it's done, and on
startup this checks what's already there and skips it. A restart now
costs at most one unfinished chunk's worth of work instead of the
entire job.

Once every chunk for both splits is checkpointed, a final
consolidation pass downloads them all and pushes one clean combined
dataset the normal way — train.py's `load_dataset(hub_dataset_id)`
call is unaffected by any of this, since that pass produces the exact
same structure as before. That pass is I/O, not CPU-bound audio
decoding, so it isn't at risk of the same crash.

Usage (run on CPU Space hardware):
    python preprocess.py --dataset-dir ./quran_dataset --hub-dataset-id <your-username>/quran-whisper-finetune-processed
"""

import argparse
import concurrent.futures as cf
import csv
import math
import os

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, hf_hub_download

BASE_MODEL = "tarteel-ai/whisper-base-ar-quran"

# librosa's mp3 backend (libmpg123/audioread, both C libraries) can
# segfault outright on a corrupt file instead of raising a catchable
# Python exception — confirmed the hard way: a previous version of this
# script wrapped the decode in try/except and it STILL died at the same
# example index with no traceback and no "SKIPPING" log line, because
# the crash happens below the Python interpreter. The only way to
# survive that is to isolate the decode in its own OS process so a
# crash there only kills that process, not this one. A single
# persistent worker (recreated whenever it dies) keeps the model
# loaded once instead of reloading it per example.
_POOL = None
_worker_processor = None


def _worker_init(base_model):
    global _worker_processor
    import librosa  # noqa: F401 (imported for its side effect of registering audio backends)
    from transformers import WhisperProcessor

    _worker_processor = WhisperProcessor.from_pretrained(base_model)


# A corrupted audio file whose header reports (or decodes to) an absurd
# duration can make librosa.load() attempt one gigantic allocation —
# still a possible cause of the container-level crash below, even
# though it turned out not to be the whole story. These are short
# verse-by-verse recitations, so a generous size ceiling catches a
# pathological file without ever touching real ones.
MAX_AUDIO_FILE_BYTES = 25_000_000


def _decode_and_extract(audio_path, text):
    import os

    import librosa

    size = os.path.getsize(audio_path)
    if size > MAX_AUDIO_FILE_BYTES:
        raise ValueError(f"audio file is {size} bytes (> {MAX_AUDIO_FILE_BYTES}), likely corrupt — refusing to decode")

    audio, sr = librosa.load(audio_path, sr=16000)
    input_features = _worker_processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
    labels = _worker_processor.tokenizer(text).input_ids
    return input_features, labels


def _get_pool():
    global _POOL
    if _POOL is None:
        _POOL = cf.ProcessPoolExecutor(max_workers=1, initializer=_worker_init, initargs=(BASE_MODEL,))
    return _POOL


# Whisper's decoder has a hard, fixed cap of 448 target positions (same
# across all whisper-base checkpoints, including the tarteel one this
# continues from) — a training example whose tokenized label sequence
# exceeds that crashes the forward pass outright. Filtering rather than
# truncating, since truncating would train the model to produce a
# cut-off transcription for that clip — actively wrong, not just
# imprecise. See train.py's original history for the exact crash this
# avoids ("Labels' sequence length 503 cannot exceed...").
MAX_LABEL_LEN = 448

# Small enough that a container restart loses at most a few seconds of
# work, large enough that checkpoint upload overhead (a few seconds
# each, mostly fixed cost) doesn't dominate. 28,062 train examples is
# ~57 chunks, 3,118 validation is ~7 — cheap to list/skip on resume.
CHUNK_SIZE = 500

CHECKPOINT_PREFIX = "checkpoint"


def _checkpoint_path(split, chunk_idx):
    return f"{CHECKPOINT_PREFIX}/{split}/chunk_{chunk_idx:05d}.parquet"


def _existing_chunks(api, hub_dataset_id, split):
    """Which chunk indices are already checkpointed on the Hub for this
    split — repo-not-found (first-ever run) and any other lookup error
    alike just mean "nothing done yet", not a reason to fail."""
    try:
        files = api.list_repo_files(hub_dataset_id, repo_type="dataset")
    except Exception:
        return set()
    prefix = f"{CHECKPOINT_PREFIX}/{split}/chunk_"
    done = set()
    for f in files:
        if f.startswith(prefix) and f.endswith(".parquet"):
            try:
                done.add(int(f[len(prefix) : -len(".parquet")]))
            except ValueError:
                pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="./quran_dataset")
    parser.add_argument("--hub-dataset-id", required=True)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Skip pushing to the Hub — for local pipeline testing.")
    # The final push_to_hub() below has no resumability of its own — a
    # container restart mid-upload (the same restart class documented
    # throughout this file) can leave a corrupted shard committed to the
    # repo even though every checkpoint chunk that fed it is fine. This
    # flag re-does only that consolidation (re-download every checkpoint
    # chunk, re-push a fresh combined dataset) without touching
    # --dataset-dir or redoing any audio decoding at all.
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help="Skip per-chunk processing; just re-download already-checkpointed chunks and re-push the final dataset.",
    )
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.hub_dataset_id, repo_type="dataset", private=True, exist_ok=True)

    if not args.consolidate_only:
        _process_chunks(api, args)

    _consolidate(api, args)


def _process_chunks(api, args):
    def load_split(split_dir):
        with open(os.path.join(split_dir, "metadata.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return Dataset.from_dict(
            {
                "audio_path": [os.path.join(split_dir, r["file_name"]) for r in rows],
                "text": [r["text"] for r in rows],
            }
        )

    dataset = DatasetDict(
        {
            "train": load_split(os.path.join(args.dataset_dir, "train")),
            "validation": load_split(os.path.join(args.dataset_dir, "validation")),
        }
    )

    # A single unreadable/corrupt audio file used to crash this whole
    # step outright — found the hard way when a live run kept dying at
    # the exact same example index regardless of how much RAM was
    # available, and then AGAIN even after wrapping the decode in
    # try/except with no traceback at all — proving the crash is a
    # native segfault in the C audio backend, not a catchable Python
    # exception. _get_pool() isolates the decode in its own process so
    # a crash there can be caught (as BrokenProcessPool) instead of
    # taking this whole job down.
    def prepare(batch, idx):
        global _POOL
        placeholder_features = [[0.0] * 3000 for _ in range(80)]  # Whisper's feature extractor always outputs this fixed (80, 3000) shape
        try:
            pool = _get_pool()
            future = pool.submit(_decode_and_extract, batch["audio_path"], batch["text"])
            input_features, labels = future.result(timeout=60)
            batch["input_features"] = input_features
            batch["labels"] = labels
            batch["_failed"] = False
        except cf.process.BrokenProcessPool as exc:
            print(f"  SKIPPING example {idx} ({batch.get('audio_path')}): worker crashed (likely a native segfault decoding this file) — {exc}", flush=True)
            _POOL = None  # force a fresh pool on the next call
            batch["input_features"] = placeholder_features
            batch["labels"] = [0]
            batch["_failed"] = True
        except cf.TimeoutError:
            print(f"  SKIPPING example {idx} ({batch.get('audio_path')}): worker timed out (likely hung decoding this file)", flush=True)
            _get_pool().shutdown(wait=False, cancel_futures=True)
            _POOL = None
            batch["input_features"] = placeholder_features
            batch["labels"] = [0]
            batch["_failed"] = True
        except Exception as exc:
            print(f"  SKIPPING example {idx} ({batch.get('audio_path')}): {type(exc).__name__}: {exc}", flush=True)
            batch["input_features"] = placeholder_features
            batch["labels"] = [0]
            batch["_failed"] = True
        return batch

    total_dropped = {"train": 0, "validation": 0}
    for split_name in ("train", "validation"):
        split_dataset = dataset[split_name]
        n = len(split_dataset)
        n_chunks = math.ceil(n / args.chunk_size)
        done = _existing_chunks(api, args.hub_dataset_id, split_name)
        if done:
            print(f"{split_name}: resuming — {len(done)}/{n_chunks} chunk(s) already checkpointed on the Hub.", flush=True)

        for chunk_idx in range(n_chunks):
            if chunk_idx in done:
                continue
            start = chunk_idx * args.chunk_size
            end = min(start + args.chunk_size, n)
            chunk = split_dataset.select(range(start, end))

            def prepare_chunk(batch, idx, _start=start):
                return prepare(batch, _start + idx)

            chunk = chunk.map(prepare_chunk, with_indices=True, remove_columns=chunk.column_names, writer_batch_size=100)
            before = len(chunk)
            chunk = chunk.filter(lambda b: not b["_failed"] and len(b["labels"]) <= MAX_LABEL_LEN)
            total_dropped[split_name] += before - len(chunk)
            chunk = chunk.remove_columns("_failed")

            local_path = f"/tmp/{split_name}_chunk_{chunk_idx:05d}.parquet"
            chunk.to_parquet(local_path)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=_checkpoint_path(split_name, chunk_idx),
                repo_id=args.hub_dataset_id,
                repo_type="dataset",
            )
            os.remove(local_path)
            print(f"{split_name}: checkpointed chunk {chunk_idx + 1}/{n_chunks} (examples {start}-{end - 1}) -> Hub", flush=True)

    if total_dropped["train"] or total_dropped["validation"]:
        print(
            f"Filtered out {total_dropped['train']} train / {total_dropped['validation']} validation example(s) "
            f"(failed to process, or exceeding {MAX_LABEL_LEN} label tokens).",
            flush=True,
        )


def _consolidate(api, args):
    # This push has no resumability of its own — see the
    # --consolidate-only argparse help above for why that matters. Each
    # chunk is re-downloaded fresh (not trusted from a prior partial
    # local state) and loaded individually rather than handing all ~57
    # files to one load_dataset(data_files=[...]) call — that combined
    # multi-file scan failed with the exact same "Couldn't deserialize
    # thrift" error even though every chunk had just validated fine on
    # its own, pointing at memory pressure from scanning ~26GB across
    # many open files at once rather than any single file being corrupt.
    # Loading one file at a time and concatenating the results avoids
    # that path entirely, and still fails loud naming the specific chunk
    # if one truly is bad.
    print("Consolidating checkpointed chunks into the final dataset...", flush=True)
    final = {}
    for split_name in ("train", "validation"):
        files = api.list_repo_files(args.hub_dataset_id, repo_type="dataset")
        prefix = f"{CHECKPOINT_PREFIX}/{split_name}/chunk_"
        chunk_files = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
        pieces = []
        for filename in chunk_files:
            local_path = hf_hub_download(repo_id=args.hub_dataset_id, repo_type="dataset", filename=filename)
            try:
                pieces.append(load_dataset("parquet", data_files=local_path, split="train"))
            except Exception as exc:
                raise RuntimeError(
                    f"Checkpoint chunk {filename} is corrupt ({type(exc).__name__}: {exc}) — it needs to be "
                    "regenerated from source audio, not just re-consolidated. Delete it from the Hub repo's "
                    f"{CHECKPOINT_PREFIX}/{split_name}/ folder and re-run preprocess.py without --consolidate-only."
                ) from exc
        final[split_name] = concatenate_datasets(pieces)
    final_dd = DatasetDict(final)

    print(f"Preprocessing done: {len(final_dd['train'])} train / {len(final_dd['validation'])} validation example(s) ready.", flush=True)
    if args.dry_run:
        print("Dry run — skipping push to Hub.")
        return
    final_dd.push_to_hub(args.hub_dataset_id, private=True)
    print(f"Done. Processed dataset pushed to https://huggingface.co/datasets/{args.hub_dataset_id}")


if __name__ == "__main__":
    main()
