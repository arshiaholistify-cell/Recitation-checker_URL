"""
Fine-tunes tarteel-ai/whisper-base-ar-quran further on a dataset
already prepared by preprocess.py (audio decoded, feature-extracted,
and tokenized — see that file), aiming to reduce the hallucination on
longer/less-common passages found empirically while testing the
shipped auto-checker (see recitation-checker/recitation_checker.py).
Continues from that checkpoint rather than vanilla openai/whisper-base,
since it already has a strong Quran-specific prior.

Standard HF Whisper fine-tuning recipe (per huggingface.co/blog/fine-tune-whisper),
adapted to continue from the tarteel checkpoint and this project's dataset.

This file used to also do the audio preprocessing itself, but that step
is CPU-only (never touches torch.cuda) and was taking ~4s/example
single-threaded across a 31,180-example dataset — roughly a day of
GPU-billed time spent on work the GPU wasn't even doing. preprocess.py
now does that part on cheap/free CPU Space hardware and pushes the
result to the Hub as a dataset repo; this file just loads it, so
everything here actually needs the GPU tier it runs on.

Run inside the training Space (GPU hardware tier) — needs an HF_TOKEN
secret with write access to push the resulting model to the Hub.

Usage:
    python train.py --hub-dataset-id <your-username>/quran-whisper-finetune-processed --hub-model-id <your-username>/whisper-base-ar-quran-finetuned-v1
"""

import argparse
import os

import evaluate
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

BASE_MODEL = "tarteel-ai/whisper-base-ar-quran"


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-dataset-id", required=True, help="Dataset repo produced by preprocess.py.")
    parser.add_argument("--hub-model-id", required=True)
    # Dropped from 4 to 2: the dataset now covers all 114 surahs instead
    # of 14, so each epoch already sees ~5x more examples than before —
    # 2 epochs here is roughly the same total training exposure as the
    # original 4-epoch run, at a fraction of the wall-clock/GPU cost.
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--dry-run", action="store_true", help="Skip pushing to the Hub — for local pipeline testing.")
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(BASE_MODEL)

    # Already feature-extracted and tokenized by preprocess.py — no
    # per-example CPU work left to do here, so this loads fast
    # regardless of dataset size and everything from here on actually
    # needs the GPU tier this script runs on.
    dataset = load_dataset(args.hub_dataset_id)

    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.generation_config.language = None
    model.generation_config.task = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

    # Training on the GPU tier hit the same kind of mid-run container
    # restart that plagued preprocess.py on CPU (see that file's
    # comments for how thoroughly that was diagnosed) — confirmed here
    # by a literal "an error occurred when try to find container ...:
    # not found" line appearing in this Space's own logs, meaning the
    # platform is killing and replacing the container itself, not a
    # crash inside this script. Since we don't know how many minutes
    # this container gets before the next restart, save_steps is kept
    # small (not the usual few-hundred-step interval) so a checkpoint
    # exists well before any restart is likely to land. save_strategy/
    # eval_strategy="steps" at this interval plus hub_strategy=
    # "checkpoint" (which pushes the latest checkpoint — model,
    # optimizer, scheduler, RNG state — to a `last-checkpoint`
    # subfolder of the model repo on every save) means a restart now
    # costs at most one save interval's worth of steps, and that
    # checkpoint survives on the Hub across restarts since local disk
    # doesn't.
    SAVE_STEPS = 50
    # /data is the Space's persistent storage bucket (survives container
    # restarts, unlike the rest of the filesystem) — writing local
    # checkpoints there too means a restart can resume from disk
    # directly, without even needing the Hub download below.
    OUTPUT_DIR = "/data/whisper-finetune-out" if os.path.isdir("/data") else "./whisper-finetune-out"
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=not args.dry_run,
        hub_model_id=args.hub_model_id,
        hub_private_repo=True,
        hub_strategy="checkpoint",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # A checkpoint already sitting on persistent storage (output_dir, if
    # it's on the /data bucket) survives a restart on its own — no Hub
    # round-trip needed. Only fall back to downloading the checkpoint
    # the Hub push kept in `last-checkpoint` when there's nothing local
    # (e.g. persistent storage isn't attached, or this is a fresh disk).
    resume_from = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    if resume_from:
        print(f"Resuming from a local checkpoint on persistent storage: {resume_from}", flush=True)
    elif not args.dry_run:
        try:
            HfApi().list_repo_files(args.hub_model_id, repo_type="model")
            snapshot_download(
                repo_id=args.hub_model_id,
                repo_type="model",
                allow_patterns=["last-checkpoint/*"],
                local_dir=training_args.output_dir,
            )
            local_checkpoint = os.path.join(training_args.output_dir, "last-checkpoint")
            if os.path.isdir(local_checkpoint) and os.listdir(local_checkpoint):
                resume_from = local_checkpoint
                print(f"Resuming from checkpoint pushed to the Hub: {resume_from}", flush=True)
        except Exception as exc:
            print(f"No resumable checkpoint found on the Hub ({type(exc).__name__}: {exc}) — starting fresh.", flush=True)

    print(f"Training on {len(dataset['train'])} clips, validating on {len(dataset['validation'])}...")
    trainer.train(resume_from_checkpoint=resume_from)
    if args.dry_run:
        print("Dry run — skipping push to Hub.")
        return
    trainer.push_to_hub()
    processor.push_to_hub(args.hub_model_id, private=True)
    print(f"Done. Model pushed to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
