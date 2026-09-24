"""
Computes word error rate (WER) for a Whisper checkpoint against this
project's validation set, so the fine-tuned model has a real baseline
number to compare against instead of just "it feels better/worse" on
a handful of spot checks.

Works on either the original base model or a fine-tuned one — pass
whichever --model-id you want a number for. Loads the same processed
dataset train.py trains against (already feature-extracted by
preprocess.py: input_features + labels, no raw audio/text) via the
same streaming=True approach train.py uses — a plain, non-streaming
load_dataset() call here hit "Couldn't deserialize thrift ...
Deserializing page header failed" against a shard the full-download
path chokes on, even though streaming reads the same underlying
Parquet files fine.

Evaluates a subset of the validation split rather than all of it —
generation is slow enough that evaluating all ~3,100 validation
examples would take a very long time for a number that's meant to be
a quick directional comparison, not a publishable benchmark.

Usage:
    python eval_baseline.py --hub-dataset-id <your-username>/quran-whisper-finetune-processed --model-id tarteel-ai/whisper-base-ar-quran
"""

import argparse
import re

import evaluate
import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# tarteel-ai/whisper-base-ar-quran's tokenizer_config.json doesn't mark
# tokens like <|startoftranscript|>/<|notimestamps|>/<|ar|>/<|transcribe|>
# as "special" — confirmed live: skip_special_tokens=True on both the
# generated predictions AND the decoded labels still left this literal
# markup in the text. That's present in the ORIGINAL upstream
# checkpoint itself, not something this project's fine-tuning
# introduced (see recitation_checker.py's _patch_extra_special_tokens_
# list_bug for the same underlying defect hit elsewhere). Left
# unstripped, these tokens count as extra mismatched "words" in the WER
# computation, inflating the score for a reason that has nothing to do
# with actual transcription quality.
_LEAKED_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def _clean(text):
    return _LEAKED_TOKEN_RE.sub("", text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-dataset-id", required=True, help="Processed dataset repo produced by preprocess.py.")
    parser.add_argument("--model-id", required=True, help="Any Whisper checkpoint — base or fine-tuned.")
    parser.add_argument("--batch-size", type=int, default=8)
    # Small enough to finish in a few minutes, large enough to be a
    # meaningful directional signal rather than noise from 5 examples.
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--num-samples-to-print", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading processor/model for {args.model_id} ...", flush=True)
    processor = WhisperProcessor.from_pretrained(args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.eval()

    print(f"Loading validation split from {args.hub_dataset_id} (streaming) ...", flush=True)
    dataset = load_dataset(
        args.hub_dataset_id,
        data_files={"validation": "validation-*.parquet"},
        streaming=True,
    )["validation"]
    if args.max_examples:
        dataset = dataset.take(args.max_examples)
    print(f"Evaluating on up to {args.max_examples} validation example(s).", flush=True)

    wer_metric = evaluate.load("wer")
    all_predictions = []
    all_references = []

    def process_batch(batch_input_features, batch_labels):
        input_features = torch.tensor(batch_input_features)
        predicted_ids = model.generate(input_features)
        pred_strs = [_clean(s) for s in processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)]

        label_ids = [[tok for tok in labels if tok != -100] for labels in batch_labels]
        label_strs = [_clean(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

        all_predictions.extend(pred_strs)
        all_references.extend(label_strs)

    batch_input_features = []
    batch_labels = []
    seen = 0
    with torch.no_grad():
        for example in dataset:
            batch_input_features.append(example["input_features"])
            batch_labels.append(example["labels"])
            if len(batch_input_features) == args.batch_size:
                process_batch(batch_input_features, batch_labels)
                seen += len(batch_input_features)
                print(f"  ...{seen} done", flush=True)
                batch_input_features = []
                batch_labels = []
        if batch_input_features:
            process_batch(batch_input_features, batch_labels)
            seen += len(batch_input_features)
            print(f"  ...{seen} done", flush=True)

    wer = 100 * wer_metric.compute(predictions=all_predictions, references=all_references)
    print(f"\nWER: {wer:.2f}% (over {len(all_predictions)} example(s), model={args.model_id})", flush=True)

    print(f"\nSample predictions (first {args.num_samples_to_print}):", flush=True)
    for i in range(min(args.num_samples_to_print, len(all_predictions))):
        print(f"  EXPECTED: {all_references[i]}", flush=True)
        print(f"  HEARD   : {all_predictions[i]}", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
