"""Compares Whisper checkpoints on the real facilitator recordings, on the
two things that actually matter, separately.

eval_real_recordings.py reports only how many errors check_recitation()
flagged. That number is the product of two independent things — how well
the model heard the audio, and how well the checker's heuristics forgive
the ways it mishears — so a checkpoint can look better purely because its
mistakes happen to land where a heuristic absorbs them. Comparing two
checkpoints needs them pulled apart:

  WER    what the model heard, against the expected text. This is the
         model's quality, and it is what fine-tuning is supposed to move.
         Broken out into substitutions, deletions and insertions, because
         they mean different things: insertions are the hallucination
         failure that motivated the fine-tune, deletions are usually the
         model giving up on audio it cannot parse.

  flags  what the deployed app would show the facilitator. Every
         recording here is correctly recited, so every flag is a false
         positive and the only healthy number is zero.

A checkpoint that lowers WER but raises flags has moved its errors into
words the heuristics do not cover, and is not necessarily an improvement.
Both columns are reported for every model, and both are per-recording, so
a single bad file cannot hide inside an average.

Usage:
    python compare_models.py --model-id tarteel-ai/whisper-base-ar-quran \
                             --model-id Holistify/whisper-base-ar-quran-finetuned-v2
    HF_TOKEN=... python compare_models.py --model-id <private-candidate>
"""

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import recitation_checker  # noqa: E402
from eval_real_recordings import MANIFEST  # noqa: E402


def _edits(reference, hypothesis):
    """Levenshtein over words, kept as (substitutions, deletions,
    insertions) rather than collapsed into one distance — the three say
    different things about a checkpoint. Deletion means a reference word
    the model never produced; insertion means a word it produced from
    nothing, which is the hallucination this fine-tune targets."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    # (cost, subs, dels, ins) per cell.
    grid = [[(0, 0, 0, 0)] * cols for _ in range(rows)]
    for i in range(1, rows):
        grid[i][0] = (i, 0, i, 0)
    for j in range(1, cols):
        grid[0][j] = (j, 0, 0, j)
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                grid[i][j] = grid[i - 1][j - 1]
                continue
            sub = grid[i - 1][j - 1]
            dele = grid[i - 1][j]
            ins = grid[i][j - 1]
            best = min((sub[0], 0), (dele[0], 1), (ins[0], 2))
            if best[1] == 0:
                grid[i][j] = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
            elif best[1] == 1:
                grid[i][j] = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
            else:
                grid[i][j] = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
    _cost, subs, dels, ins = grid[-1][-1]
    return subs, dels, ins


_REFERENCE_CACHE = {}


def expected_words(surah, first, last):
    """The reference the model is scored against — the same words
    check_recitation() diffs, normalized the same way, so WER and the
    flag count are measured against one text and not two. Cached because
    every model re-scores every recording and the text comes over the
    network."""
    key = (surah, first, last)
    if key not in _REFERENCE_CACHE:
        flat, _tajweed = recitation_checker.fetch_expected_words(surah, first, last)
        words = []
        for _ayah, _idx, text in flat:
            words.extend(recitation_checker.normalize_arabic(text).split())
        _REFERENCE_CACHE[key] = words
    return _REFERENCE_CACHE[key]


def heard_words(transcription, reference):
    """What the model heard, with a recited preamble dropped the way
    check_recitation() drops it. Saying the isti'adhah or a basmala before
    an ayah that does not contain one is correct etiquette, not a
    transcription error, and counting it would charge the model four
    insertions for the reciter's manners."""
    words = recitation_checker.normalize_arabic(transcription).split()
    skip = recitation_checker._preamble_word_count(words, reference)
    return words[skip:]


def evaluate_model(model_id, audio_dir):
    recitation_checker.MODEL_NAME = model_id
    recitation_checker._model = None
    recitation_checker._processor = None

    rows = {}
    for fname, surah, first, last in MANIFEST:
        path = audio_dir / fname
        if not path.is_file():
            continue
        try:
            transcription, duration, errors = recitation_checker.check_recitation(
                str(path), surah, first, last)
        except Exception as exc:
            rows[fname] = {"failed": f"{type(exc).__name__}: {exc}"}
            print("  %-26s FAILED %s" % (fname, rows[fname]["failed"]), flush=True)
            continue

        reference = expected_words(surah, first, last)
        heard = heard_words(transcription, reference)
        subs, dels, ins = _edits(reference, heard)
        wer = (subs + dels + ins) / max(1, len(reference))
        rows[fname] = {
            "ref": "%d:%d-%d" % (surah, first, last),
            "words": len(reference), "heard": len(heard),
            "subs": subs, "dels": dels, "ins": ins, "wer": wer,
            "flags": len(errors),
            "flagged_words": [e["word_text"] for e in errors],
            "transcription": transcription,
            "duration": duration,
        }
        print("  %-26s wer %5.1f%%  (s%3d d%3d i%3d of %3d)  flags %2d"
              % (fname, 100 * wer, subs, dels, ins, len(reference), len(errors)),
              flush=True)
    return rows


def corpus_totals(rows):
    ok = [r for r in rows.values() if "failed" not in r]
    words = sum(r["words"] for r in ok)
    subs = sum(r["subs"] for r in ok)
    dels = sum(r["dels"] for r in ok)
    ins = sum(r["ins"] for r in ok)
    return {
        "files": len(ok), "words": words, "subs": subs, "dels": dels, "ins": ins,
        "wer": (subs + dels + ins) / max(1, words),
        "flags": sum(r["flags"] for r in ok),
        "clean_files": sum(1 for r in ok if r["flags"] == 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", action="append", required=True, dest="model_ids")
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--out", default=None, help="Write the full per-file detail as JSON.")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir) if args.audio_dir else Path(__file__).resolve().parent.parent
    everything = {}
    for model_id in args.model_ids:
        print("\n=== %s ===" % model_id, flush=True)
        everything[model_id] = evaluate_model(model_id, audio_dir)

    print("\n=== Corpus totals ===")
    print("%-52s %6s %6s %6s %6s %7s %6s" %
          ("model", "WER", "subs", "dels", "ins", "flags", "clean"))
    totals = {}
    for model_id in args.model_ids:
        t = totals[model_id] = corpus_totals(everything[model_id])
        print("%-52s %5.1f%% %6d %6d %6d %7d %4d/%d" %
              (model_id[-52:], 100 * t["wer"], t["subs"], t["dels"], t["ins"],
               t["flags"], t["clean_files"], t["files"]))

    if len(args.model_ids) > 1:
        base, *candidates = args.model_ids
        for cand in candidates:
            b, c = totals[base], totals[cand]
            print("\n=== %s vs %s ===" % (cand, base))
            print("  WER   %5.1f%% -> %5.1f%%  (%+.1f points)"
                  % (100 * b["wer"], 100 * c["wer"], 100 * (c["wer"] - b["wer"])))
            for key in ("subs", "dels", "ins", "flags"):
                print("  %-5s %5d   -> %5d    (%+d)" % (key, b[key], c[key], c[key] - b[key]))
            worse = [f for f in everything[base]
                     if "failed" not in everything[base][f]
                     and "failed" not in everything[cand].get(f, {"failed": 1})
                     and everything[cand][f]["flags"] > everything[base][f]["flags"]]
            print("  recordings where %s flags more: %s" % (cand, ", ".join(worse) or "none"))

    if args.out:
        Path(args.out).write_text(json.dumps(everything, ensure_ascii=False, indent=2))
        print("\nper-file detail written to %s" % args.out)


if __name__ == "__main__":
    main()
