"""
Evaluates one or more Whisper checkpoints against real facilitator
recordings, not the synthetic reciter-audio validation split eval_baseline.py
uses. Calls recitation_checker.check_recitation() directly — the exact
production code path (transcription + error-diffing against fetched verse
text) — rather than reimplementing WER separately, so a "looks good here"
result actually means the deployed app would behave the same way.

The manifest below was built by transcribing each recording with the base
model and matching the output to Quran text (see conversation/session notes)
— every entry's (surah, ayah_start, ayah_end) was verified this way, not
guessed from filenames. Every recording is assumed correctly recited (these
are the same clips used to empirically validate/debug the checker itself),
so on a healthy model the expected error count on each is 0 or close to it;
a spike is a regression signal, not evidence the facilitator misspoke.

Only covers what still exists on disk: Al-Fatiha, Al-Baqarah 2-5, Al-Isra
1-2, Al-Ikhlas. Recordings referenced elsewhere in this project's test logs
for Al-Kahf/Yaseen/Al-Ahzab (the surahs an earlier narrow fine-tune was
scoped to) are gone from disk — only their output logs survive — so those
passages aren't covered here unless new recordings are sourced.

Usage:
    python eval_real_recordings.py --model-id tarteel-ai/whisper-base-ar-quran
    python eval_real_recordings.py --model-id tarteel-ai/whisper-base-ar-quran --model-id <candidate> --audio-dir ../.
    HF_TOKEN=... python eval_real_recordings.py --model-id <private-candidate-repo>
"""

import argparse
import os
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    # Windows' default console codepage (cp1252) can't encode Arabic —
    # reconfigure before any Arabic text gets printed, not just wrapped
    # per-call, since check_recitation()'s own error notes print too.
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import recitation_checker  # noqa: E402

# (file, surah_number, ayah_start, ayah_end) — ayah range is inclusive,
# matching check_recitation's own convention.
MANIFEST = [
    ("fatiha_1.mp3", 1, 1, 1),
    ("fatiha_2.mp3", 1, 2, 2),
    ("fatiha_3.mp3", 1, 3, 3),
    ("fatiha_4.mp3", 1, 4, 4),
    ("fatiha_5.mp3", 1, 5, 5),
    ("fatiha_6.mp3", 1, 6, 6),
    ("fatiha_7.mp3", 1, 7, 7),
    ("fatiha_full_clean.mp3", 1, 1, 7),
    ("fatiha_32k.webm", 1, 1, 7),
    ("fatiha_96k.webm", 1, 1, 7),
    ("baq_9.mp3", 2, 2, 2),
    ("baq_10.mp3", 2, 3, 3),
    ("baq_11.mp3", 2, 4, 4),
    ("baq_12.mp3", 2, 5, 5),
    ("baqarah_2to5_clean.mp3", 2, 2, 5),
    ("isra_1.mp3", 17, 1, 1),
    ("isra_2.mp3", 17, 2, 2),
    ("isra_clean_retest.mp3", 17, 1, 2),
    ("test_ayah.mp3", 112, 1, 1),
    ("test_ayah_2.mp3", 112, 2, 2),
]


def evaluate_model(model_id, audio_dir):
    recitation_checker.MODEL_NAME = model_id
    recitation_checker._model = None
    recitation_checker._processor = None

    results = {}
    for fname, surah, ayah_start, ayah_end in MANIFEST:
        path = audio_dir / fname
        if not path.is_file():
            print(f"  skip {fname}: not found in {audio_dir}")
            continue
        try:
            transcription, duration, errors = recitation_checker.check_recitation(
                str(path), surah, ayah_start, ayah_end
            )
        except Exception as exc:
            print(f"  FAILED {fname}: {type(exc).__name__}: {exc}")
            results[fname] = {"error_count": None, "errors": [], "duration": None}
            continue
        results[fname] = {"error_count": len(errors), "errors": errors, "duration": duration}
        print(f"  {fname}: {len(errors)} error(s) flagged ({duration:.1f}s audio)")
        for e in errors:
            print(f"      ayah {e['ayah_number']} word {e['word_index']} '{e['word_text']}': "
                  f"{e['label']} — {e['note']}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", action="append", required=True, dest="model_ids",
                         help="Repeatable. Any Whisper checkpoint id — base or a fine-tuned candidate.")
    parser.add_argument("--audio-dir", default=None,
                         help="Directory containing the recordings (default: recitation-checker/, next to this script).")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir) if args.audio_dir else Path(__file__).resolve().parent.parent

    all_results = {}
    for model_id in args.model_ids:
        print(f"\n=== Evaluating {model_id} ===")
        all_results[model_id] = evaluate_model(model_id, audio_dir)

    print("\n=== Summary (errors flagged per recording, lower is better on these known-correct clips) ===")
    header = f"{'file':<28}" + "".join(f"{m:<40}" for m in args.model_ids)
    print(header)
    for fname, *_ in MANIFEST:
        row = f"{fname:<28}"
        for model_id in args.model_ids:
            r = all_results[model_id].get(fname)
            cell = "—" if r is None else ("FAILED" if r["error_count"] is None else str(r["error_count"]))
            row += f"{cell:<40}"
        print(row)

    if len(args.model_ids) > 1:
        base, *candidates = args.model_ids
        print(f"\n=== Regressions vs {base} ===")
        any_regression = False
        for fname, *_ in MANIFEST:
            base_r = all_results[base].get(fname)
            if base_r is None or base_r["error_count"] is None:
                continue
            for cand in candidates:
                cand_r = all_results[cand].get(fname)
                if cand_r is None or cand_r["error_count"] is None:
                    continue
                if cand_r["error_count"] > base_r["error_count"]:
                    any_regression = True
                    print(f"  {fname}: {base} flagged {base_r['error_count']}, "
                          f"{cand} flagged {cand_r['error_count']} (+{cand_r['error_count'] - base_r['error_count']})")
        if not any_regression:
            print("  none found")


if __name__ == "__main__":
    main()
