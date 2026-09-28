"""The gate alignment_checker.py has to pass before it goes live.

Two halves, because either alone is misleading. The clean corpus shows it
does not cry wolf on correct recitation — the failure mode that made the
Whisper path unusable, where a correctly recited Al-Fatiha came back with
24 errors. The negative controls show it still barks, and barks in the
right place: a checker that returns zero on everything would pass the
first half perfectly.

Every case is built from the recordings committed alongside this file, so
no synthetic speech is involved. The negative cases splice real audio —
recite ayahs 1, 2 and 4 and ask for 1-4 — which is exactly what a student
skipping a line sounds like.

    python eval_alignment_checker.py            # both halves
    python eval_alignment_checker.py clean      # correct recitation only
    python eval_alignment_checker.py negative   # controls only
"""

import os
import sys
import tempfile

import numpy as np
import soundfile as sf
from librosa import load

from alignment_checker import build_word_spans, check_recitation_aligned

HERE = os.path.dirname(os.path.abspath(__file__))

# Correct recitations. Anything other than zero errors is a false positive.
CLEAN = [
    ("fatiha_1.mp3", 1, 1, 1), ("fatiha_2.mp3", 1, 2, 2), ("fatiha_3.mp3", 1, 3, 3),
    ("fatiha_4.mp3", 1, 4, 4), ("fatiha_5.mp3", 1, 5, 5), ("fatiha_6.mp3", 1, 6, 6),
    ("fatiha_7.mp3", 1, 7, 7), ("fatiha_full_clean.mp3", 1, 1, 7),
    # Same recitation at two Opus bitrates: the app records webm in browser.
    ("fatiha_32k.webm", 1, 1, 7), ("fatiha_96k.webm", 1, 1, 7),
    ("baq_9.mp3", 2, 2, 2), ("baq_10.mp3", 2, 3, 3), ("baq_11.mp3", 2, 4, 4),
    ("baq_12.mp3", 2, 5, 5), ("baqarah_2to5_clean.mp3", 2, 2, 5),
    ("isra_1.mp3", 17, 1, 1), ("isra_2.mp3", 17, 2, 2),
    ("isra_clean_retest.mp3", 17, 1, 2),
]


def _splice(names, out_path):
    """Join recordings with the pause a student leaves between ayahs."""
    waves = [load(os.path.join(HERE, n), sr=16000, mono=True)[0] for n in names]
    gap = np.zeros(int(0.35 * 16000), dtype=waves[0].dtype)
    joined = np.concatenate([part for w in waves for part in (w, gap)])
    sf.write(out_path, joined, 16000)
    return out_path


def _negative_cases(tmp):
    """(label, audio, surah, start, end, expected flagged positions).

    Expected positions are "ayah:word_index". None means the count is
    reported but not asserted — a wrong-passage check has no single
    defensible answer for which words should flag.
    """
    return [
        ("ayah 3 skipped from 1:1-4",
         _splice(["fatiha_1.mp3", "fatiha_2.mp3", "fatiha_4.mp3"],
                 os.path.join(tmp, "skip_fatiha_3.wav")),
         1, 1, 4, ["3:0", "3:1"]),
        ("ayah 2:3 skipped from 2:2-4",
         _splice(["baq_9.mp3", "baq_11.mp3"], os.path.join(tmp, "skip_baq_3.wav")),
         2, 2, 4, ["3:%d" % i for i in range(8)]),
        ("audio of 1:3 checked against 1:5",
         os.path.join(HERE, "fatiha_3.mp3"), 1, 5, 5, None),
        # Reciting past the assigned range is not a word accuracy error:
        # every expected word was said. Extra material is the transcription
        # path's job, and this asserts the aligner stays quiet about it.
        ("extra ayah 1:3 beyond reference 1:1-2",
         _splice(["fatiha_1.mp3", "fatiha_2.mp3", "fatiha_3.mp3"],
                 os.path.join(tmp, "extra_fatiha_3.wav")),
         1, 1, 2, []),
    ]


def run_clean():
    failures = 0
    print("Correct recitation — every error is a false positive")
    for name, surah, first, last in CLEAN:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            print("  %-26s SKIPPED (not present)" % name)
            continue
        errors, ref, _pred = check_recitation_aligned(path, surah, first, last)
        mark = "ok" if not errors else "FALSE POSITIVE"
        print("  %-26s %d:%d-%-3d %2d errors  %-14s %s"
              % (name, surah, first, last, len(errors), mark,
                 ", ".join(e["word_text"] for e in errors)))
        failures += bool(errors)
    return failures


def run_negative():
    failures = 0
    print("Negative controls — the error has to be caught, and localised")
    with tempfile.TemporaryDirectory() as tmp:
        for label, path, surah, first, last, expected in _negative_cases(tmp):
            _text, spans = build_word_spans(surah, first, last)
            errors, _ref, _pred = check_recitation_aligned(path, surah, first, last)
            flagged = ["%d:%d" % (e["ayah_number"], e["word_index"]) for e in errors]
            if expected is None:
                mark = "reported"
            elif flagged == expected:
                mark = "ok"
            else:
                mark = "WRONG WORDS"
                failures += 1
            print("  %-40s %2d/%2d words  %-12s %s"
                  % (label, len(errors), len(spans), mark, " ".join(flagged)))
    return failures


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    bad = 0
    if which in ("both", "clean"):
        bad += run_clean()
    if which in ("both", "negative"):
        if which == "both":
            print()
        bad += run_negative()
    print("\n%s" % ("all cases as expected" if not bad else "%d case(s) wrong" % bad))
    sys.exit(1 if bad else 0)
