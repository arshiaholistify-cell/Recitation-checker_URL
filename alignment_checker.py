"""
Word accuracy by aligning the audio against the expected text, instead
of transcribing freely and diffing the result.

The checker always knows what the student was asked to recite. The
Whisper path in recitation_checker.py throws that away: it asks "what
did they say?" — an open question over every Arabic word and every
vowelling — and then compares the answer to the expected ayahs. Every
mis-transcription of a CORRECTLY recited word becomes a false error, and
the model can produce words nobody said. Both were observed live: a
basmala transcribed "في السم اللات الرحمن رحيم", and "اللاعب سبيلا"
invented at the end of a Ya-Sin recitation.

This module asks the narrower question the app actually has — "does this
audio match THESE phonemes, and where does it not?" — using the same
Muaalem model tajweed_checker.py already runs against the same reference.
A word can only be flagged when its expected phonemes are absent from the
audio. There is no free vocabulary to hallucinate into, letter-name
versus glyph spelling (ياسين/يسٓ) stops mattering because both phonetize
to the same sounds, and اللات cannot appear because it was never in the
reference.

WORD-LEVEL ATTRIBUTION

tajweed_checker.py attributes a phoneme diff to an ayah, and its
docstrings call finer attribution an "unsolved phoneme-to-word
correlation". It is solvable, just not the obvious way:

  - Phonetizing each word ALONE and summing the lengths drifts, because
    Hafs rules assimilate across word boundaries (idgham, iqlab and
    friends). Measured on 10 ayahs it was wrong on 9, by -4 to +5
    phonemes, and the error accumulates along the ayah.

  - Phonetizing PREFIXES — the first k words, joined — keeps every word
    in its real context. Each boundary is then the only approximation,
    and it does not accumulate: measured across Al-Fatiha, Ya-Sin 1-10,
    Al-Baqarah 1-5 and Al-Ikhlas, the last prefix equals the whole
    passage exactly and offsets are monotonic in every case.

So word k occupies phonemes [len(phonemes(words[:k])), len(phonemes(words[:k+1]))).
Cost is one phonetizer call per word, ~8ms each — 0.7s for ten ayahs,
against a check that already takes minutes.

WHAT THIS DELIBERATELY DOES NOT DO

Judge tajweed. A short madd or a missed ghunnah is a handful of phoneme
differences inside an otherwise correct word, and tajweed_checker.py
already reports those with the rule names attached. A word is only
flagged here when enough of it is missing or different to mean the wrong
word was said — see _MISMATCH_THRESHOLD.

It also cannot tell you the student recited a completely different
passage: every word would simply be wrong. Whisper remains better at
that, which is why the transcription path stays.
"""

import difflib

from librosa import load

from tajweed_checker import _MOSHAF, _load_muaalem
from quran_transcript import Aya, quran_phonetizer

# A word is reported only when at least this fraction of its expected
# phonemes fall inside a mismatched region. Below it, the difference is
# a pronunciation detail within a recognisable word — tajweed's job, not
# word accuracy's. A third is deliberately forgiving: the cost of a
# false "wrong word" on a child's recitation is higher than the cost of
# missing a marginal one, and the facilitator still reviews every result.
_MISMATCH_THRESHOLD = 1 / 3


def _phonemes(text):
    return quran_phonetizer(text, _MOSHAF, remove_spaces=True).phonemes


def build_word_spans(surah_number, ayah_start, ayah_end):
    """Every expected word with the phoneme range it occupies.

    Returns (full_text, [{ayah_number, word_index, word_text, ph_start,
    ph_end}]), where word_index counts within its own ayah — matching
    what is_recitation_errors stores and what the app's word-pinning UI
    expects.
    """
    ayah_words = []
    for ayah in range(ayah_start, ayah_end + 1):
        for idx, word in enumerate(Aya(surah_number, ayah).get().uthmani.split()):
            ayah_words.append((ayah, idx, word))

    full_text = " ".join(w for (_, _, w) in ayah_words)

    spans, previous_end = [], 0
    for k, (ayah, idx, word) in enumerate(ayah_words, start=1):
        # The prefix, in context — not the word alone. See module docstring.
        end = len(_phonemes(" ".join(w for (_, _, w) in ayah_words[:k])))
        spans.append({
            "ayah_number": ayah,
            "word_index": idx,
            "word_text": word,
            "ph_start": previous_end,
            "ph_end": max(end, previous_end),
        })
        previous_end = max(end, previous_end)
    return full_text, spans


def _mismatch_by_word(spans, opcodes, ref_len):
    """How many of each word's phonemes sit inside a mismatched region,
    and whether the region deleted them outright rather than replacing
    them. Returns {word position: (mismatched_count, all_deleted)}."""
    tally = {}
    for pos, span in enumerate(spans):
        start, end = span["ph_start"], min(span["ph_end"], ref_len)
        if end <= start:
            continue
        mismatched = 0
        deleted = 0
        for tag, i1, i2, _j1, _j2 in opcodes:
            if tag == "equal":
                continue
            overlap = max(0, min(end, i2) - max(start, i1))
            if not overlap:
                continue
            mismatched += overlap
            if tag == "delete":
                deleted += overlap
        if mismatched:
            tally[pos] = (mismatched, deleted == mismatched)
    return tally


def check_recitation_aligned(audio_path, surah_number, ayah_start, ayah_end,
                             allowed_error_types=None, device="cpu"):
    """Word-accuracy errors, shaped exactly like check_recitation()'s so
    the two can be swapped or compared without touching service.py.

    Returns (errors, reference_phonemes, predicted_phonemes) — the two
    phoneme strings come back so a caller can store them for debugging;
    they are the alignment equivalent of a transcript.
    """
    full_text, spans = build_word_spans(surah_number, ayah_start, ayah_end)
    phonetizer_out = quran_phonetizer(full_text, _MOSHAF, remove_spaces=True)
    ref_phonemes = phonetizer_out.phonemes

    muaalem = _load_muaalem(device=device)
    wave, _ = load(audio_path, sr=16000, mono=True)
    result = muaalem([wave], [phonetizer_out], sampling_rate=16000)[0]
    pred_phonemes = result.phonemes.text

    opcodes = difflib.SequenceMatcher(None, ref_phonemes, pred_phonemes).get_opcodes()
    tally = _mismatch_by_word(spans, opcodes, len(ref_phonemes))

    errors = []

    def emit(entry):
        if allowed_error_types is not None:
            if entry["label"] not in allowed_error_types:
                return
            entry["error_type_id"] = allowed_error_types[entry["label"]]
        errors.append(entry)

    for pos, span in enumerate(spans):
        mismatched, all_deleted = tally.get(pos, (0, False))
        if not mismatched:
            continue
        length = max(1, span["ph_end"] - span["ph_start"])
        if mismatched / length < _MISMATCH_THRESHOLD:
            continue  # a pronunciation detail inside a recognisable word
        if all_deleted:
            label = "Skipped word"
            note = "This word's sounds were not found in the recording."
        else:
            label = "Wrong word"
            note = (
                "What was recited here did not match this word "
                f"({mismatched} of {length} sounds differed)."
            )
        emit({
            "ayah_number": span["ayah_number"],
            "word_index": span["word_index"],
            "word_text": span["word_text"],
            "category": "recitation",
            "label": label,
            "note": note,
            "source": "auto",
        })

    return errors, ref_phonemes, pred_phonemes
