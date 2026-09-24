"""
Acoustic Madd (elongation) checking, layered on top of the word-accuracy
checker in recitation_checker.py. Uses the Quran Muaalem model
(obadx/muaalem-model-v3_2, MIT licensed, https://github.com/obadx/quran-muaalem)
to predict phonemes from audio and compares them against a reference
built with the quran_transcript package, which tags each expected
phoneme span with which madd (elongation) rule, if any, governs it.

Scope, deliberately narrow: this only checks madd/elongation length —
NOT qalqalah, ghunnah, idgham, or iqlab. Those need comparing the
model's predicted acoustic *characteristics* (sifat) against expected
ones (see quran_muaalem.explain.expalin_sifat for the library's own
approach), which requires correlating two independently-grouped
phoneme sequences — attempted during prototyping and not reliably
solved in the time available. Madd checking uses a much simpler,
directly-verified signal instead: mappings[].tajweed_rules already
tags exactly which phonemes carry a madd rule and its required length,
with no grouping/correlation step needed.

Key finding from prototyping (validated against a real 7-ayah
continuous recitation): comparing a multi-ayah recording's audio
against one flat phoneme reference produces a false "error" at every
ayah boundary, because real reciters naturally take a breath between
ayahs — an Aared Madd (optional pause-elongation a reciter may freely
extend), not a mistake. Every diff opcode near an ayah-boundary offset
is therefore suppressed unconditionally, regardless of which specific
madd rule the phonetizer tagged there. A more "precise" version was
tried first — only tolerating a boundary diff when the phonetizer
specifically tagged that position AaredMaddRule rather than any madd
rule — but that undercounted false positives badly: the phonetizer
only tags AaredMaddRule at the very last ayah of whatever reference
text it's given, since it has no way to know a real reciter also
pauses between ayahs it wasn't told to treat as a stop. Every other
ayah's ending madd gets the stricter NormalMaddRule despite being the
same kind of natural pause acoustically, so the "smarter" check ended
up flagging nearly every ayah boundary again. KNOWN GAP, confirmed empirically, not just theoretical: unconditional
boundary suppression will also hide a genuine madd violation that
happens to sit at an ayah's last word - and that's a common case, since
necessary (lazem) madds are often word-final/ayah-final. Tested this
directly against a real recording of Yaseen 36:1 ("يس", a necessary-
madd muqatta'at) with a deliberately shortened elongation (confirmed
via raw model output: 6 elongation phonemes dropped to 5, a real,
model-detected difference) - the module reported nothing, because that
defect sits exactly 1 phoneme-character from the ayah's end boundary,
identical in distance to the harmless breathing-pause artifacts
_BOUNDARY_TOLERANCE exists to suppress. A tolerance sweep from 0-7
confirmed no value separates the two cases: at tolerance 0 the false
positives come back, at tolerance 1+ this true positive disappears
along with them - they are geometrically indistinguishable by position
alone. This is a hard ceiling on a phoneme-position-only heuristic, not
a tuning problem; solving it properly needs real per-word audio timing
data (a bigger, separate effort), which is why it hasn't been solved
here. Chose the current tolerance (see _BOUNDARY_TOLERANCE) because a
false positive is worse for an unproven auto-detector than a false
negative - so this module reliably catches genuine mid-ayah madd
violations away from any ayah boundary, and reliably misses violations
at/near an ayah's start or end.

Attribution is ayah-level, not word-level: correlating a specific
phoneme-string diff position back to an exact word index turned out to
need the same unsolved grouping-correlation problem above. This
reports "somewhere in this ayah" rather than a precise word.
"""

import difflib

from librosa.core import load
from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

_MUAALEM = None

_MOSHAF = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=2,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=2,
)

_MADD_LABEL = "Madd not elongated"

# Chars of slack in the phoneme string around each ayah boundary when
# deciding whether a diff opcode belongs to that boundary's natural
# pause rather than genuinely falling elsewhere in the ayah - loose on
# purpose since this is a coarse, ayah-level check, not exact position
# matching.
_BOUNDARY_TOLERANCE = 6


def _load_muaalem(device="cpu"):
    global _MUAALEM
    if _MUAALEM is None:
        from quran_muaalem import Muaalem
        _MUAALEM = Muaalem(device=device)
    return _MUAALEM


def _build_reference(surah_number, ayah_start, ayah_end):
    """Returns (full_uthmani_text, ayah_uthmani_spans, phonetizer_out,
    ayah_phoneme_offsets) for the given ayah range. ayah_uthmani_spans
    is [(ayah_number, char_start, char_end)] into full_uthmani_text -
    used to filter mappings (which give original-text positions) down
    to "which ayah is this madd rule in". ayah_phoneme_offsets is the
    cumulative phoneme-string length after each ayah - the same
    coordinate space difflib's opcodes operate in, so a diff's position
    can be attributed to an ayah with no further correlation step."""
    ayah_uthmani_spans = []
    pieces = []
    offset = 0
    for ayah in range(ayah_start, ayah_end + 1):
        text = Aya(surah_number, ayah).get().uthmani
        ayah_uthmani_spans.append((ayah, offset, offset + len(text)))
        pieces.append(text)
        offset += len(text) + 1  # +1 for the joining space added below
    full_text = " ".join(pieces)

    phonetizer_out = quran_phonetizer(full_text, _MOSHAF, remove_spaces=True)

    ayah_phoneme_offsets = [0]
    for ayah in range(ayah_start, ayah_end + 1):
        text = Aya(surah_number, ayah).get().uthmani
        ph_len = len(quran_phonetizer(text, _MOSHAF, remove_spaces=True).phonemes)
        ayah_phoneme_offsets.append(ayah_phoneme_offsets[-1] + ph_len)

    return full_text, ayah_uthmani_spans, phonetizer_out, ayah_phoneme_offsets


def _ayah_for_phoneme_pos(ayah_phoneme_offsets, ayah_start, pos):
    for idx in range(len(ayah_phoneme_offsets) - 1):
        if ayah_phoneme_offsets[idx] <= pos < ayah_phoneme_offsets[idx + 1]:
            return ayah_start + idx
    return ayah_start + len(ayah_phoneme_offsets) - 2  # trailing edge case: end of last ayah


# AaredMaddRule deliberately excluded here even though it's a real madd
# rule type: an ayah only ever gets flagged for a mid-ayah diff (every
# boundary diff is unconditionally suppressed, see check_tajweed), and
# Aared Madd only ever governs an ayah-boundary position - naming it in
# a mid-ayah finding's note would point the facilitator at a letter
# that was never actually the diff's cause.
_NAMEABLE_RULE_TYPES = {
    "LazemMaddRule", "LeenMaddRule", "MaddRule", "MonfaselMaddRule",
    "MoshaddadOrModghamNoonRule", "MottaselMaddPauseRule", "MottaselMaddRule",
    "NormalMaddRule",
}


def _named_madd_rules_in_ayah(mappings, ayah_uthmani_spans, ayah_number):
    """Every distinct nameable madd rule tagged anywhere in the given
    ayah, as (english_name, arabic_name, golden_len) tuples - used to
    name the specific rule(s) in a flagged ayah's note instead of a
    generic "somewhere in this ayah". Not narrowed to the exact diff
    position (that needs the same unsolved phoneme-to-word correlation
    flagged in the module docstring) - if an ayah carries more than one
    nameable madd rule, all of them are listed since which one is
    actually short can't be pinned down yet.

    CAVEAT, found empirically: quran_transcript's own rule
    classification doesn't always match the traditional/widely-taught
    one for hard edge cases. Confirmed directly against Yaseen 36:1
    ("يس", a muqatta'at) - api.quran.com's own tajweed tagging (used
    elsewhere in this app) calls this a "necessary" madd
    (madda_necessary), but quran_transcript's phonetizer classifies the
    same letter as NormalMaddRule (2 beats) rather than LazemMaddRule (6
    beats). This function reports whatever quran_transcript says, since
    that's the same reference the acoustic comparison itself runs
    against - just worth knowing the named rule can occasionally
    disagree with traditional tajweed teaching on muqatta'at."""
    span = next((s for s in ayah_uthmani_spans if s[0] == ayah_number), None)
    if not span:
        return []
    _, start, end = span
    seen = {}
    for m in mappings:
        if not m.tajweed_rules or not (start <= m.pos[0] < end):
            continue
        for rule in m.tajweed_rules:
            cls_name = type(rule).__name__
            if cls_name not in _NAMEABLE_RULE_TYPES or cls_name in seen:
                continue
            seen[cls_name] = (rule.name.en, rule.name.ar, rule.golden_len)
    return list(seen.values())


def check_tajweed(audio_path, surah_number, ayah_start, ayah_end, allowed_error_types=None, device="cpu"):
    """Returns a list of error dicts shaped like recitation_checker.py's
    check_recitation() output (category='tajweed', word_index=0 since
    attribution is ayah-level - see module docstring), ready to insert
    into is_recitation_errors alongside the word-accuracy errors."""
    if allowed_error_types is not None and _MADD_LABEL not in allowed_error_types:
        return []  # facilitator hasn't defined this label - nothing to check for

    _full_text, ayah_uthmani_spans, phonetizer_out, ayah_phoneme_offsets = _build_reference(
        surah_number, ayah_start, ayah_end
    )

    muaalem = _load_muaalem(device=device)
    wave, _ = load(audio_path, sr=16000, mono=True)
    result = muaalem([wave], [phonetizer_out], sampling_rate=16000)[0]

    ref_phonemes = phonetizer_out.phonemes
    pred_phonemes = result.phonemes.text

    flagged_ayahs = set()
    opcodes = difflib.SequenceMatcher(None, ref_phonemes, pred_phonemes).get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        near_boundary = any(
            abs(i1 - b) <= _BOUNDARY_TOLERANCE or abs(i2 - b) <= _BOUNDARY_TOLERANCE
            for b in ayah_phoneme_offsets
        )
        if near_boundary:
            # Every near-boundary opcode is, by construction, about an
            # ayah-ending position - a natural inter-ayah/end-of-
            # recording breath, not a mistake (see module docstring).
            # An earlier version tried to be more precise by only
            # tolerating this when the phonetizer specifically tagged
            # the position AaredMaddRule (the "discretionary pause
            # elongation" rule) rather than suppressing unconditionally
            # - that backfired: the phonetizer only tags a position
            # AaredMaddRule when it's the very last ayah of the whole
            # reference passed to it, since it has no way to know a
            # real reciter also pauses between ayahs it wasn't told to
            # treat as a stop. Every other ayah's ending madd gets
            # tagged the stricter NormalMaddRule despite being the same
            # kind of natural pause acoustically, which made the
            # "smarter" version flag nearly every ayah boundary again.
            continue
        ayah_num = _ayah_for_phoneme_pos(ayah_phoneme_offsets, ayah_start, i1)
        flagged_ayahs.add(ayah_num)

    errors = []
    for ayah_num in sorted(flagged_ayahs):
        named_rules = _named_madd_rules_in_ayah(phonetizer_out.mappings, ayah_uthmani_spans, ayah_num)
        if named_rules:
            rule_list = "; ".join(f"{en} ({ar}, {golden_len} beats)" for en, ar, golden_len in named_rules)
            if len(named_rules) == 1:
                note = (
                    f"Possible {rule_list} issue detected by acoustic analysis - "
                    "not anchored to a specific word yet, please check this ayah's "
                    "madd letters governed by this rule."
                )
            else:
                note = (
                    f"Possible elongation issue detected by acoustic analysis. This ayah "
                    f"carries more than one madd rule that could be the cause: {rule_list} - "
                    "not anchored to a specific word yet, please check each."
                )
        else:
            # Shouldn't normally happen (a mid-ayah diff with no
            # nameable rule nearby would mean the phonetizer disagrees
            # with itself about the flagged span) but fall back to the
            # generic wording rather than fail the whole check.
            note = (
                "Possible elongation (madd) length issue detected somewhere in this ayah "
                "by acoustic analysis - not anchored to a specific word yet, please review "
                "the full ayah's madd letters."
            )
        entry = {
            "ayah_number": ayah_num, "word_index": 0, "word_text": "",
            "category": "tajweed", "label": _MADD_LABEL,
            "note": note,
            "source": "auto",
        }
        if allowed_error_types is not None:
            entry["error_type_id"] = allowed_error_types[_MADD_LABEL]
        errors.append(entry)
    return errors


if __name__ == "__main__":
    import json
    import sys

    audio_path, surah, a_start, a_end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    errors = check_tajweed(audio_path, surah, a_start, a_end)
    print(json.dumps(errors, ensure_ascii=False, indent=2))
