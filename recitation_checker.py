"""
Core auto-detection logic: transcribe a recitation with Whisper
(a fine-tune of tarteel-ai/whisper-base-ar-quran, continued on longer/
less-common passages — see finetune/README.md) and diff it against the
expected ayah text to classify errors, the same way the app's manual
word-pinning does.

Validated against a real reciter's audio (Alafasy, Surah Al-Ikhlas 1) —
see test_transcribe.py for the empirical test this was extracted from.
"""

import os
import re
import difflib
import unicodedata
import librosa
import numpy as np
import requests
import torch
from scipy.signal import butter, sosfiltfilt
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _patch_extra_special_tokens_list_bug():
    """Monkey-patches around a real transformers bug, confirmed live via
    a Railway traceback and matching a documented, known issue
    (huggingface/transformers#45376): PreTrainedTokenizerBase's
    _set_model_specific_special_tokens calls .keys()/.items() on
    extra_special_tokens, i.e. it requires a dict - but this fine-tuned
    model's tokenizer_config.json (inherited from an earlier training
    run in this project, not something safely fixable here without Hub
    write credentials) already has that field in transformers 5.0's
    list format, which crashes every 4.x release with exactly
    "AttributeError: 'list' object has no attribute 'keys'".
    transformers>=4.55.0 is required here for quran-muaalem's own
    compatibility (see requirements.txt); confirmed directly that this
    validation exists at 4.55.0 already, so there's no version
    satisfying both constraints. Treating a list as "no extra special
    tokens" (a no-op) rather than crashing matches the community's own
    suggested fix for this exact issue - Whisper's standard special
    tokens are unaffected either way, since those are handled
    elsewhere, not through this model-specific extension point."""
    original = PreTrainedTokenizerBase._set_model_specific_special_tokens

    def patched(self, special_tokens):
        if isinstance(special_tokens, list):
            return
        return original(self, special_tokens)

    PreTrainedTokenizerBase._set_model_specific_special_tokens = patched


_patch_extra_special_tokens_list_bug()

# Below this average per-token confidence, a chunk's mismatches get
# flagged as "model wasn't sure" rather than asserted as a confirmed
# error — a starting threshold, not a scientifically derived cutoff;
# tune based on how it performs across more real recordings.
LOW_CONFIDENCE_THRESHOLD = 0.5

MODEL_NAME = "tarteel-ai/whisper-base-ar-quran"
_processor = None
_model = None

_LEAKED_TOKEN_RE = re.compile(r"<\|[^|]*\|>")
_WHITESPACE_RE = re.compile(r"\s+")
# Hand-picked Unicode ranges for Arabic diacritics/pause-marks kept
# missing edge cases (there's a long tail of Quranic annotation marks in
# Indo-Pak typography). Stripping by Unicode *category* instead is more
# complete: category "Mn" covers all nonspacing combining marks
# (diacritics, pause/annotation marks alike) and "Cf" covers invisible
# formatting characters (zero-width space, RTL/LTR marks, BOM, etc.).


def _load_model():
    global _processor, _model

    if _model is None:
        hf_token = (os.environ.get("HF_TOKEN") or "").strip() or None

        _processor = WhisperProcessor.from_pretrained(
            MODEL_NAME,
            token=hf_token,
        )

        _model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            token=hf_token,
        )

        # Transformers 5.x expects the Whisper generation configuration
        # to contain the language/task lookup mappings.
        tokenizer = _processor.tokenizer

        ar_token_id = tokenizer.convert_tokens_to_ids("<|ar|>")
        transcribe_token_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_token_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")

        if ar_token_id is None or ar_token_id < 0:
            raise RuntimeError("Could not find <|ar|> token in tokenizer.")

        if transcribe_token_id is None or transcribe_token_id < 0:
            raise RuntimeError("Could not find <|transcribe|> token in tokenizer.")

        if notimestamps_token_id is None or notimestamps_token_id < 0:
            raise RuntimeError("Could not find <|notimestamps|> token in tokenizer.")

        generation_config = _model.generation_config

        generation_config.lang_to_id = {
            "ar": ar_token_id,
            "<|ar|>": ar_token_id,
        }

        generation_config.task_to_id = {
            "transcribe": transcribe_token_id,
        }

        generation_config.language = "ar"
        generation_config.task = "transcribe"
        generation_config.no_timestamps_token_id = notimestamps_token_id

        print(
            "Whisper generation config repaired: "
            f"ar={ar_token_id}, "
            f"transcribe={transcribe_token_id}, "
            f"no_timestamps={notimestamps_token_id}",
            flush=True,
        )

    return _processor, _model
    


def normalize_arabic(text):
    """Strip leaked special tokens, diacritics, invisible formatting
    characters, and alef variants so Whisper's output and the
    Uthmani/Indo-Pak source text can be compared fairly — without this,
    e.g. 'Allah' (u+0671 alef wasla) vs plain alef, or a trailing
    zero-width space, would be flagged as a false error on nearly every
    ayah (found both empirically — see test_transcribe.py)."""
    text = _LEAKED_TOKEN_RE.sub("", text)
    text = text.replace("ٱ", "ا")  # alef wasla -> plain alef
    # Uthmani script (text_uthmani_tajweed) uses the "wavy hamza" alef
    # variants (ٲ/ٳ, U+0672/U+0673) in places Indo-Pak/standard spelling
    # just uses a plain hamza-alef (أ/إ) — found empirically once the
    # reference text switched from Indo-Pak to Uthmani (e.g. "ذٲلك" for
    # "ذلك" was flagged wrong until these were added here). It also
    # sometimes spells the same sound as a standalone hamza letter (ء,
    # U+0621) immediately before a plain alef instead of a single
    # precomposed letter (e.g. "ءَاتَيْنَا" for what Whisper transcribes
    # as "اتينا") — folding ء to a plain alef the same way turns that
    # into a double-alef, which _diff_key's internal-alef stripping
    # already absorbs as equivalent, confirmed empirically.
    for ch in ("آ", "أ", "إ", "ٲ", "ٳ", "ء"):  # madda/hamza-above/below alef -> plain alef
        text = text.replace(ch, "ا")
    # Small silent-letter recitation marks (small waw ۥ U+06E5, small
    # yeh ۦ U+06E6) — Quranic Hafs annotation showing a normally-silent
    # و/ي that guides pronunciation without being part of the word's
    # spelling. Like tatweel, these are Unicode category "Lm" (a letter
    # modifier), not a combining mark, so the category-based strip below
    # doesn't catch them — found the same way tatweel was: they survived
    # into a word and broke an otherwise-correct match ("حَوْلَهُۥ" vs
    # Whisper's "حوله").
    text = text.replace("ۥ", "").replace("ۦ", "")
    # Alef maksura (ى, U+0649) vs regular yeh (ي, U+064A) — pronounced
    # identically in word-final position, but Indo-Pak script and
    # Whisper's own output don't consistently agree on which one to use
    # for the same word (e.g. الذي/دوني). Left as-is here (unlike the
    # alef-variant unification above) and handled instead in
    # _diff_key(), alongside the internal-alef stripping — doing both
    # together in one place avoids a word ending up asymmetric where one
    # side's trailing real alef gets stripped by _diff_key but the other
    # side's already-converted alef-maksura survives untouched (found
    # this ordering bug directly, converting ى->ي here first before
    # _diff_key ran caused exactly that split-outcome mismatch).
    # Tatweel/kashida (ـ, U+0640) is a purely decorative letter-
    # stretching character used in Indo-Pak typography for visual
    # justification — no phonetic content at all. It isn't a combining
    # mark by Unicode category (it's "Lm", not Mn/Cf/Co), so the
    # category-based strip below doesn't catch it; found this the hard
    # way when it survived into a word and broke an otherwise-correct
    # match.
    text = text.replace("ـ", "")
    # Superscript alef (U+0670) marks a long vowel in Quranic orthography
    # (e.g. مٰلِك = "maalik") that standard spelling usually writes as a
    # real alef — but a handful of very common words (الله، الرحمن) are
    # conventionally spelled *without* that alef even in standard Arabic
    # ("defective spelling"), so there's no single rule that's correct
    # for every word: Whisper spells some of these words out in full and
    # others short, inconsistently. Left un-normalized here (stripped
    # like any other diacritic below, i.e. "not there" from either
    # spelling's perspective) — the ambiguity this creates is instead
    # resolved at the word-matching step via _diff_key(), which is
    # insensitive to a word having an extra internal alef or not,
    # without needing to guess which spelling Whisper will produce for
    # any given word.
    # Strip every remaining combining mark (diacritics, Quranic
    # pause/annotation signs) and invisible formatting character by
    # Unicode category, rather than an incomplete hand-picked list of
    # ranges. Category "Co" (Private Use Area, e.g. U+E021) is also
    # stripped — this Indo-Pak text source embeds font-specific
    # ligature/rendering hints as trailing PUA codepoints on some words
    # (invisible on screen, no phonetic meaning) that would otherwise
    # silently break otherwise-identical word comparisons.
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Mn", "Cf", "Co"))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


_HIGH_PASS_CUTOFF_HZ = 80  # below the lowest fundamental in speech; cuts mic rumble/handling noise, not voice
_TARGET_RMS = 0.1  # -20 dBFS-ish, a common speech-ASR normalization target


def _load_audio_raw(audio_path):
    """Plain 16kHz mono load, no enhancement — matches the tested
    inference path (see test_transcribe.py) that produced accurate
    transcriptions, unlike Railway's production path which ran audio
    through _load_and_enhance_audio() first. Real recordings showed
    substantial recognition errors independent of the word-alignment
    logic downstream (e.g. "لَمِنَ الْمُرْسَلِينَ" transcribed as
    "لَمِنَ ٱللَّهِ") traced to that divergence, not to the diff/
    alignment algorithm. Whisper now gets the raw, un-enhanced signal;
    _load_and_enhance_audio is kept for the acoustic checks below
    (pause detection) that were never implicated."""
    return librosa.load(audio_path, sr=16000, mono=True)


def _load_and_enhance_audio(audio_path):
    """Loads audio the same way every caller already did (16kHz mono via
    librosa), plus two cheap, standard ASR preprocessing steps neither
    caller had: a high-pass filter to remove sub-speech rumble (room
    noise, mic handling, AC hum) that a raw recording can carry, and RMS
    loudness normalization so a quiet phone recording gets boosted to
    the same effective level a loud one already has — Whisper's own
    feature extraction has no loudness normalization built in, so a
    quiet clip is disproportionately harder for it to transcribe well.
    Does not attempt real denoising (spectral subtraction etc.) — that
    risks distorting the very phonetic detail accuracy depends on;
    these two steps are safe, low-risk wins.

    No longer used ahead of Whisper transcription itself (see
    _load_audio_raw and its docstring) — kept for the acoustic checks
    below (pause detection) where the enhancement was never implicated."""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    if len(audio) == 0:
        return audio, sr

    sos = butter(4, _HIGH_PASS_CUTOFF_HZ, btype="highpass", fs=sr, output="sos")
    audio = sosfiltfilt(sos, audio).astype(np.float32)

    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-6:
        audio = audio * (_TARGET_RMS / rms)
        peak = np.max(np.abs(audio))
        if peak > 0.99:
            audio = audio * (0.99 / peak)  # avoid clipping on a normalized loud recording

    return audio, sr


def transcribe(audio_path):
    if not audio_path:
        return "", 0.0, []

    processor, model = _load_model()

    audio, sr = _load_audio_raw(audio_path)
    duration = len(audio) / sr

    try:
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            torch_dtype=dtype,
        )

        result = transcriber(
            audio,
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=True,
            # The fine-tuned checkpoint can fall into a degenerate
            # repetition loop on harder/longer passages (confirmed via
            # eval_baseline.py: "وَٱلْأَرْضِ" repeated back-to-back
            # instead of finishing the sentence) — without a cap, every
            # Railway request since the last restart hung indefinitely
            # at generation with no completion, no error, no timeout,
            # stacking up permanently-stuck "processing" jobs.
            # no_repeat_ngram_size directly blocks the model from
            # repeating the same phrase, and max_new_tokens bounds the
            # worst case per chunk instead of letting a stuck generation
            # run indefinitely.
            generate_kwargs={"no_repeat_ngram_size": 3, "max_new_tokens": 225},
        )

        chunks = result.get("chunks", [])

        # This checkpoint's tokenizer_config.json doesn't mark
        # <|startoftranscript|>/<|notimestamps|>/etc. as "special" (see
        # _patch_extra_special_tokens_list_bug's docstring above for the
        # same underlying tokenizer defect), so skip_special_tokens
        # alone — which the pipeline uses internally — doesn't strip
        # them; confirmed live via a stored transcript that literally
        # started "<|startoftranscript|><|notimestamps|>...". The old
        # manual model.generate() code worked around this by slicing off
        # the forced prompt tokens before decoding; the pipeline gives
        # no equivalent hook, so strip any leaked token markup from the
        # decoded text directly instead.
        def _clean(text):
            return _LEAKED_TOKEN_RE.sub("", text).strip()

        if chunks:
            chunk_pieces = [
                (
                    _clean(chunk.get("text", "")),
                    1.0,
                )
                for chunk in chunks
                if _clean(chunk.get("text", ""))
            ]
        else:
            text = _clean(result.get("text", ""))
            chunk_pieces = [(text, 1.0)] if text else []

        full_text = " ".join(
            text for text, _ in chunk_pieces if text
        ).strip()

        return full_text, duration, chunk_pieces

    except Exception as e:
        print(
            f"Pipeline transcription failed: {type(e).__name__}: {e}",
            flush=True,
        )
        raise

# Maps api.quran.com's tajweed rule classes (from the <tajweed class="...">
# markup in text_uthmani_tajweed — same source the app's read-only tajweed
# mushaf viewer already renders) to short human-readable labels. ham_wasl
# and silent are left out deliberately — they're pronunciation-shortcut
# notations, not rules a facilitator would need to specifically verify.
_TAJWEED_RULE_LABELS = {
    "madda_normal": "Madd (elongation)",
    "madda_permissible": "Madd (elongation)",
    "madda_necessary": "Madd (elongation)",
    "madda_obligatory": "Madd (elongation)",
    "qalqalah": "Qalqalah (echo/bounce)",
    "qalaqah": "Qalqalah (echo/bounce)",
    "ikhafa": "Ikhfa (concealment)",
    "ikhafa_shafawi": "Ikhfa (concealment)",
    "iqlab": "Iqlab (conversion)",
    "idgham_ghunnah": "Idgham (merging)",
    "idgham_wo_ghunnah": "Idgham (merging)",
    "idgham_shafawi": "Idgham (merging)",
    "idgham_mutajanisayn": "Idgham (merging)",
    "idgham_mutaqaribayn": "Idgham (merging)",
    "ghunnah": "Ghunnah (nasalization)",
    "laam_shamsiyah": "Laam Shamsiyah (sun-letter assimilation)",
}
# api.quran.com's tajweed markup uses unquoted attributes
# (<tajweed class=madda_normal>...</tajweed>), not the quoted HTML
# convention — found by inspecting the raw API response directly rather
# than assuming the quoted form.
_TAJWEED_CLASS_RE = re.compile(r'<tajweed class="?([^>"]+)"?>')
_TAJWEED_OPEN_RE = re.compile(r"^<tajweed\b", re.IGNORECASE)
_TAJWEED_CLOSE_RE = re.compile(r"^</tajweed>", re.IGNORECASE)
_TAJWEED_TAG_STRIP_RE = re.compile(r"</?tajweed[^>]*>")
# text_uthmani_tajweed appends an ayah-end marker as its own
# whitespace-separated token, e.g. <span class=end>٢</span> — not
# spoken text, and not wrapped in a <tajweed> tag so
# _TAJWEED_TAG_STRIP_RE alone doesn't catch it. Strips any HTML tag
# (not just <tajweed>) to get the token's plain text, which then lets
# _ARABIC_DIGITS_RE recognise and skip the bare ayah-number digits —
# found empirically as a "Skipped word" false positive on every ayah's
# final token once the reference text switched from text_indopak
# (which didn't embed this marker inline).
_ANY_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_ARABIC_DIGITS_RE = re.compile(r"^[٠-٩]+$")


def _split_tajweed_words(html):
    """Splits text_uthmani_tajweed into per-word HTML chunks. A plain
    whitespace split — even one that avoids cutting *inside* a tag's
    attributes — isn't enough: idgham/ikhfa/iqlab rules are specifically
    about a letter at the end of one word interacting with the next
    word's first letter, so api.quran.com routinely emits a single
    <tajweed class=...>...</tajweed> tag whose *content* spans the space
    between two words (e.g. "<tajweed class=idgham_wo_ghunnah>دًى
    ل</tajweed>", confirmed by inspecting the raw API response).
    Splitting on that space would cut the tag into an unclosed opener in
    one word and an orphan closer in the next — for plain-text
    extraction that's harmless (each fragment still contains one
    complete, independently-strippable tag), but it silently drops the
    tajweed rule from whichever word lands with only the orphan closer,
    since _tajweed_rule_for_word has no opening tag left to read the
    rule's class from. This walks the string character by character,
    tracking whether a <tajweed> tag is currently open, and closes/
    reopens it across a word boundary so every resulting chunk carries
    its own complete tag and _tajweed_rule_for_word sees the rule on
    both sides of the split."""
    words = []
    current = []
    open_tag = None
    i, n = 0, len(html)
    while i < n:
        ch = html[i]
        if ch == "<":
            close = html.find(">", i)
            if close == -1:
                current.append(html[i:])
                break
            tag = html[i:close + 1]
            current.append(tag)
            if _TAJWEED_OPEN_RE.match(tag):
                open_tag = tag
            elif _TAJWEED_CLOSE_RE.match(tag):
                open_tag = None
            i = close + 1
            continue
        if ch.isspace():
            if open_tag:
                current.append("</tajweed>")
            if current:
                words.append("".join(current))
            current = [open_tag] if open_tag else []
            while i < n and html[i].isspace():
                i += 1
            continue
        current.append(ch)
        i += 1
    if current:
        words.append("".join(current))
    return words


def _tajweed_rule_for_word(word_html):
    match = _TAJWEED_CLASS_RE.search(word_html)
    if not match:
        return None
    for cls in match.group(1).split():
        label = _TAJWEED_RULE_LABELS.get(cls)
        if label:
            return label
    return None


def fetch_expected_words(surah_number, ayah_start, ayah_end):
    """Uses api.quran.com's text_uthmani_tajweed as the single source of
    truth for both the expected word text and tajweed-rule tagging — the
    same Uthmani script the app's colour-coded tajweed mushaf view
    already renders, and the script Whisper's own output (tarteel-ai's
    Quran model) is far closer to than the Indo-Pak spelling convention
    previously used here. Comparing Whisper's Uthmani-leaning output
    against Indo-Pak reference spellings (e.g. "الصلوة" vs the Uthmani
    "الصلاة") was found empirically to be a major source of false-
    positive "wrong word" flags on otherwise-correct recitations —
    switching the reference text itself fixes this at the root, instead
    of trying to special-case every spelling divergence in _diff_key().

    The frontend's pin panel renders this same text_uthmani_tajweed
    field, tag-aware-split the same way, so word_index values here still
    line up with what the facilitator clicks on screen.

    Returns (flat, tajweed_by_position) where flat is a list of
    (ayah_number, word_index, plain_word_text), and tajweed_by_position
    is {(ayah_number, word_index): rule_label} for words that carry a
    tajweed rule — not a claim the rule was followed correctly, just a
    heads-up flag for facilitator review (see _with_tajweed_context).
    """
    url = (
        f"https://api.quran.com/api/v4/verses/by_chapter/{surah_number}"
        "?fields=text_uthmani_tajweed&per_page=300"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    verses = resp.json().get("verses", [])
    flat = []  # list of (ayah_number, word_index, word_text)
    tajweed_by_position = {}
    for v in verses:
        ayah_num = v["verse_number"]
        if ayah_num < ayah_start or ayah_num > ayah_end:
            continue
        tajweed_words = _split_tajweed_words((v.get("text_uthmani_tajweed") or "").strip())
        for idx, tw in enumerate(tajweed_words):
            plain = _ANY_TAG_STRIP_RE.sub("", tw).strip()
            # Quranic pause marks (e.g. ۚ) and the ayah-end number
            # marker appear as their own whitespace-separated "word" —
            # neither is spoken, so leaving either in would silently
            # shift alignment after them by one slot. idx is kept from
            # the *unfiltered* enumerate() so it still matches the
            # frontend's word_index for the words that ARE kept.
            if _ARABIC_DIGITS_RE.match(plain):
                continue
            if normalize_arabic(plain):
                flat.append((ayah_num, idx, plain))
                rule = _tajweed_rule_for_word(tw)
                if rule:
                    tajweed_by_position[(ayah_num, idx)] = rule
    return flat, tajweed_by_position


# Reciters conventionally say these before starting the actual ayah —
# correct etiquette, but not part of the verse text itself, so it has
# nothing to match against and would otherwise corrupt the alignment at
# the very start of every properly-recited submission.
_ISTIADHAH_WORDS = normalize_arabic("أعوذ بالله من الشيطان الرجيم").split()
_BASMALA_WORDS = normalize_arabic("بسم الله الرحمن الرحيم").split()


def _preamble_word_count(heard_words):
    """If the transcription opens with the isti'adhah and/or basmala,
    return how many leading words to drop so the real diff starts at
    the actual ayah content (0 if no match). Whisper's transcription of
    these is rarely word-for-word identical (e.g. heard "راجيم" for
    "الرجيم") so this compares *characters*, not whole words, via
    difflib — a partial/fuzzy match still finds where the known phrase
    ends within the noisy output."""
    combined_ref = " ".join(_ISTIADHAH_WORDS + _BASMALA_WORDS)
    window_word_count = min(len(heard_words), len(_ISTIADHAH_WORDS) + len(_BASMALA_WORDS) + 4)
    window_words = heard_words[:window_word_count]
    window_str = " ".join(window_words)

    matcher = difflib.SequenceMatcher(None, combined_ref, window_str)
    if matcher.ratio() < 0.5:
        return 0

    last_end_char = max((b.b + b.size for b in matcher.get_matching_blocks() if b.size > 0), default=0)
    consumed_chars = 0
    words_to_strip = 0
    for w in window_words:
        consumed_chars += len(w) + 1  # +1 for the joining space
        words_to_strip += 1
        if consumed_chars >= last_end_char:
            break
    return words_to_strip


def _confidence_note(base_note, confidences):
    """Prefix a note with a caveat when the model itself wasn't
    confident about the word(s) in question, so a garbled/hallucinated
    stretch reads as "please verify" rather than a flat assertion."""
    if not confidences:
        return base_note
    avg = sum(confidences) / len(confidences)
    if avg < LOW_CONFIDENCE_THRESHOLD:
        return f"(Low model confidence — please verify) {base_note}"
    return base_note


def _with_tajweed_context(note, ayah_num, word_idx, tajweed_by_position):
    """Appends a heads-up when the word an error was flagged on also
    carries a tajweed rule — not a claim the rule was mispronounced
    (this system can't judge that), just a flag that this specific word
    deserves closer facilitator attention for that reason too."""
    rule = tajweed_by_position.get((ayah_num, word_idx))
    if not rule:
        return note
    return f"{note} This word also carries a {rule} rule — worth checking separately."


def _detect_pauses(audio_path, min_pause_sec=2.5, top_db=35):
    """Whole-recording hesitation signal for memorisation checks: how
    many silences longer than min_pause_sec occurred. Not anchored to a
    specific word/ayah — this pipeline has no per-word timestamps, only
    30s-chunk-level text, so precise pause *location* isn't available,
    only whether/how often long pauses happened at all."""
    audio, sr = _load_and_enhance_audio(audio_path)
    intervals = librosa.effects.split(audio, top_db=top_db)
    pauses = []
    prev_end = 0
    for start, end in intervals:
        gap_sec = (start - prev_end) / sr
        if gap_sec >= min_pause_sec:
            pauses.append(round(gap_sec, 1))
        prev_end = end
    return pauses


def _is_repeat(matcher_a, matcher_b, threshold=0.6):
    """Rough check for whether two word sequences are substantially the
    same words — used to tell a memorisation self-correction repeat
    ("...huwal huwallahu ahad...") apart from a genuinely new inserted
    word, without claiming more precision than a word-overlap ratio
    actually supports."""
    if not matcher_a or not matcher_b:
        return False
    return difflib.SequenceMatcher(None, matcher_a, matcher_b).ratio() >= threshold


_DIFF_KEY_ELIDABLE_LAST = ("ا", "ى", "ي")


def _diff_key(norm_word):
    """A looser key used only for word-alignment matching, never for
    display. Handles two related sources of spelling ambiguity between
    Indo-Pak/Uthmani orthography and Whisper's own output, both found
    empirically by comparing a *known-correct* recitation's transcript
    against the reference text and seeing real words get flagged wrong:

    1. Internal long-vowel ambiguity: Uthmani/Indo-Pak sometimes spells
       a long vowel as a diacritic (already stripped by normalize_arabic
       before this runs), while Whisper spells the same sound with a
       real alef letter for some words but not others, with no reliable
       per-word rule (converting the diacritic to a real alef everywhere
       fixed "مالك"/"سبحان" but broke "الله"/"الرحمن", conventionally
       spelled short even in full vowelization). Internal (non-first,
       non-last) real alefs are dropped so matching doesn't care which
       side has one.

    2. Word-final letter ambiguity: a word's very last letter, when it's
       ا/ى/ي, is often just which convention was used for the same final
       sound (الذي vs الذى; الأقصى spelled with a final ى in Uthmani
       script but a final ا in this Indo-Pak source) rather than a real
       difference. The last character is dropped whenever it's one of
       these three, on both sides, rather than trying to guess which
       specific substitution rule applies to which word.

    3. Silent waw-as-alef-bearer: the Uthmani rasm spells a handful of
       very common words (الصلاة، الزكاة، الحياة...) with a "و" standing
       in for the long-aa sound instead of an alef (الصلوٲة، الزكوٲة،
       الحيوٲة) — a documented historical spelling convention, not a
       pronounced waw. Found empirically: after hamza-alef unification,
       "الصلوٲة" still didn't match Whisper's "الصلاة" because of that
       one extra internal "و". Interior "وا" pairs are collapsed to
       nothing (alongside the lone-alef stripping above) so both sides
       reduce to the same key. This is lossy the same way #1 is — a
       genuinely-pronounced internal "وا" (e.g. in "التوابون") would also
       be collapsed — accepted for the same reason: missing a rare, real
       difference beats cascading false positives on common words.

    Both are lossy on purpose — a rare genuine difference that happens
    to only be a final ا/ى/ي, or an internal alef, could be missed. That
    trade-off was judged better than the alternative just found in
    testing: a single early ambiguous word cascading difflib into
    flagging several correct neighbouring words as wrong too, on a
    perfectly correct recitation."""
    if len(norm_word) <= 1:
        return norm_word
    body = norm_word[:-1] if norm_word[-1] in _DIFF_KEY_ELIDABLE_LAST else norm_word
    if len(body) <= 1:
        return body
    interior = body[1:-1].replace("وا", "").replace("ا", "")
    return body[0] + interior + body[-1]


def check_recitation(audio_path, surah_number, ayah_start, ayah_end, assessment_type="recitation", allowed_error_types=None):
    """Returns (transcription, duration_seconds, errors) where errors is
    a list of dicts ready to insert into is_recitation_errors (minus the
    caller-supplied recitation_id/student_id/created_by).

    allowed_error_types, when given, is a {label: error_type_id} map
    built from the facilitator's own is_error_types rows (see
    service.py). Every finding this function can produce maps to one of
    six fixed (category, label) pairs below; a finding whose label isn't
    in that map is dropped rather than reported under a label the
    facilitator doesn't recognise or has deliberately removed — this is
    what lets a facilitator's Error Types tab actually scope what
    auto-check surfaces, instead of it always reporting a fixed builtin
    taxonomy regardless of what's configured. When None (e.g. the CLI
    below, run without a facilitator context), no filtering happens and
    every finding is reported with error_type_id left unset, matching
    this function's original behaviour.

    assessment_type controls which rules get applied to the same
    underlying word-level diff:
      "recitation"    — word/tajweed accuracy (the original behaviour).
      "memorisation"  — same word diff, but insertions that look like a
                         self-correction repeat are labelled as that
                         instead of a generic extra word, a long trailing
                         block of missed words is reported as one
                         "stopped early" finding instead of many
                         individual skipped-word entries, and a
                         whole-recording pause count is added if
                         hesitation looks significant. This does not
                         attempt to judge tajweed/pronunciation
                         correctness during memorisation any differently
                         than during recitation — that's still governed
                         by the same word-accuracy signal either way.
    """
    transcription, duration, chunk_pieces = transcribe(audio_path)
    expected, tajweed_by_position = fetch_expected_words(surah_number, ayah_start, ayah_end)
    norm_expected_words = [normalize_arabic(w) for (_, _, w) in expected]

    # Build the heard-word list and a parallel per-word confidence list
    # (every word in a chunk gets that chunk's average confidence — a
    # coarser granularity than true per-word, but robust and simple).
    heard_words, heard_confidences = [], []
    for chunk_text, chunk_conf in chunk_pieces:
        words = normalize_arabic(chunk_text).split()
        heard_words.extend(words)
        heard_confidences.extend([chunk_conf] * len(words))

    strip_n = _preamble_word_count(heard_words)
    norm_heard_words = heard_words[strip_n:]
    heard_word_confidences = heard_confidences[strip_n:]

    # The matcher itself runs on the loosened keys (alef-ambiguity
    # insensitive); every downstream use of norm_expected_words/
    # norm_heard_words below (display text, heard-span notes) still
    # reads the original, unloosened words at those same indices/
    # positions — only which words *count as equal* changes.
    diff_expected = [_diff_key(w) for w in norm_expected_words]
    diff_heard = [_diff_key(w) for w in norm_heard_words]
    opcodes = difflib.SequenceMatcher(None, diff_expected, diff_heard).get_opcodes()
    is_memorisation = assessment_type == "memorisation"
    errors = []

    def emit(entry):
        if allowed_error_types is not None:
            if entry["label"] not in allowed_error_types:
                return
            entry["error_type_id"] = allowed_error_types[entry["label"]]
        errors.append(entry)

    for opcode_i, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            continue
        if tag == "delete":
            # A trailing skip that runs all the way to the end of the
            # expected range and is more than a word or two is more
            # useful, for a memorisation check, reported as one "the
            # student stopped early" finding than as N separate
            # skipped-word rows.
            is_trailing_stop = (
                is_memorisation
                and opcode_i == len(opcodes) - 1
                and i2 == len(expected)
                and (i2 - i1) >= 3
            )
            if is_trailing_stop:
                ayah_num, word_idx, word_text = expected[i1]
                emit({
                    "ayah_number": ayah_num, "word_index": word_idx, "word_text": word_text,
                    "category": "memorisation", "label": "Stopped early / incomplete",
                    "note": f"Recitation stopped {i2 - i1} word(s) before the end of the assigned range.",
                    "source": "auto",
                })
                continue
            for i in range(i1, i2):
                ayah_num, word_idx, word_text = expected[i]
                note = _with_tajweed_context(
                    "Word was not heard in the recording.", ayah_num, word_idx, tajweed_by_position
                )
                emit({
                    "ayah_number": ayah_num, "word_index": word_idx, "word_text": word_text,
                    "category": "recitation", "label": "Skipped word",
                    "note": note, "source": "auto",
                })
        elif tag == "replace":
            heard_span = " ".join(norm_heard_words[j1:j2])
            base_note = _confidence_note(f"Heard \"{heard_span}\" instead.", heard_word_confidences[j1:j2])
            for i in range(i1, i2):
                ayah_num, word_idx, word_text = expected[i]
                note = _with_tajweed_context(base_note, ayah_num, word_idx, tajweed_by_position)
                emit({
                    "ayah_number": ayah_num, "word_index": word_idx, "word_text": word_text,
                    "category": "recitation", "label": "Wrong word",
                    "note": note, "source": "auto",
                })
        elif tag == "insert":
            # Extra word(s) with no expected position of their own —
            # anchor to the nearest expected word so the schema's
            # required ayah_number/word_index are still meaningful.
            anchor = expected[i1] if i1 < len(expected) else expected[-1]
            ayah_num, word_idx, word_text = anchor
            inserted = norm_heard_words[j1:j2]
            extra_words = " ".join(inserted)
            note = _confidence_note(f"Extra word(s) \"{extra_words}\" recited near here.", heard_word_confidences[j1:j2])
            # A self-correction repeat (reciting the same word/phrase
            # twice) shows up in the diff as an insertion that closely
            # matches the words immediately before it — worth telling
            # apart from a genuinely new, unrelated extra word.
            preceding = norm_heard_words[max(0, j1 - (j2 - j1)):j1]
            if is_memorisation and _is_repeat(inserted, preceding):
                emit({
                    "ayah_number": ayah_num, "word_index": word_idx, "word_text": word_text,
                    "category": "memorisation", "label": "Repeated word/phrase",
                    "note": f"Repeated \"{extra_words}\" near here.", "source": "auto",
                })
            else:
                emit({
                    "ayah_number": ayah_num, "word_index": word_idx, "word_text": word_text,
                    "category": "recitation", "label": "Extra word",
                    "note": note, "source": "auto",
                })

    if is_memorisation and expected:
        pauses = _detect_pauses(audio_path)
        if pauses:
            ayah_num, word_idx, word_text = expected[0]
            emit({
                "ayah_number": ayah_num, "word_index": word_idx, "word_text": word_text,
                "category": "memorisation", "label": "Hesitation / pause",
                "note": (
                    f"{len(pauses)} pause(s) of 2.5s or longer detected across the recording "
                    f"(longest: {max(pauses)}s). Not anchored to a specific word — a whole-"
                    "recording signal only."
                ),
                "source": "auto",
            })

    return transcription, duration, errors


if __name__ == "__main__":
    import sys
    import json
    audio_path, surah, a_start, a_end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    mode = sys.argv[5] if len(sys.argv) > 5 else "recitation"
    transcription, duration, errors = check_recitation(audio_path, surah, a_start, a_end, assessment_type=mode)
    print(json.dumps({"transcription": transcription, "duration": duration, "errors": errors}, ensure_ascii=False, indent=2))
