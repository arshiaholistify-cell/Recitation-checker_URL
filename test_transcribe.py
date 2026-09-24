import re
import difflib
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

AUDIO_FILE = "test_ayah.mp3"
EXPECTED_TEXT = "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ قُلْ هُوَ ٱللَّهُ أَحَدٌ"

def normalize_arabic(text):
    text = re.sub(r"<\|[^|]*\|>", "", text)          # strip any leaked special tokens
    text = re.sub(r"[ً-ْٰ]", "", text)  # strip diacritics/tashkeel + superscript alef
    text = text.replace("ٱ", "ا")            # alef wasla -> plain alef
    text = text.replace("آ", "ا").replace("أ", "ا").replace("إ", "ا")  # alef variants -> plain alef
    text = re.sub(r"[ۖ-ۭ]", "", text)        # strip Quranic pause/annotation marks
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Loading tarteel-ai/whisper-base-ar-quran ...")
processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-base-ar-quran")
model = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-base-ar-quran")
print("Model loaded.")

audio, sr = librosa.load(AUDIO_FILE, sr=16000, mono=True)
print(f"Loaded audio: {len(audio)/sr:.2f}s at {sr}Hz")

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
predicted_ids = model.generate(inputs.input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print()
print("RAW EXPECTED :", EXPECTED_TEXT)
print("RAW HEARD    :", transcription)

norm_expected = normalize_arabic(EXPECTED_TEXT)
norm_heard = normalize_arabic(transcription)
print()
print("NORM EXPECTED:", norm_expected)
print("NORM HEARD   :", norm_heard)
print()

target_words = norm_expected.split()
spoken_words = norm_heard.split()
matcher = difflib.SequenceMatcher(None, target_words, spoken_words)
print("Diff opcodes:", matcher.get_opcodes())
print(f"Match ratio: {matcher.ratio():.3f}")

for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    if tag == 'equal':
        for i in range(i1, i2):
            print(f"CORRECT: '{target_words[i]}'")
    elif tag == 'delete':
        for i in range(i1, i2):
            print(f"SKIPPED: '{target_words[i]}'")
    elif tag == 'replace':
        for i_t, i_s in zip(range(i1, i2), range(j1, j2)):
            print(f"MISTAKE: expected '{target_words[i_t]}' heard '{spoken_words[i_s]}'")
    elif tag == 'insert':
        for j in range(j1, j2):
            print(f"EXTRA: '{spoken_words[j]}'")
