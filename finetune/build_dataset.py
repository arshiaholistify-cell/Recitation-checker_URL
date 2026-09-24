"""
Builds a Whisper fine-tuning dataset for Quranic Arabic from public,
already-known-correct sources: reciter audio (alquran.cloud) paired with
the actual verse text (api.quran.com) for the same ayah. No manual
transcription needed — the Quran's text is fixed and already known, so
alignment is just "this reciter's audio for ayah X" + "the text of ayah
X", which both APIs already index consistently by surah/ayah number
(the same approach the main app already uses for its mushaf viewer).

Bounded first-pass scope: a deliberately chosen sample of reciters and
surahs — including the specific surahs where the shipped auto-checker
was empirically found to hallucinate more (Al-Baqarah, Al-Kahf,
Al-Ahzab, Yaseen) — rather than the full 114-surah Quran, so this stays
cheap and fast to validate before deciding whether to scale up.

Output: a HuggingFace `datasets`-compatible "AudioFolder" layout
(audio/*.mp3 + metadata.csv with file_name,text columns), split into
train/ and validation/ subfolders. Run locally (CPU-only, no GPU
needed) or in the training Space before the training step.

Usage:
    python build_dataset.py [--out ./quran_dataset] [--limit N]
"""

import argparse
import csv
import random
import time
from pathlib import Path

import requests

# Five reciters spanning different styles/paces/recording conditions —
# diversity here is what should help generalization, more than raw
# volume of audio from any single reciter.
RECITERS = [
    "ar.alafasy",             # Mishary Alafasy
    "ar.abdulbasitmurattal",  # Abdul Basit, Murattal
    "ar.husary",               # Mahmoud Khalil Al-Husary
    "ar.minshawi",             # Mohamed Siddiq El-Minshawi
    "ar.abdurrahmaansudais",   # Abdur-Rahman As-Sudais
]

# All 114 surahs — the first fine-tuning pass scoped this to just 14
# surahs (including the ones the shipped auto-checker was empirically
# tested against and found to hallucinate on: 2, 18, 33, 36), which
# improved accuracy on those but caused a real regression elsewhere
# (found empirically: a Surah Al-Isra (17) recitation — never included
# in that narrower set — got nearly every word flagged wrong after the
# first fine-tune, a classic catastrophic-forgetting symptom from
# training on too narrow a slice). Training on the full Quran avoids
# creating blind spots on whichever surahs aren't in scope.
SURAHS = list(range(1, 115))

VALIDATION_FRACTION = 0.1
RANDOM_SEED = 42


def fetch_verse_text(surah_number):
    """Uthmani plain script (no tajweed markup) — matches the style the
    model already outputs, so the training target vocabulary lines up
    with what it's already predisposed to produce."""
    url = f"https://api.quran.com/api/v4/verses/by_chapter/{surah_number}?fields=text_uthmani&per_page=300"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return {v["verse_number"]: v["text_uthmani"] for v in resp.json().get("verses", [])}


def fetch_audio_urls(surah_number, reciter):
    url = f"https://api.alquran.cloud/v1/surah/{surah_number}/{reciter}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    ayahs = resp.json().get("data", {}).get("ayahs", [])
    return {a["numberInSurah"]: a["audio"] for a in ayahs}


def download_audio(url, dest_path, session, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            dest_path.write_bytes(resp.content)
            return True
        except Exception as exc:
            if attempt == retries:
                print(f"  FAILED after {retries + 1} attempts: {url} ({exc})")
                return False
            time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./quran_dataset")
    parser.add_argument("--limit", type=int, default=None, help="Cap total pairs (for a quick smoke test)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "train" / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "validation" / "audio").mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    pairs = []  # (reciter, surah, ayah, audio_url, text)

    print(f"Fetching verse text for {len(SURAHS)} surahs...")
    text_by_surah = {}
    for s in SURAHS:
        text_by_surah[s] = fetch_verse_text(s)
        time.sleep(0.2)

    print(f"Indexing audio URLs for {len(RECITERS)} reciters...")
    for reciter in RECITERS:
        for surah in SURAHS:
            try:
                audio_by_ayah = fetch_audio_urls(surah, reciter)
            except Exception as exc:
                print(f"  Could not fetch {reciter} surah {surah}: {exc}")
                continue
            texts = text_by_surah.get(surah, {})
            for ayah_num, audio_url in audio_by_ayah.items():
                text = texts.get(ayah_num)
                if not text or not audio_url:
                    continue
                pairs.append((reciter, surah, ayah_num, audio_url, text))
            time.sleep(0.2)

    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Total audio/text pairs to download: {len(pairs)}")

    random.Random(RANDOM_SEED).shuffle(pairs)
    n_val = max(1, int(len(pairs) * VALIDATION_FRACTION))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split_name, split_pairs in (("train", train_pairs), ("validation", val_pairs)):
        print(f"Downloading {split_name}: {len(split_pairs)} files...")
        rows = []
        ok, failed = 0, 0
        for i, (reciter, surah, ayah_num, audio_url, text) in enumerate(split_pairs, start=1):
            file_name = f"{reciter}_{surah:03d}_{ayah_num:03d}.mp3"
            dest = out_dir / split_name / "audio" / file_name
            if dest.exists() or download_audio(audio_url, dest, session):
                rows.append({"file_name": f"audio/{file_name}", "text": text})
                ok += 1
            else:
                failed += 1
            if i % 100 == 0 or i == len(split_pairs):
                print(f"  {split_name}: {i}/{len(split_pairs)} ({ok} ok, {failed} failed)")
        metadata_path = out_dir / split_name / "metadata.csv"
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split_name}: {ok} downloaded, {failed} failed -> {metadata_path}")

    print("Done. Load with: datasets.load_dataset('audiofolder', data_dir=" + repr(str(out_dir)) + ")")


if __name__ == "__main__":
    main()
