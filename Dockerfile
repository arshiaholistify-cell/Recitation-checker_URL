FROM python:3.11-slim

# librosa/audioread need ffmpeg on PATH to decode the webm/mp4/opus audio
# students actually submit (same requirement we hit developing this locally).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Plain `pip install torch` on Linux pulls the CUDA-bundled wheel (multi-GB) —
# this container is CPU-only, so install the much smaller CPU build first.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY recitation_checker.py tajweed_checker.py service.py ./

# Railway injects PORT at runtime; service.py reads it (defaults to 8090
# for local/non-Railway use).
EXPOSE 8090
CMD ["python", "service.py"]
