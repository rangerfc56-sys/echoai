#!/usr/bin/env python3
import math
import os
from typing import List

os.environ.setdefault("GRADIO_TEMP_DIR", "/home/surya/anuj/script2video/agent/gradio_tmp")

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import pipeline

MODEL_DIR = "/home/surya/anuj/script2video/whisper-hindi2hinglish-prime"
TARGET_SR = 16000
CHUNK_SECONDS = 30
MIN_CHUNK_SECONDS = 0.2

device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    feature_extractor=MODEL_DIR,
    ignore_warning=True,
    dtype=dtype,
    device=device,
)


def _chunk_audio(audio: np.ndarray, sr: int, chunk_seconds: int) -> List[np.ndarray]:
    chunk_len = int(sr * chunk_seconds)
    min_len = int(sr * MIN_CHUNK_SECONDS)
    chunks: List[np.ndarray] = []
    for start in range(0, len(audio), chunk_len):
        chunk = audio[start : start + chunk_len]
        if len(chunk) >= min_len:
            chunks.append(chunk)
    return chunks


def transcribe(audio_path: str, progress=gr.Progress(track_tqdm=False)) -> str:
    if not audio_path:
        return "Upload audio and click Submit."
    audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    if audio.size == 0:
        return "Empty/invalid audio."

    chunks = _chunk_audio(audio, sr, CHUNK_SECONDS)
    if not chunks:
        return "Could not create chunks from audio."

    texts: List[str] = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        progress(i / total, desc=f"Transcribing chunk {i}/{total}")
        try:
            out = asr({"array": chunk, "sampling_rate": sr}, return_timestamps=False)
            text = out.get("text", "").strip()
            if text:
                texts.append(text)
        except Exception as e:
            texts.append(f"[chunk {i} error: {e}]")

    return " ".join(texts).strip()


app = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Audio Input"),
    outputs=gr.Textbox(label="Merged Transcript", lines=12),
    title="Whisper-Hindi2Hinglish-Prime (30s Chunked)",
    description="Splits input audio into 30-second chunks, transcribes each chunk, then merges text.",
)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7863, share=True, show_error=True)
