"""Generates realistic multi-speaker sample audio files for testing and demonstrations.

Supports:
1. High-fidelity OpenAI TTS speech generation (real human speech) when OPENAI_API_KEY is available.
2. Pure-Python acoustic tone synthesis (offline, zero-dependency fallback).
"""

import io
import math
import os
import struct
import wave
from pathlib import Path

SAMPLE_DIALOGUE = [
    {
        "speaker_id": "Speaker_1",
        "name": "Alex",
        "voice": "alloy",
        "text": "Hello everyone, welcome to our Q4 product sync. Let's do a quick round of updates.",
        "frequency": 140.0,
        "duration_sec": 3.0,
    },
    {
        "speaker_id": "Speaker_2",
        "name": "Sarah",
        "voice": "nova",
        "text": "Hi Alex, this is Sarah from backend engineering. The database migration is complete and fully tested.",
        "frequency": 220.0,
        "duration_sec": 4.0,
    },
    {
        "speaker_id": "Speaker_3",
        "name": "David",
        "voice": "echo",
        "text": "David here. I will deploy the staging infrastructure and complete the security review by Friday.",
        "frequency": 115.0,
        "duration_sec": 4.0,
    },
    {
        "speaker_id": "Speaker_1",
        "name": "Alex",
        "voice": "alloy",
        "text": "Excellent. We officially agreed to launch the beta release on October 15th.",
        "frequency": 140.0,
        "duration_sec": 3.5,
    },
]


def generate_synthetic_acoustic_wav(
    output_path: str | Path,
    sample_rate: int = 16000,
    pause_sec: float = 0.6,
) -> Path:
    """Generates an acoustic multi-speaker WAV file using tone modulation and pauses."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames = bytearray()

    for item in SAMPLE_DIALOGUE:
        freq = item["frequency"]
        duration = item["duration_sec"]
        num_samples = int(sample_rate * duration)

        # Generate voiced harmonic pattern
        for i in range(num_samples):
            t = i / sample_rate
            # Syllable rhythm envelope (~3-4 syllables per second)
            envelope = 0.5 * (1.0 + math.sin(2 * math.pi * 3.5 * t))
            # Primary pitch + second harmonic for speech-like timbre
            val = (
                0.7 * math.sin(2 * math.pi * freq * t)
                + 0.3 * math.sin(2 * math.pi * (freq * 2.0) * t)
            )
            sample_val = int(val * envelope * 16000.0)
            sample_val = max(-32767, min(32767, sample_val))
            frames.extend(struct.pack("<h", sample_val))

        # Pause between turns (silence)
        pause_samples = int(sample_rate * pause_sec)
        frames.extend(struct.pack("<h", 0) * pause_samples)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))

    return path


def generate_real_speech_wav(
    output_path: str | Path,
    api_key: str | None = None,
) -> Path:
    """Generates real spoken English conversation using OpenAI TTS if API key is provided,

    falling back to acoustic synthesis if unavailable.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return generate_synthetic_acoustic_wav(output_path)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        combined_frames = bytearray()
        sample_rate = 16000

        for item in SAMPLE_DIALOGUE:
            response = client.audio.speech.create(
                model="tts-1",
                voice=item["voice"],
                input=item["text"],
                response_format="wav",
            )
            # Read bytes of generated wav
            with wave.open(io.BytesIO(response.content), "rb") as turn_wav:
                sample_rate = turn_wav.getframerate()
                turn_frames = turn_wav.readframes(turn_wav.getnframes())
                combined_frames.extend(turn_frames)

            # Add 0.7s silence between speakers
            pause_samples = int(sample_rate * 0.7)
            combined_frames.extend(struct.pack("<h", 0) * pause_samples)

        with wave.open(str(path), "wb") as out_wav:
            out_wav.setnchannels(1)
            out_wav.setsampwidth(2)
            out_wav.setframerate(sample_rate)
            out_wav.writeframes(bytes(combined_frames))

        return path
    except Exception:
        # Fallback to acoustic tone generator
        return generate_synthetic_acoustic_wav(output_path)
