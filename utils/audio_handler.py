"""
audio_handler.py
Handles microphone recording and audio file validation.
"""

import os
import wave
import tempfile
from pathlib import Path

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


def validate_audio_file(file_path: str) -> tuple[bool, str]:
    """
    Validate that the given file exists and is a supported audio format.

    Returns:
        (is_valid: bool, message: str)
    """
    path = Path(file_path)

    if not path.exists():
        return False, f"File not found: {file_path}"

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, (
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    if path.stat().st_size == 0:
        return False, "Audio file is empty."

    return True, "Valid audio file."


def record_audio(duration: int = 10, sample_rate: int = 16000) -> str | None:
    """
    Record audio from the default microphone for `duration` seconds.

    Returns:
        Path to the recorded .wav file, or None on failure.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print(
            "[ERROR] sounddevice or numpy not installed. "
            "Run: pip install sounddevice numpy"
        )
        return None

    print(f"[INFO] Recording for {duration} second(s)... Speak now.")
    try:
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        print("[INFO] Recording complete.")
    except Exception as e:
        print(f"[ERROR] Recording failed: {e}")
        return None

    # Save to a temp WAV file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        print(f"[INFO] Audio saved to: {tmp.name}")
        return tmp.name
    except Exception as e:
        print(f"[ERROR] Failed to save audio: {e}")
        return None


def save_uploaded_bytes(file_bytes: bytes, suffix: str = ".wav") -> str:
    """
    Save raw bytes (e.g. from Streamlit uploader) to a temp file.

    Returns:
        Path to the saved temp file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def cleanup_temp_file(file_path: str) -> None:
    """Remove a temporary audio file if it exists."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass
