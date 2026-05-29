"""
transcriber.py
Handles speech-to-text using OpenAI Whisper (local, offline).
"""

from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Model cache — load once, reuse across calls
# ---------------------------------------------------------------------------
_whisper_model = None
_loaded_model_size = None


def load_whisper_model(model_size: str = "base") -> object:
    """
    Load (or return cached) Whisper model.

    Sizes: tiny | base | small | medium | large
    Recommendation: 'base' for speed, 'small' for better accuracy.
    """
    global _whisper_model, _loaded_model_size

    if _whisper_model is not None and _loaded_model_size == model_size:
        return _whisper_model

    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )

    print(f"[INFO] Loading Whisper model: {model_size}")
    _whisper_model = whisper.load_model(model_size)
    _loaded_model_size = model_size
    print(f"[INFO] Whisper model '{model_size}' loaded successfully.")
    return _whisper_model


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: str = "en",
) -> dict:
    """
    Transcribe an audio file to text using Whisper.

    Args:
        audio_path:  Path to audio file (WAV, MP3, M4A, etc.).
        model_size:  Whisper model size.
        language:    ISO language code (default: English).

    Returns:
        {
            "success": bool,
            "transcript": str,      # raw transcript
            "language": str,        # detected / forced language
            "error": str | None,
        }
    """
    result = {"success": False, "transcript": "", "language": language, "error": None}

    if not Path(audio_path).exists():
        result["error"] = f"Audio file not found: {audio_path}"
        return result

    try:
        model = load_whisper_model(model_size)
        print(f"[INFO] Transcribing: {audio_path}")
        raw = model.transcribe(audio_path, language=language, fp16=False)
        transcript = raw.get("text", "").strip()

        if not transcript:
            result["error"] = "Transcription returned empty text."
            return result

        result["success"] = True
        result["transcript"] = transcript
        result["language"] = raw.get("language", language)
        print(f"[INFO] Transcript: {transcript}")
        return result

    except Exception as e:
        result["error"] = f"Transcription failed: {e}"
        return result
