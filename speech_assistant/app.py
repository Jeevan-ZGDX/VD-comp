"""
app.py
CLI entry point for the AI-Powered Professional Speech Correction Assistant.

Usage:
    python app.py --file audio/sample.wav
    python app.py --record --duration 10
    python app.py --text "uh i like wanted to discuss the project"
"""

import argparse
import sys
from pathlib import Path

# Ensure utils/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from utils.audio_handler import (
    cleanup_temp_file,
    record_audio,
    save_uploaded_bytes,
    validate_audio_file,
)
from utils.corrector import check_ollama_running, correct_transcript, list_available_models
from utils.transcriber import transcribe_audio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_banner():
    print("\n" + "=" * 60)
    print("  AI-Powered Professional Speech Correction Assistant")
    print("=" * 60)


def print_section(title: str):
    print(f"\n{'─' * 40}")
    print(f"  {title}")
    print("─" * 40)


def run_pipeline(
    audio_path: str,
    model: str,
    whisper_size: str,
    cleanup: bool = False,
) -> dict:
    """
    Full pipeline: audio → transcript → corrected text.

    Returns dict with all intermediate results.
    """
    output = {
        "audio_path": audio_path,
        "transcript": None,
        "corrected": None,
        "latency_ms": None,
        "errors": [],
    }

    # 1. Transcribe
    print_section("Step 1 — Transcribing Audio")
    t_result = transcribe_audio(audio_path, model_size=whisper_size)

    if not t_result["success"]:
        output["errors"].append(f"Transcription error: {t_result['error']}")
        return output

    transcript = t_result["transcript"]
    output["transcript"] = transcript
    print(f"  Raw Transcript : {transcript}")

    # 2. Correct
    print_section("Step 2 — Correcting with Ollama")
    c_result = correct_transcript(transcript, model=model)

    if not c_result["success"]:
        output["errors"].append(f"Correction error: {c_result['error']}")
        return output

    output["corrected"] = c_result["corrected"]
    output["latency_ms"] = c_result["latency_ms"]
    print(f"  Corrected Text : {c_result['corrected']}")
    print(f"  LLM Latency    : {c_result['latency_ms']}ms")

    if cleanup:
        cleanup_temp_file(audio_path)

    return output


def run_text_only(text: str, model: str) -> dict:
    """Skip transcription — correct raw text directly."""
    output = {"transcript": text, "corrected": None, "latency_ms": None, "errors": []}

    print_section("Correcting Text with Ollama")
    c_result = correct_transcript(text, model=model)

    if not c_result["success"]:
        output["errors"].append(c_result["error"])
        return output

    output["corrected"] = c_result["corrected"]
    output["latency_ms"] = c_result["latency_ms"]
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-Powered Professional Speech Correction Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --file audio/meeting.wav
  python app.py --record --duration 15
  python app.py --text "uh i like need update on the project"
  python app.py --file audio/sample.mp3 --model mistral --whisper small
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", type=str, help="Path to audio file")
    input_group.add_argument(
        "--record", action="store_true", help="Record from microphone"
    )
    input_group.add_argument(
        "--text", type=str, help="Pass raw text directly (skips transcription)"
    )

    parser.add_argument(
        "--duration", type=int, default=10, help="Recording duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--model", type=str, default="phi3", help="Ollama model name (default: phi3)"
    )
    parser.add_argument(
        "--whisper",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available Ollama models and exit"
    )

    return parser


def main():
    print_banner()
    parser = build_parser()
    args = parser.parse_args()

    # ── List models ─────────────────────────────────────────────────────────
    if args.list_models:
        models = list_available_models()
        if models:
            print("\nAvailable Ollama models:")
            for m in models:
                print(f"  • {m}")
        else:
            print("\nNo models found. Pull one with: ollama pull phi3")
        sys.exit(0)

    # ── Preflight: check Ollama (skip for --text since it still needs Ollama) 
    print_section("Preflight Checks")
    ok, msg = check_ollama_running()
    if ok:
        print(f"  ✓ {msg}")
    else:
        print(f"  ✗ {msg}")
        print("  → Start Ollama with: ollama serve")
        sys.exit(1)

    # ── Text-only mode ───────────────────────────────────────────────────────
    if args.text:
        print(f"\n  Input Text : {args.text}")
        result = run_text_only(args.text, model=args.model)

    # ── File mode ────────────────────────────────────────────────────────────
    elif args.file:
        valid, vmsg = validate_audio_file(args.file)
        if not valid:
            print(f"  ✗ {vmsg}")
            sys.exit(1)
        print(f"  ✓ Audio file OK: {args.file}")
        result = run_pipeline(args.file, model=args.model, whisper_size=args.whisper)

    # ── Record mode ──────────────────────────────────────────────────────────
    elif args.record:
        audio_path = record_audio(duration=args.duration)
        if not audio_path:
            print("  ✗ Recording failed.")
            sys.exit(1)
        result = run_pipeline(
            audio_path, model=args.model, whisper_size=args.whisper, cleanup=True
        )

    # ── Final output ─────────────────────────────────────────────────────────
    print_section("Result")

    if result.get("errors"):
        for err in result["errors"]:
            print(f"  ✗ Error: {err}")
        sys.exit(1)

    print(f"  Input     : {result.get('transcript', '')}")
    print(f"  Output    : {result.get('corrected', '')}")
    if result.get("latency_ms"):
        print(f"  Latency   : {result['latency_ms']}ms")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
