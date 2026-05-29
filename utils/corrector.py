"""
corrector.py
Sends raw transcripts to a local Ollama LLM for professional correction.

Key fixes vs previous version:
- Uses STREAMING mode (stream=True) — reads token-by-token so the TCP
  connection never times out mid-response (fixes WinError 10054).
- Retry logic: up to 3 attempts with exponential back-off.
- Two-stage prompting: first generate plain corrected text, then ask for
  annotations separately. This keeps each request small and fast, which
  avoids connection resets on slower machines.
- Simplified JSON schema so smaller models (phi3/mistral) can follow it.
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "correction_prompt.txt"
_MAX_RETRIES = 3
_RETRY_DELAY = 2   # seconds, doubled each retry


# ─────────────────────────────────────────────────────────────────────────────
# Connectivity helpers
# ─────────────────────────────────────────────────────────────────────────────
def check_ollama_running() -> tuple[bool, str]:
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return (True, "Ollama is running.") if resp.status == 200 else (False, "Unexpected response.")
    except Exception as e:
        return False, f"Ollama not reachable: {e}"


def list_available_models() -> list[str]:
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Ollama call  (fixes WinError 10054)
# ─────────────────────────────────────────────────────────────────────────────
def _ollama_generate(prompt: str, model: str, temperature: float = 0.3) -> tuple[str, int]:
    """
    Call Ollama with stream=True, collect all tokens, return (full_text, latency_ms).
    Streaming keeps the TCP socket alive and avoids Windows connection resets.
    Raises urllib.error.URLError or RuntimeError on failure.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,   # cap tokens so small models don't hang
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    collected = []

    # Large timeout — streaming keeps connection alive so this is safe
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("response", "")
            collected.append(token)
            if chunk.get("done", False):
                break

    full_text = "".join(collected).strip()
    latency_ms = int((time.time() - start) * 1000)
    return full_text, latency_ms


def _ollama_generate_with_retry(
    prompt: str, model: str, temperature: float = 0.3
) -> tuple[str, int]:
    """Wraps _ollama_generate with retry + exponential back-off."""
    delay = _RETRY_DELAY
    last_err = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _ollama_generate(prompt, model, temperature)
        except Exception as e:
            last_err = e
            print(f"[WARN] Attempt {attempt}/{_MAX_RETRIES} failed: {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"All {_MAX_RETRIES} attempts failed. Last error: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────
_CORRECTION_PROMPT = """You are a professional speech coach at VDart.
Speaker role: {role}
Target audience: {audience}
Speaker name: {name}

Rewrite the following spoken text as polished, professional English.
Rules:
1. Remove ALL filler words (uh, um, like, you know, basically, so, right, kind of, sort of, actually, literally, well, okay).
2. Fix all grammar errors.
3. Match tone and vocabulary appropriate for a {role} speaking to {audience}.
4. Preserve the original meaning exactly.
5. Output ONLY the corrected sentence(s). No explanation, no labels, no quotes.

Input: "{transcript}"
"""

_ANNOTATION_PROMPT = """You are a speech coach. Analyze the original speech below and identify specific phrases that need improvement.

Speaker role: {role}
Target audience: {audience}
Original speech: "{transcript}"
Corrected version: "{corrected}"

Return a JSON object ONLY — no markdown, no explanation, no text before or after the JSON.
Use this exact structure:
{{
  "annotations": [
    {{"phrase": "exact phrase from original", "type": "filler", "suggestion": "coaching tip"}},
    {{"phrase": "exact phrase from original", "type": "grammar", "suggestion": "coaching tip"}},
    {{"phrase": "exact phrase from original", "type": "delivery", "suggestion": "coaching tip"}}
  ],
  "summary": {{
    "fillers_removed": 2,
    "grammar_fixes": 1,
    "tone_score_before": 4,
    "tone_score_after": 9,
    "top_tip": "single most important coaching tip"
  }}
}}

Annotation types allowed: filler, grammar, tone, delivery, hook, structure
Keep annotations list to 3-6 items maximum. Only flag real issues.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────
def correct_transcript(
    transcript: str,
    model: str = "phi3",
    temperature: float = 0.3,
    role: str = "Professional",
    audience: str = "General Business",
    name: str = "User",
    **_kwargs,   # absorb legacy timeout kwarg gracefully
) -> dict:
    """
    Two-stage correction pipeline:
      Stage 1 — Generate corrected text (fast, reliable)
      Stage 2 — Generate structured annotations JSON (separate smaller request)

    Returns:
        {success, corrected, annotations, summary, model, latency_ms, error}
    """
    result = {
        "success": False,
        "corrected": "",
        "annotations": [],
        "summary": {},
        "raw_response": "",
        "model": model,
        "latency_ms": 0,
        "error": None,
    }

    if not transcript or not transcript.strip():
        result["error"] = "Empty transcript provided."
        return result

    total_latency = 0

    # ── Stage 1: Corrected text ──────────────────────────────────────────────
    prompt1 = _CORRECTION_PROMPT.format(
        role=role, audience=audience, name=name,
        transcript=transcript.strip()
    )
    try:
        corrected_text, lat1 = _ollama_generate_with_retry(prompt1, model, temperature)
        total_latency += lat1
    except Exception as e:
        result["error"] = f"Correction failed: {e}"
        return result

    if not corrected_text:
        result["error"] = "Model returned empty correction."
        return result

    result["corrected"] = corrected_text

    # ── Stage 2: Annotations JSON ────────────────────────────────────────────
    prompt2 = _ANNOTATION_PROMPT.format(
        role=role, audience=audience,
        transcript=transcript.strip(),
        corrected=corrected_text,
    )
    try:
        ann_text, lat2 = _ollama_generate_with_retry(prompt2, model, temperature=0.1)
        total_latency += lat2
        parsed = _extract_json(ann_text)
        if parsed:
            result["annotations"] = parsed.get("annotations", [])
            result["summary"]     = parsed.get("summary", {})
        else:
            # Non-fatal — we still have the corrected text
            result["annotations"] = []
            result["summary"]     = {}
    except Exception:
        # Annotations are best-effort; don't fail the whole pipeline
        result["annotations"] = []
        result["summary"]     = {}

    result["success"]    = True
    result["latency_ms"] = total_latency
    result["raw_response"] = corrected_text
    return result


# ─────────────────────────────────────────────────────────────────────────────
# JSON extractor
# ─────────────────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> dict | None:
    clean = text.strip()
    # Strip markdown fences
    for fence in ["```json", "```JSON", "```"]:
        if clean.startswith(fence):
            clean = clean[len(fence):]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    # Direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Find outermost { ... }
    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(clean[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None