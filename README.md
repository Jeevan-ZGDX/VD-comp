# 🎙️ SpeechPro — AI-Powered Professional Speech Correction Assistant

A fully local, offline AI assistant that converts messy spoken transcripts into
polished professional text using **Whisper** (speech-to-text) and **Ollama** (LLM correction).

---

## 📁 Project Structure

```
speech_assistant/
├── app.py                  ← CLI entry point
├── requirements.txt
├── prompts/
│   └── correction_prompt.txt
├── utils/
│   ├── audio_handler.py    ← Recording & file validation
│   ├── transcriber.py      ← Whisper STT
│   └── corrector.py        ← Ollama LLM correction
├── ui/
│   └── streamlit_app.py    ← Web UI
├── audio/                  ← Put your audio files here
└── models/                 ← Reserved for future model storage
```

---

## ⚙️ Prerequisites

### 1. Python 3.10+
```bash
python --version
```

### 2. FFmpeg (required for MP3 / M4A decoding)
- **Windows**: Download from https://ffmpeg.org/download.html and add to PATH
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### 3. Ollama
Download from https://ollama.com and install.

Pull the recommended model:
```bash
ollama pull phi3
# or for higher quality (slower):
ollama pull mistral
```

Start the Ollama server:
```bash
ollama serve
```

---

## 🚀 Setup

```bash
# 1. Clone / navigate to project folder
cd speech_assistant

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Option A — Streamlit Web UI (recommended)
```bash
streamlit run ui/streamlit_app.py
```
Then open http://localhost:8501 in your browser.

---

### Option B — Command Line

**Correct typed/pasted text directly:**
```bash
python app.py --text "uh like i wanted to discuss the project status with team you know"
```

**Transcribe and correct an audio file:**
```bash
python app.py --file audio/meeting.wav
```

**Record from microphone (10 seconds):**
```bash
python app.py --record --duration 10
```

**Use a different model:**
```bash
python app.py --text "uh send me that file fast" --model mistral
```

**Use higher-accuracy Whisper model:**
```bash
python app.py --file audio/meeting.mp3 --whisper small
```

**List available Ollama models:**
```bash
python app.py --list-models
```

---

## 📊 Example

| Input | Output |
|---|---|
| "uh i like wanted to discuss the project" | "I would like to discuss the project." |
| "so basically he don't know the status" | "He is unaware of the status." |
| "like we already completed the module yesterday you know" | "We completed the module yesterday." |
| "uh send me that file fast okay" | "Please send me the file at your earliest convenience." |

---

## 🔧 Configuration

All key settings are exposed in the Streamlit sidebar or as CLI flags:

| Setting | Default | Description |
|---|---|---|
| `--model` | `phi3` | Ollama LLM model |
| `--whisper` | `base` | Whisper model size |
| `temperature` | `0.3` | LLM temperature (lower = more consistent) |

---

## 🛠️ Troubleshooting

**"Ollama not reachable"**
→ Run `ollama serve` in a separate terminal.

**"Model not found"**
→ Run `ollama pull phi3` to download the model first.

**Audio transcription is inaccurate**
→ Switch from `base` to `small` Whisper model: `--whisper small`

**Slow inference**
→ Try `phi3` (smallest, fastest) or reduce audio length.

**No microphone / sounddevice error**
→ Install PortAudio: `sudo apt install portaudio19-dev` (Linux) or `brew install portaudio` (macOS)

---

## 🔮 Future Enhancements

- Real-time streaming correction
- Pronunciation scoring
- Multi-language support
- Meeting summarization
- LoRA fine-tuning on domain-specific correction data
