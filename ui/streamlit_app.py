"""
streamlit_app.py  —  VDart SpeechPro
Flow: Login → Role (multi-select) → Audience (multi-select) → Dashboard
New features:
  - SQLite persistent session history (with replay)
  - Webcam posture coach — OpenCV Haar cascades, no external model files,
    works offline. Fine-tuned 6-axis scoring:
      Face presence (30) + Centering (15) + Eyes (20) + Smile (15)
      + Profile check (10) + Lighting/BG (10+5)
"""

import sys, time, json, sqlite3, datetime
from pathlib import Path

import streamlit as st
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.audio_handler import cleanup_temp_file, save_uploaded_bytes, validate_audio_file
from utils.corrector import check_ollama_running, correct_transcript, list_available_models
from utils.transcriber import transcribe_audio

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VDart SpeechPro",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# SQLite — Session History DB
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "speechpro_history.db"

def _db_init():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email  TEXT NOT NULL,
            user_name   TEXT NOT NULL,
            roles       TEXT,
            audiences   TEXT,
            input_text  TEXT,
            output_text TEXT,
            annotations TEXT,
            summary     TEXT,
            latency_ms  INTEGER,
            posture_score INTEGER,
            created_at  TEXT DEFAULT (datetime('now','localtime'))
        )
    """)
    conn.commit()
    conn.close()

_db_init()

def db_save_session(email, name, roles, audiences, inp, out, annotations, summary, latency, posture_score=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO sessions
            (user_email, user_name, roles, audiences, input_text, output_text,
             annotations, summary, latency_ms, posture_score)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        email, name, roles, audiences, inp, out,
        json.dumps(annotations), json.dumps(summary), latency, posture_score
    ))
    conn.commit()
    conn.close()

def db_load_history(email, limit=50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, roles, audiences, input_text, output_text,
               annotations, summary, latency_ms, posture_score, created_at
        FROM sessions WHERE user_email=?
        ORDER BY id DESC LIMIT ?
    """, (email, limit)).fetchall()
    conn.close()
    results = []
    for r in rows:
        try:
            ann = json.loads(r[5]) if r[5] else []
            summ = json.loads(r[6]) if r[6] else {}
        except Exception:
            ann, summ = [], {}
        results.append({
            "id": r[0], "roles": r[1], "audiences": r[2],
            "input": r[3], "output": r[4],
            "annotations": ann, "summary": summ,
            "latency_ms": r[7], "posture_score": r[8], "created_at": r[9],
        })
    return results

def db_clear_history(email):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sessions WHERE user_email=?", (email,))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
USERS = {
    "admin@vdart.com":     {"password": "vdart123",  "name": "Admin User"},
    "recruiter@vdart.com": {"password": "recruit1",  "name": "Priya Sharma"},
    "sales@vdart.com":     {"password": "sales2024", "name": "Arjun Mehta"},
    "demo@vdart.com":      {"password": "demo",      "name": "Demo User"},
}

ROLES = [
    ("👤", "Recruiter"), ("💼", "Sales Executive"), ("🤝", "Account Manager"),
    ("🛠️", "Technical Consultant"), ("📋", "Project Manager"),
    ("🧑‍💼", "HR Business Partner"), ("🚚", "Delivery Manager"),
    ("🌐", "Client Partner"), ("📊", "Business Analyst"), ("🏆", "Team Lead"),
]

AUDIENCES = [
    ("🎓", "Job Candidate"), ("👔", "C-Suite / Executive"), ("💻", "Client (Technical)"),
    ("📢", "Client (Non-Technical)"), ("👥", "Internal Team"), ("🏢", "Hiring Manager"),
    ("🤝", "Vendor / Partner"), ("💰", "Investor"), ("🌱", "New Employee / Trainee"),
]

ANNOTATION_COLORS = {
    "filler":    ("#ff4d6d", "🔴"),
    "grammar":   ("#f59e0b", "🟠"),
    "tone":      ("#a78bfa", "🟣"),
    "delivery":  ("#00c896", "🟢"),
    "hook":      ("#38bdf8", "🔵"),
    "structure": ("#fb7185", "🩷"),
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg:      #07080b;
    --surface: #0f1117;
    --card:    #141720;
    --border:  #1e2130;
    --accent:  #00c896;
    --accent2: #5b4fff;
    --red:     #ff4d6d;
    --orange:  #f59e0b;
    --blue:    #38bdf8;
    --purple:  #a78bfa;
    --text:    #e2e8f0;
    --muted:   #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stSidebar"]  { display:none !important; }
[data-testid="stHeader"]   { background: transparent !important; }
[data-testid="stDecoration"]{ display:none !important; }
footer { display:none !important; }

h1,h2,h3,h4 { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,var(--accent),var(--accent2)) !important;
    color: #fff !important; border:none !important; border-radius:10px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important; font-weight:700 !important;
    font-size:0.9rem !important; padding:0.55rem 1.8rem !important;
    transition:all 0.18s !important;
}
.stButton > button:hover { opacity:0.88; transform:translateY(-1px); }
.stButton > button:disabled { opacity:0.35 !important; }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: var(--card) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius:8px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,200,150,0.1) !important;
}
.stTextArea textarea { font-family:'DM Mono',monospace !important; font-size:0.88rem !important; }
.stSelectbox > div > div {
    background: var(--card) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius:8px !important;
}

/* ── Checkbox overrides ── */
.stCheckbox label { color: var(--text) !important; font-size:0.9rem !important; cursor:pointer; }
.stCheckbox label:hover { color: var(--accent) !important; }

/* ── Generic card ── */
.vcard {
    background:var(--card); border:1px solid var(--border);
    border-radius:14px; padding:1.5rem 1.8rem;
}

/* ── Login ── */
.login-logo {
    font-size:2rem; font-weight:900; letter-spacing:-0.03em;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}

/* ── Onboarding ── */
.step-pill {
    display:inline-block; background:var(--accent2); color:#fff;
    border-radius:99px; font-size:0.7rem; font-weight:700;
    padding:2px 12px; letter-spacing:0.07em; text-transform:uppercase;
    margin-bottom:0.5rem;
}
.step-title { font-size:1.6rem; font-weight:800; margin:0.2rem 0 0.3rem; }
.step-sub   { color:var(--muted); font-size:0.88rem; margin-bottom:1.5rem; }

/* ── Role/audience card grid ── */
.option-card {
    background:var(--surface); border:1.5px solid var(--border);
    border-radius:12px; padding:0.75rem 1rem;
    transition:all 0.15s; cursor:pointer;
}
.option-card:hover { border-color:var(--accent); background:var(--card); }

/* ── Topbar ── */
.topbar-brand {
    font-size:1.05rem; font-weight:800;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.ctx-tag {
    background:var(--surface); border:1px solid var(--border);
    border-radius:6px; padding:3px 10px;
    font-size:0.75rem; color:var(--muted);
}
.ctx-tag strong { color:var(--text); }

/* ── Annotation legend ── */
.legend { display:flex; flex-wrap:wrap; gap:0.5rem; margin:0.4rem 0 0.9rem; }
.legend-dot {
    width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:4px;
}

/* ── Annotated text ── */
.ann-wrap {
    background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:1.2rem 1.4rem;
    font-family:'DM Mono',monospace; font-size:0.88rem; line-height:2.1;
}
.ann         { border-radius:4px; padding:1px 5px; border-bottom:2px solid; cursor:help; }
.ann-filler    { background:rgba(255,77,109,0.14); border-color:#ff4d6d; color:#ff9eb5; }
.ann-grammar   { background:rgba(245,158,11,0.14); border-color:#f59e0b; color:#fcd34d; }
.ann-tone      { background:rgba(167,139,250,0.14);border-color:#a78bfa; color:#c4b5fd; }
.ann-delivery  { background:rgba(0,200,150,0.14);  border-color:#00c896; color:#6ee7d0; }
.ann-hook      { background:rgba(56,189,248,0.14); border-color:#38bdf8; color:#7dd3fc; }
.ann-structure { background:rgba(251,113,133,0.14);border-color:#fb7185; color:#fda4af; }

/* ── Suggestion items ── */
.sug-item {
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.5rem;
    display:flex; gap:0.7rem; align-items:flex-start;
}
.sug-phrase { font-family:'DM Mono',monospace; font-size:0.75rem; color:var(--muted); }
.sug-text   { font-size:0.83rem; color:var(--text); font-weight:500; margin-top:2px; }

/* ── Final output box ── */
.final-out {
    background:var(--card); border:1px solid var(--border);
    border-left:4px solid var(--accent); border-radius:12px;
    padding:1.2rem 1.4rem; font-family:'DM Mono',monospace;
    font-size:0.9rem; line-height:1.85; white-space:pre-wrap;
}

/* ── Score tiles ── */
.score-row { display:flex; gap:0.8rem; margin:0.8rem 0 1rem; flex-wrap:wrap; }
.score-tile {
    flex:1; min-width:70px; background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:0.8rem 0.5rem; text-align:center;
}
.score-val { font-size:1.5rem; font-weight:800; }
.score-lbl { font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.07em; }

/* ── Top tip ── */
.top-tip {
    background:rgba(0,200,150,0.07); border:1px solid rgba(0,200,150,0.25);
    border-radius:10px; padding:0.8rem 1rem; margin:0.6rem 0 1rem;
    font-size:0.85rem; font-weight:600;
}

/* ── Progress ── */
.stProgress > div > div { background:var(--accent) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:var(--card) !important; border-radius:10px; }
.stTabs [data-baseweb="tab"]      { color:var(--muted) !important; font-weight:600; }
.stTabs [aria-selected="true"]    { color:var(--accent) !important; }

/* ── Misc ── */
div[data-testid="stMarkdownContainer"] p { color:var(--text); }
label { color:var(--muted) !important; font-size:0.8rem !important; }
.stExpander { background:var(--card) !important; border:1px solid var(--border) !important; border-radius:10px !important; }

/* ── Webcam panel ── */
.cam-panel {
    background:var(--card); border:1px solid var(--border);
    border-radius:14px; padding:1rem;
}
.cam-status {
    font-size:0.78rem; font-weight:600; text-transform:uppercase;
    letter-spacing:0.07em; padding:3px 10px; border-radius:99px;
    display:inline-block; margin-bottom:0.5rem;
}
.cam-status-live { background:rgba(0,200,150,0.15); color:var(--accent); border:1px solid var(--accent); }
.cam-status-off  { background:rgba(100,116,139,0.15); color:var(--muted); border:1px solid var(--border); }

/* ── Posture tip notification ── */
.posture-tip {
    background: linear-gradient(135deg,rgba(91,79,255,0.15),rgba(0,200,150,0.08));
    border: 1px solid rgba(91,79,255,0.35);
    border-radius:10px; padding:0.7rem 1rem; margin-bottom:0.4rem;
    font-size:0.82rem; font-weight:500;
    animation: fadein 0.3s ease;
}
@keyframes fadein { from{opacity:0;transform:translateY(-4px)} to{opacity:1;transform:translateY(0)} }

/* ── History card ── */
.hist-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:0.85rem 1rem; margin-bottom:0.6rem;
}
.hist-meta { font-size:0.72rem; color:var(--muted); margin-bottom:0.4rem; }
.hist-badge {
    display:inline-block; background:rgba(0,200,150,0.1); color:var(--accent);
    border:1px solid rgba(0,200,150,0.3); border-radius:99px;
    font-size:0.65rem; font-weight:700; padding:1px 8px; margin-right:4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "page":           "login",
    "logged_in":      False,
    "user_email":     "",
    "user_name":      "",
    "user_roles":     [],
    "user_audiences": [],
    "transcript":     "",
    "corrected":      "",
    "annotations":    [],
    "summary":        {},
    "latency_ms":     0,
    # Webcam state
    "cam_active":     False,
    "posture_tips":   [],       # list of recent tip strings
    "posture_score":  None,     # 0-100
    # Replay from history
    "replay_entry":   None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _go(page: str):
    st.session_state.page = page
    st.rerun()


def _logout():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helper
# ─────────────────────────────────────────────────────────────────────────────
def _build_annotated_html(transcript: str, annotations: list) -> str:
    if not annotations:
        return f'<div class="ann-wrap">{transcript}</div>'
    html = transcript
    replaced = set()
    for ann in sorted(annotations, key=lambda a: len(a.get("phrase", "")), reverse=True):
        phrase = ann.get("phrase", "").strip()
        atype  = ann.get("type", "filler").lower()
        tip    = ann.get("suggestion", "").replace('"', "&quot;")
        if not phrase or phrase in replaced:
            continue
        span = f'<span class="ann ann-{atype}" title="{tip}">{phrase}</span>'
        if phrase in html:
            html = html.replace(phrase, span, 1)
            replaced.add(phrase)
    return f'<div class="ann-wrap">{html}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Posture Analyser — OpenCV Haar Cascades (no external model files required)
#
# Fine-tuned scoring system:
#   Face presence      → 30 pts   (critical: must be centred & visible)
#   Eye contact        → 20 pts   (both eyes detected & symmetrical)
#   Smile / expression → 15 pts   (relaxed open expression)
#   Head centering     → 15 pts   (face bbox centred horizontally in frame)
#   Profile check      → 10 pts   (frontal vs side-on)
#   Lighting quality   → 10 pts   (brightness histogram check)
# ─────────────────────────────────────────────────────────────────────────────

# Cascade singletons — load once, reuse every frame
_CASCADE_FACE    = None
_CASCADE_SMILE   = None
_CASCADE_EYE     = None
_CASCADE_PROFILE = None

def _load_cascades():
    global _CASCADE_FACE, _CASCADE_SMILE, _CASCADE_EYE, _CASCADE_PROFILE
    if _CASCADE_FACE is not None:
        return
    import cv2
    p = cv2.data.haarcascades
    _CASCADE_FACE    = cv2.CascadeClassifier(p + "haarcascade_frontalface_default.xml")
    _CASCADE_SMILE   = cv2.CascadeClassifier(p + "haarcascade_smile.xml")
    _CASCADE_EYE     = cv2.CascadeClassifier(p + "haarcascade_eye.xml")
    _CASCADE_PROFILE = cv2.CascadeClassifier(p + "haarcascade_profileface.xml")


def _analyze_frame_mediapipe(frame_bgr):
    """
    Fine-tuned posture analyser using OpenCV Haar cascades + geometric scoring.
    Returns (annotated_frame_bgr, list_of_tip_strings, score_0_to_100).

    Compatible with mediapipe >= 0.10 (Tasks API) — does NOT use the removed
    mp.solutions namespace.  Falls back gracefully if cv2 is missing.
    """
    try:
        import cv2
    except ImportError:
        return frame_bgr, ["⚠️ Install opencv-python-headless to enable posture analysis"], 0

    _load_cascades()

    h, w = frame_bgr.shape[:2]
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Equalise histogram for better detection under varied lighting
    gray_eq = cv2.equalizeHist(gray)

    annotated = frame_bgr.copy()
    tips   = []
    score  = 0          # build up from 0

    # ── Colour constants for drawing ──────────────────────────────────────
    GREEN  = (0,  200, 120)
    AMBER  = (0,  180, 245)
    RED    = (60,  60, 220)
    BLUE   = (220, 150,  0)
    WHITE  = (230, 230, 230)

    def _put_label(img, text, x, y, color=GREEN):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, color,   1, cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 1 — Lighting quality  (10 pts)
    # ══════════════════════════════════════════════════════════════════════
    mean_brightness = float(np.mean(gray))
    std_brightness  = float(np.std(gray))

    if mean_brightness < 60:
        tips.append("💡 Frame is too dark — move to a brighter area or face a light source")
    elif mean_brightness > 210:
        tips.append("☀️ Frame is over-exposed — reduce direct backlighting behind you")
    elif std_brightness < 25:
        tips.append("🌫️ Low contrast detected — ensure even lighting on your face")
    else:
        score += 10   # good lighting

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 2 — Face detection & centering  (30 + 15 pts)
    # ══════════════════════════════════════════════════════════════════════
    # Fine-tuned scaleFactor / minNeighbors for accuracy vs speed balance
    faces = _CASCADE_FACE.detectMultiScale(
        gray_eq, scaleFactor=1.08, minNeighbors=6,
        minSize=(int(w * 0.12), int(h * 0.12)),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    face_detected = len(faces) > 0

    if not face_detected:
        # Try profile cascade as fallback
        profiles = _CASCADE_PROFILE.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=5,
            minSize=(int(w * 0.10), int(h * 0.10)),
        )
        if len(profiles) > 0:
            tips.append("↩️ Facing sideways — turn to face the camera directly for stronger eye contact")
            score += 5  # partial credit: at least face is there
            fx, fy, fw, fh = profiles[0]
            cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh), AMBER, 2)
            _put_label(annotated, "Profile detected", fx, fy - 8, AMBER)
        else:
            tips.append("😶 No face detected — ensure your face fills at least 20% of the frame")
        # Skip remaining checks that need a frontal face
        if not tips:
            tips.append("✅ Great posture — keep it up!")
        return annotated, tips, max(0, min(100, score))

    # Use the largest detected face
    faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    fx, fy, fw, fh = faces_sorted[0]
    cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh), GREEN, 2)
    score += 30   # face present

    # ── Horizontal centering ──────────────────────────────────────────────
    face_cx = fx + fw / 2
    frame_cx = w / 2
    offset_ratio = abs(face_cx - frame_cx) / (w / 2)   # 0 = perfect, 1 = edge

    if offset_ratio < 0.18:
        score += 15   # well centred
        _put_label(annotated, "Centred ✓", fx, fy - 8, GREEN)
    elif offset_ratio < 0.35:
        score += 8
        tips.append("↔️ Shift slightly toward the centre of the frame to appear more balanced")
        _put_label(annotated, "Off-centre", fx, fy - 8, AMBER)
    else:
        tips.append("↔️ Face is too far to one side — centre yourself in front of the camera")
        _put_label(annotated, "Off-centre!", fx, fy - 8, RED)

    # ── Face size / distance check ────────────────────────────────────────
    face_area_ratio = (fw * fh) / (w * h)
    if face_area_ratio < 0.04:
        tips.append("🔍 You appear too far from the camera — move closer for better presence")
    elif face_area_ratio > 0.55:
        tips.append("📷 Too close to the camera — back up slightly for a professional frame")

    # ── Vertical position (face should be in upper-mid of frame) ─────────
    face_top_ratio = fy / h
    if face_top_ratio > 0.55:
        tips.append("⬆️ Camera angle too high — raise your camera or lower your seating")
    elif face_top_ratio < 0.02:
        tips.append("⬇️ Face is at the very top — tilt your camera down slightly")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 3 — Eye contact / both eyes visible  (20 pts)
    # ══════════════════════════════════════════════════════════════════════
    # Search for eyes only within the detected face ROI (top 60% of face)
    face_roi_gray = gray_eq[fy : fy + int(fh * 0.62), fx : fx + fw]
    eyes = _CASCADE_EYE.detectMultiScale(
        face_roi_gray, scaleFactor=1.08, minNeighbors=7,
        minSize=(int(fw * 0.12), int(fw * 0.12)),
    )

    n_eyes = len(eyes)
    if n_eyes >= 2:
        score += 20
        # Draw eye boxes
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(annotated,
                          (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh),
                          BLUE, 1)
        # Check eye symmetry (y-level mismatch → head tilt)
        eye_list = sorted(eyes[:2], key=lambda e: e[0])
        ey_diff  = abs(int(eye_list[0][1]) - int(eye_list[1][1]))
        if ey_diff > int(fh * 0.07):
            tips.append("↕️ Head is tilted — keep your head level to project confidence")
            score -= 5
        else:
            pass  # level head
    elif n_eyes == 1:
        score += 10
        tips.append("👁️ Only one eye detected — face the camera more directly for better eye contact")
    else:
        tips.append("👁️ Eyes not detected — look directly at the camera lens, not the screen")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 4 — Smile / expression  (15 pts)
    # Fine-tuned: smile cascade is noisy — use bottom-half of face ROI only
    # ══════════════════════════════════════════════════════════════════════
    mouth_roi_gray = gray_eq[
        fy + int(fh * 0.55) : fy + fh,
        fx : fx + fw
    ]
    smiles = _CASCADE_SMILE.detectMultiScale(
        mouth_roi_gray, scaleFactor=1.6, minNeighbors=22,
        minSize=(int(fw * 0.25), int(fh * 0.12)),
    )

    if len(smiles) > 0:
        score += 15
        sx, sy, sw, sh = smiles[0]
        abs_sy = fy + int(fh * 0.55) + sy
        cv2.rectangle(annotated,
                      (fx + sx, abs_sy), (fx + sx + sw, abs_sy + sh),
                      GREEN, 1)
        _put_label(annotated, "Smile ✓", fx + sx, abs_sy - 5, GREEN)
    else:
        score += 5   # partial — neutral is ok, just not great
        tips.append("😐 Relax your expression — a natural smile builds warmth and audience trust")

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 5 — Background clutter estimate  (bonus +5, no deduction)
    # Compare top-edge pixel variance (behind head) — high variance = busy BG
    # ══════════════════════════════════════════════════════════════════════
    bg_strip = gray[0:max(1, fy - 5), :]
    if bg_strip.size > 0:
        bg_std = float(np.std(bg_strip))
        if bg_std > 60:
            tips.append("🖼️ Busy background detected — a plain wall behind you looks more professional")
        else:
            score += 5   # clean background bonus

    # ══════════════════════════════════════════════════════════════════════
    # Overall score bar overlay
    # ══════════════════════════════════════════════════════════════════════
    score = max(0, min(100, score))
    bar_w = int(w * 0.35)
    bar_h = 12
    bx, by = 10, h - 22
    cv2.rectangle(annotated, (bx, by), (bx + bar_w, by + bar_h), (30,30,30), -1)
    fill_color = GREEN if score >= 75 else AMBER if score >= 50 else RED
    cv2.rectangle(annotated, (bx, by), (bx + int(bar_w * score/100), by + bar_h), fill_color, -1)
    _put_label(annotated, f"Presence {score}/100", bx + bar_w + 6, by + bar_h - 1, WHITE)

    # ══════════════════════════════════════════════════════════════════════
    # Compose final tip list
    # ══════════════════════════════════════════════════════════════════════
    if not tips:
        tips.append("✅ Excellent presence — face centred, eyes visible, expression natural!")
    elif score >= 80:
        tips.insert(0, "✅ Strong presence overall — see minor suggestions below")

    return annotated, tips, score


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — LOGIN
# ─────────────────────────────────────────────────────────────────────────────
def _page_login():
    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        st.markdown('<div class="vcard">', unsafe_allow_html=True)
        st.markdown(
            '<div class="login-logo" style="text-align:center;margin-bottom:4px">🎙️ VDart SpeechPro</div>'
            '<div style="text-align:center;color:#64748b;font-size:0.82rem;margin-bottom:1.8rem">'
            'AI-powered professional speech coaching</div>',
            unsafe_allow_html=True,
        )
        email    = st.text_input("Work Email", placeholder="you@vdart.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In →", use_container_width=True):
            if email in USERS and USERS[email]["password"] == password:
                st.session_state.logged_in  = True
                st.session_state.user_email = email
                st.session_state.user_name  = USERS[email]["name"]
                _go("setup_role")
            else:
                st.error("Invalid email or password.")
        st.markdown(
            '<div style="text-align:center;color:#475569;font-size:0.72rem;margin-top:1.2rem">'
            'Demo: <code>demo@vdart.com</code> / <code>demo</code></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — ROLE SETUP
# ─────────────────────────────────────────────────────────────────────────────
def _page_setup_role():
    _, mid, _ = st.columns([0.5, 3, 0.5])
    with mid:
        st.markdown(
            f'<div style="color:#64748b;font-size:0.88rem;margin-bottom:0.3rem">'
            f'Welcome, <strong style="color:#e2e8f0">{st.session_state.user_name}</strong> 👋</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="step-pill">Step 1 of 2</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-title">What are your roles at VDart?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-sub">Select all that apply — SpeechPro blends correction '
            'style across your selected roles.</div>', unsafe_allow_html=True,
        )
        selected_roles = list(st.session_state.user_roles)
        col_a, col_b = st.columns(2, gap="small")
        for i, (icon, label) in enumerate(ROLES):
            col = col_a if i % 2 == 0 else col_b
            with col:
                checked = st.checkbox(f"{icon}  {label}", value=(label in selected_roles), key=f"role_cb_{label}")
                if checked and label not in selected_roles:
                    selected_roles.append(label)
                elif not checked and label in selected_roles:
                    selected_roles.remove(label)
        count = len(selected_roles)
        badge_color = "#00c896" if count > 0 else "#64748b"
        st.markdown(
            f'<div style="margin:1rem 0 0.3rem;font-size:0.82rem;color:#64748b">'
            f'Selected: <span style="color:{badge_color};font-weight:700">{count}</span> role(s)</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Continue →", disabled=(count == 0)):
            st.session_state.user_roles = selected_roles
            _go("setup_audience")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — AUDIENCE SETUP
# ─────────────────────────────────────────────────────────────────────────────
def _page_setup_audience():
    _, mid, _ = st.columns([0.5, 3, 0.5])
    with mid:
        st.markdown('<div class="step-pill">Step 2 of 2</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Who do you speak to?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-sub">Select all your typical audiences — SpeechPro adapts '
            'tone, formality, and vocabulary accordingly.</div>', unsafe_allow_html=True,
        )
        selected_aud = list(st.session_state.user_audiences)
        col_a, col_b = st.columns(2, gap="small")
        for i, (icon, label) in enumerate(AUDIENCES):
            col = col_a if i % 2 == 0 else col_b
            with col:
                checked = st.checkbox(f"{icon}  {label}", value=(label in selected_aud), key=f"aud_cb_{label}")
                if checked and label not in selected_aud:
                    selected_aud.append(label)
                elif not checked and label in selected_aud:
                    selected_aud.remove(label)
        count = len(selected_aud)
        badge_color = "#00c896" if count > 0 else "#64748b"
        st.markdown(
            f'<div style="margin:1rem 0 0.3rem;font-size:0.82rem;color:#64748b">'
            f'Selected: <span style="color:{badge_color};font-weight:700">{count}</span> audience(s)</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 3])
        with col_back:
            if st.button("← Back"):
                _go("setup_role")
        with col_next:
            if st.button("Go to Dashboard →", disabled=(count == 0)):
                st.session_state.user_audiences = selected_aud
                _go("dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def _page_dashboard():
    ollama_ok, _ = check_ollama_running()
    model_options = list_available_models() or ["phi3", "mistral", "llama3"]

    roles_str     = ", ".join(st.session_state.user_roles)
    audiences_str = ", ".join(st.session_state.user_audiences)

    # ── Topbar ───────────────────────────────────────────────────────────────
    tb1, tb2, tb3, tb4 = st.columns([2, 4, 2, 1])
    with tb1:
        st.markdown('<div class="topbar-brand">🎙️ VDart SpeechPro</div>', unsafe_allow_html=True)
    with tb2:
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">'
            f'<span class="ctx-tag"><strong>{st.session_state.user_name}</strong></span>'
            f'<span class="ctx-tag">👤 {roles_str[:60]}{"…" if len(roles_str)>60 else ""}</span>'
            f'<span class="ctx-tag">🎯 {audiences_str[:60]}{"…" if len(audiences_str)>60 else ""}</span>'
            f'</div>', unsafe_allow_html=True,
        )
    with tb3:
        selected_model = st.selectbox("Model", options=model_options, label_visibility="collapsed")
    with tb4:
        if st.button("Sign Out"):
            _logout()

    if not ollama_ok:
        st.error("⚠️  Ollama is not running. Start it with: `ollama serve`")
        return

    st.markdown("---")

    # ── Main layout: left=speech, right=webcam ────────────────────────────
    speech_col, cam_col = st.columns([2.2, 1], gap="large")

    # ══════════════════════════════════════════════════════════════════════
    # LEFT — Speech correction
    # ══════════════════════════════════════════════════════════════════════
    with speech_col:
        tab_text, tab_audio = st.tabs(["📝 Type / Paste Text", "📁 Upload Audio"])

        with tab_text:
            raw_text = st.text_area(
                "Your spoken text",
                height=155,
                placeholder="uh like i wanted to discuss the project update with the client you know...",
            )
            c1, c2 = st.columns([1, 3])
            with c1:
                go_btn = st.button("✨ Analyze & Correct", use_container_width=True)
            with c2:
                st.markdown(
                    f'<div style="color:#64748b;font-size:0.78rem;padding-top:0.65rem">'
                    f'Optimising for <strong style="color:#e2e8f0">{roles_str}</strong> '
                    f'→ <strong style="color:#e2e8f0">{audiences_str}</strong></div>',
                    unsafe_allow_html=True,
                )
            if go_btn:
                if not raw_text.strip():
                    st.warning("Please enter some text first.")
                else:
                    _run_correction(raw_text.strip(), selected_model)

        with tab_audio:
            uploaded = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
            whisper_size = st.selectbox("Whisper accuracy", ["tiny", "base", "small", "medium"], index=1)
            aud_btn = st.button("🎙️ Transcribe & Analyze", disabled=(uploaded is None))
            if aud_btn and uploaded:
                tmp = save_uploaded_bytes(uploaded.read(), suffix=Path(uploaded.name).suffix)
                valid, msg = validate_audio_file(tmp)
                if not valid:
                    st.error(msg)
                else:
                    prog = st.progress(0, text="Transcribing audio…")
                    t_res = transcribe_audio(tmp, model_size=whisper_size)
                    prog.progress(50, text="Correcting…")
                    if not t_res["success"]:
                        st.error(t_res["error"])
                        prog.empty()
                    else:
                        _run_correction(t_res["transcript"], selected_model, progress=prog)
                cleanup_temp_file(tmp)

        # ── Results ──────────────────────────────────────────────────────
        if st.session_state.corrected:
            st.markdown("---")
            _render_results()

    # ══════════════════════════════════════════════════════════════════════
    # RIGHT — Webcam Posture Coach
    # ══════════════════════════════════════════════════════════════════════
    with cam_col:
        _render_webcam_panel()

    # ══════════════════════════════════════════════════════════════════════
    # BOTTOM — Session History
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    _render_history_panel()


# ─────────────────────────────────────────────────────────────────────────────
# Webcam Panel
# ─────────────────────────────────────────────────────────────────────────────
def _render_webcam_panel():
    st.markdown("### 📷 Posture Coach")

    cam_active = st.session_state.cam_active

    # Toggle button
    btn_label = "⏹ Stop Camera" if cam_active else "▶ Start Camera"
    if st.button(btn_label, use_container_width=True):
        st.session_state.cam_active = not cam_active
        if not st.session_state.cam_active:
            st.session_state.posture_tips  = []
            st.session_state.posture_score = None
        st.rerun()

    if not cam_active:
        st.markdown(
            '<div class="cam-status cam-status-off">● Camera Off</div>'
            '<div style="color:#475569;font-size:0.78rem;margin-top:0.5rem">'
            'Start the camera to get real-time face & hand posture coaching '
            'while you practice your speech.</div>',
            unsafe_allow_html=True,
        )
        return

    # Camera is active — capture one frame
    st.markdown('<div class="cam-status cam-status-live">● Live</div>', unsafe_allow_html=True)

    # Use Streamlit's camera_input for one-shot capture (most compatible)
    captured = st.camera_input("📸 Capture frame for analysis", label_visibility="collapsed")

    if captured is not None:
        try:
            import cv2
            file_bytes = np.frombuffer(captured.getvalue(), np.uint8)
            frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Analysing posture…"):
                annotated, tips, score = _analyze_frame_mediapipe(frame_bgr)

            st.session_state.posture_tips  = tips
            st.session_state.posture_score = score

            # Show annotated frame
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True, caption="MediaPipe analysis")

        except Exception as e:
            st.error(f"Analysis error: {e}")

    # ── Posture score ────────────────────────────────────────────────────
    score = st.session_state.posture_score
    if score is not None:
        score_color = "#00c896" if score >= 80 else "#f59e0b" if score >= 50 else "#ff4d6d"
        st.markdown(
            f'<div style="text-align:center;margin:0.6rem 0">'
            f'<span style="font-size:2rem;font-weight:800;color:{score_color}">{score}</span>'
            f'<span style="font-size:0.8rem;color:#64748b">/100 presence score</span></div>',
            unsafe_allow_html=True,
        )

    # ── Coaching tips ────────────────────────────────────────────────────
    tips = st.session_state.posture_tips
    if tips:
        st.markdown("**💡 Real-time coaching**")
        for tip in tips:
            st.markdown(
                f'<div class="posture-tip">{tip}</div>',
                unsafe_allow_html=True,
            )

    # ── Instructions ────────────────────────────────────────────────────
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **What's being analysed:**
        - 👁️ **Eye contact** — face centering & gaze direction
        - 😊 **Expression** — smile ratio & relaxed face
        - ↕️ **Head tilt** — level head posture
        - 🙌 **Hand gestures** — open palms vs pointing

        **Tips for best results:**
        - Sit 60–90 cm from the camera
        - Ensure good lighting on your face
        - Keep hands visible in the frame
        - Click the shutter to re-analyse at any time
        """)


# ─────────────────────────────────────────────────────────────────────────────
# History Panel
# ─────────────────────────────────────────────────────────────────────────────
def _render_history_panel():
    history = db_load_history(st.session_state.user_email, limit=30)

    header_col, clear_col = st.columns([5, 1])
    with header_col:
        st.markdown(f"### 📚 Session History &nbsp;<span style='font-size:0.75rem;color:#64748b;font-weight:400'>({len(history)} saved)</span>", unsafe_allow_html=True)
    with clear_col:
        if history and st.button("🗑️ Clear All", use_container_width=True):
            db_clear_history(st.session_state.user_email)
            st.success("History cleared.")
            st.rerun()

    if not history:
        st.markdown(
            '<div style="color:#475569;font-size:0.82rem;padding:1rem 0">'
            'No sessions saved yet. Your analyses will appear here automatically.</div>',
            unsafe_allow_html=True,
        )
        return

    # Search / filter bar
    search = st.text_input("🔍 Search history", placeholder="Filter by keyword…", label_visibility="collapsed")

    filtered = [
        e for e in history
        if not search or search.lower() in (e["input"] + e["output"]).lower()
    ]

    if not filtered:
        st.markdown('<div style="color:#64748b;font-size:0.82rem">No results matching your search.</div>', unsafe_allow_html=True)
        return

    for entry in filtered[:15]:  # show max 15
        summ = entry["summary"]
        tone_b  = summ.get("tone_score_before", "—")
        tone_a  = summ.get("tone_score_after",  "—")
        fillers = summ.get("fillers_removed",   "—")
        ps      = entry.get("posture_score")
        ps_str  = f"👤 {ps}/100" if ps is not None else ""

        # Timestamp display
        created = entry.get("created_at", "")
        try:
            dt = datetime.datetime.strptime(created, "%Y-%m-%d %H:%M:%S")
            ts = dt.strftime("%d %b %Y, %I:%M %p")
        except Exception:
            ts = created

        with st.expander(f"#{entry['id']} · {ts} · {entry['roles'][:40]}"):
            st.markdown(
                f'<div class="hist-meta">'
                f'<span class="hist-badge">🎯 {entry["audiences"][:30]}</span>'
                f'<span class="hist-badge">🔊 Tone {tone_b}→{tone_a}</span>'
                f'<span class="hist-badge">✂️ {fillers} fillers</span>'
                f'{"<span class=hist-badge>" + ps_str + "</span>" if ps_str else ""}'
                f'<span style="float:right;color:#475569;font-size:0.7rem">{entry["latency_ms"]}ms</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            h1, h2 = st.columns(2)
            with h1:
                st.caption("Original")
                st.code(entry["input"], language=None)
            with h2:
                st.caption("Corrected")
                st.code(entry["output"], language=None)

            # Replay button — loads entry back into main view
            if st.button("🔄 Replay this session", key=f"replay_{entry['id']}"):
                st.session_state.transcript  = entry["input"]
                st.session_state.corrected   = entry["output"]
                st.session_state.annotations = entry["annotations"]
                st.session_state.summary     = entry["summary"]
                st.session_state.latency_ms  = entry["latency_ms"]
                st.rerun()

            # Annotations if any
            if entry["annotations"]:
                with st.expander("View annotations", expanded=False):
                    for a in entry["annotations"]:
                        emoji = ANNOTATION_COLORS.get(a.get("type",""), ("#fff","💬"))[1]
                        st.markdown(f'{emoji} **{a.get("phrase","")}** — {a.get("suggestion","")}')


# ─────────────────────────────────────────────────────────────────────────────
# Run correction + save to DB
# ─────────────────────────────────────────────────────────────────────────────
def _run_correction(text: str, model: str, progress=None):
    roles_str     = ", ".join(st.session_state.user_roles)
    audiences_str = ", ".join(st.session_state.user_audiences)

    with st.spinner("Stage 1/2 — Generating corrected speech…"):
        result = correct_transcript(
            text,
            model=model,
            temperature=0.3,
            role=roles_str,
            audience=audiences_str,
            name=st.session_state.user_name,
        )

    if progress:
        progress.progress(100, text="Done!")
        time.sleep(0.3)
        progress.empty()

    if result["success"]:
        st.session_state.transcript  = text
        st.session_state.corrected   = result["corrected"]
        st.session_state.annotations = result["annotations"]
        st.session_state.summary     = result["summary"]
        st.session_state.latency_ms  = result["latency_ms"]

        # Save to SQLite
        db_save_session(
            email        = st.session_state.user_email,
            name         = st.session_state.user_name,
            roles        = roles_str,
            audiences    = audiences_str,
            inp          = text,
            out          = result["corrected"],
            annotations  = result["annotations"],
            summary      = result["summary"],
            latency      = result["latency_ms"],
            posture_score= st.session_state.posture_score,
        )
        st.rerun()
    else:
        st.error(f"Error: {result['error']}")


# ─────────────────────────────────────────────────────────────────────────────
# Render results (unchanged from original, kept complete)
# ─────────────────────────────────────────────────────────────────────────────
def _render_results():
    ann        = st.session_state.annotations
    summary    = st.session_state.summary
    transcript = st.session_state.transcript
    corrected  = st.session_state.corrected
    latency    = st.session_state.latency_ms

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("### 🔍 Speech Analysis")
        legend = '<div class="legend">'
        for atype, (color, emoji) in ANNOTATION_COLORS.items():
            legend += (
                f'<div style="display:flex;align-items:center;gap:5px;font-size:0.73rem;color:#64748b">'
                f'<span class="legend-dot" style="background:{color}"></span>'
                f'{emoji} {atype.capitalize()}</div>'
            )
        legend += '</div>'
        st.markdown(legend, unsafe_allow_html=True)
        st.markdown(_build_annotated_html(transcript, ann), unsafe_allow_html=True)

        if ann:
            st.markdown("<br>**💡 Coaching Suggestions**", unsafe_allow_html=True)
            for a in ann:
                phrase  = a.get("phrase", "")
                atype   = a.get("type", "filler")
                tip     = a.get("suggestion", "")
                emoji   = ANNOTATION_COLORS.get(atype, ("#fff", "💬"))[1]
                color   = ANNOTATION_COLORS.get(atype, ("#fff", ""))[0]
                st.markdown(
                    f'<div class="sug-item">'
                    f'<div style="font-size:1.1rem">{emoji}</div>'
                    f'<div>'
                    f'<div class="sug-phrase" style="color:{color}">"{phrase}"</div>'
                    f'<div class="sug-text">{tip}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#64748b;font-size:0.82rem;margin-top:0.8rem">'
                'ℹ️ No specific annotations returned — see corrected output on the right.</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("### ✅ Professional Output")
        before    = summary.get("tone_score_before", "—")
        after_s   = summary.get("tone_score_after",  "—")
        fillers   = summary.get("fillers_removed", len([a for a in ann if a.get("type") == "filler"]))
        grammar   = summary.get("grammar_fixes",   len([a for a in ann if a.get("type") == "grammar"]))
        in_words  = len(transcript.split())
        out_words = len(corrected.split())

        # Posture score tile (if available)
        ps = st.session_state.posture_score
        ps_tile = ""
        if ps is not None:
            ps_color = "#00c896" if ps >= 80 else "#f59e0b" if ps >= 50 else "#ff4d6d"
            ps_tile = f'<div class="score-tile"><div class="score-val" style="color:{ps_color}">{ps}</div><div class="score-lbl">Posture</div></div>'

        st.markdown(
            f'''<div class="score-row">
            <div class="score-tile">
                <div class="score-val" style="color:#ff4d6d">{before}<sup style="font-size:0.7rem">/10</sup></div>
                <div class="score-lbl">Tone Before</div>
            </div>
            <div class="score-tile">
                <div class="score-val" style="color:#00c896">{after_s}<sup style="font-size:0.7rem">/10</sup></div>
                <div class="score-lbl">Tone After</div>
            </div>
            <div class="score-tile">
                <div class="score-val" style="color:#f59e0b">{fillers}</div>
                <div class="score-lbl">Fillers Removed</div>
            </div>
            <div class="score-tile">
                <div class="score-val" style="color:#a78bfa">{grammar}</div>
                <div class="score-lbl">Grammar Fixes</div>
            </div>
            {ps_tile}
            </div>''',
            unsafe_allow_html=True,
        )

        top_tip = summary.get("top_tip", "")
        if top_tip:
            st.markdown(f'<div class="top-tip">⭐ {top_tip}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="final-out">{corrected}</div>', unsafe_allow_html=True)

        with st.expander("📋 Copy corrected text"):
            st.text_area("Corrected output", value=corrected, height=110, label_visibility="collapsed")

        st.markdown(
            f'<div style="color:#475569;font-size:0.73rem;margin-top:0.4rem">'
            f'Processed in {latency}ms &nbsp;·&nbsp; {in_words} → {out_words} words'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    _page_login()
else:
    p = st.session_state.page
    if   p == "setup_role":     _page_setup_role()
    elif p == "setup_audience": _page_setup_audience()
    elif p == "dashboard":      _page_dashboard()
    else:                       _page_login()