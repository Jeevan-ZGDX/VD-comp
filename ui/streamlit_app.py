"""
streamlit_app.py  —  VDart SpeechPro
Flow: Login → Role (multi-select) → Audience (multi-select) → Dashboard
"""

import sys, time
from pathlib import Path

import streamlit as st

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
# Data
# ─────────────────────────────────────────────────────────────────────────────
USERS = {
    "admin@vdart.com":     {"password": "vdart123",  "name": "Admin User"},
    "recruiter@vdart.com": {"password": "recruit1",  "name": "Priya Sharma"},
    "sales@vdart.com":     {"password": "sales2024", "name": "Arjun Mehta"},
    "demo@vdart.com":      {"password": "demo",      "name": "Demo User"},
}

ROLES = [
    ("👤", "Recruiter"),
    ("💼", "Sales Executive"),
    ("🤝", "Account Manager"),
    ("🛠️", "Technical Consultant"),
    ("📋", "Project Manager"),
    ("🧑‍💼", "HR Business Partner"),
    ("🚚", "Delivery Manager"),
    ("🌐", "Client Partner"),
    ("📊", "Business Analyst"),
    ("🏆", "Team Lead"),
]

AUDIENCES = [
    ("🎓", "Job Candidate"),
    ("👔", "C-Suite / Executive"),
    ("💻", "Client (Technical)"),
    ("📢", "Client (Non-Technical)"),
    ("👥", "Internal Team"),
    ("🏢", "Hiring Manager"),
    ("🤝", "Vendor / Partner"),
    ("💰", "Investor"),
    ("🌱", "New Employee / Trainee"),
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

/* ── Selected count badge ── */
.sel-badge {
    display:inline-block; background:rgba(0,200,150,0.15);
    color:var(--accent); border:1px solid var(--accent);
    border-radius:99px; font-size:0.75rem; font-weight:700;
    padding:2px 10px; margin-left:8px;
}

/* ── Topbar ── */
.topbar {
    background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:0.7rem 1.4rem;
    display:flex; align-items:center; gap:1rem; margin-bottom:1.2rem;
}
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
.score-row { display:flex; gap:0.8rem; margin:0.8rem 0 1rem; }
.score-tile {
    flex:1; background:var(--surface); border:1px solid var(--border);
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
    "user_roles":     [],      # list — multi-select
    "user_audiences": [],      # list — multi-select
    "transcript":     "",
    "corrected":      "",
    "annotations":    [],
    "summary":        {},
    "latency_ms":     0,
    "history":        [],
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
# PAGE 2 — ROLE SETUP  (multi-select checkboxes)
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
            'style across your selected roles.</div>',
            unsafe_allow_html=True,
        )

        selected_roles = list(st.session_state.user_roles)

        # 2-column checkbox grid
        col_a, col_b = st.columns(2, gap="small")
        for i, (icon, label) in enumerate(ROLES):
            col = col_a if i % 2 == 0 else col_b
            with col:
                checked = st.checkbox(
                    f"{icon}  {label}",
                    value=(label in selected_roles),
                    key=f"role_cb_{label}",
                )
                if checked and label not in selected_roles:
                    selected_roles.append(label)
                elif not checked and label in selected_roles:
                    selected_roles.remove(label)

        # Live count badge
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
# PAGE 3 — AUDIENCE SETUP  (multi-select checkboxes)
# ─────────────────────────────────────────────────────────────────────────────
def _page_setup_audience():
    _, mid, _ = st.columns([0.5, 3, 0.5])
    with mid:
        st.markdown('<div class="step-pill">Step 2 of 2</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Who do you speak to?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-sub">Select all your typical audiences — SpeechPro adapts '
            'tone, formality, and vocabulary accordingly.</div>',
            unsafe_allow_html=True,
        )

        selected_aud = list(st.session_state.user_audiences)

        col_a, col_b = st.columns(2, gap="small")
        for i, (icon, label) in enumerate(AUDIENCES):
            col = col_a if i % 2 == 0 else col_b
            with col:
                checked = st.checkbox(
                    f"{icon}  {label}",
                    value=(label in selected_aud),
                    key=f"aud_cb_{label}",
                )
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
            f'</div>',
            unsafe_allow_html=True,
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

    # ── Input tabs ───────────────────────────────────────────────────────────
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

    # ── Results ──────────────────────────────────────────────────────────────
    if st.session_state.corrected:
        st.markdown("---")
        _render_results()

    # ── History ──────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("---")
        with st.expander(f"📚 Session History  ({len(st.session_state.history)} analyses)"):
            for i, e in enumerate(reversed(st.session_state.history)):
                n = len(st.session_state.history) - i
                st.markdown(
                    f"**#{n}** &nbsp; _{e.get('roles','')}_  →  _{e.get('audiences','')}_"
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Input"); st.code(e["input"], language=None)
                with c2:
                    st.caption("Output"); st.code(e["output"], language=None)
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()


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
        st.session_state.history.append({
            "input":     text,
            "output":    result["corrected"],
            "roles":     roles_str,
            "audiences": audiences_str,
        })
        st.rerun()
    else:
        st.error(f"Error: {result['error']}")


def _render_results():
    ann        = st.session_state.annotations
    summary    = st.session_state.summary
    transcript = st.session_state.transcript
    corrected  = st.session_state.corrected
    latency    = st.session_state.latency_ms

    left, right = st.columns([1.05, 1], gap="large")

    # ── LEFT panel ───────────────────────────────────────────────────────────
    with left:
        st.markdown("### 🔍 Speech Analysis")

        # Legend
        legend = '<div class="legend">'
        for atype, (color, emoji) in ANNOTATION_COLORS.items():
            legend += (
                f'<div style="display:flex;align-items:center;gap:5px;font-size:0.73rem;color:#64748b">'
                f'<span class="legend-dot" style="background:{color}"></span>'
                f'{emoji} {atype.capitalize()}</div>'
            )
        legend += '</div>'
        st.markdown(legend, unsafe_allow_html=True)

        # Annotated transcript
        st.markdown(_build_annotated_html(transcript, ann), unsafe_allow_html=True)

        # Coaching suggestions
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

    # ── RIGHT panel ──────────────────────────────────────────────────────────
    with right:
        st.markdown("### ✅ Professional Output")

        # Scores
        before    = summary.get("tone_score_before", "—")
        after_s   = summary.get("tone_score_after",  "—")
        fillers   = summary.get("fillers_removed", len([a for a in ann if a.get("type") == "filler"]))
        grammar   = summary.get("grammar_fixes",   len([a for a in ann if a.get("type") == "grammar"]))
        in_words  = len(transcript.split())
        out_words = len(corrected.split())

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
            </div>''',
            unsafe_allow_html=True,
        )

        top_tip = summary.get("top_tip", "")
        if top_tip:
            st.markdown(
                f'<div class="top-tip">⭐ {top_tip}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="final-out">{corrected}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("📋 Copy corrected text"):
            st.text_area(
                "Corrected output",
                value=corrected,
                height=110,
                label_visibility="collapsed",
            )

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