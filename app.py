"""
ScamShield - demo app
-----------------------
1. Make sure scamshield_model.joblib and scamshield_vectorizer.joblib
   (created by train_scamshield.py) are in this SAME folder.
2. Run:  streamlit run app.py   (or: python -m streamlit run app.py)
   Missing packages install themselves automatically.
"""

import subprocess
import sys
import os


def ensure_installed(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


for pkg in ["streamlit", "scikit-learn", "joblib"]:
    ensure_installed(pkg, "sklearn" if pkg == "scikit-learn" else pkg)

import streamlit as st
import joblib
import numpy as np
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "scamshield_model.joblib")
VEC_PATH = os.path.join(SCRIPT_DIR, "scamshield_vectorizer.joblib")

st.set_page_config(page_title="ScamShield", page_icon="🛡️", layout="centered")

if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
    st.error(
        "Couldn't find scamshield_model.joblib / scamshield_vectorizer.joblib "
        "in this folder. Run train_scamshield.py first."
    )
    st.stop()

clf = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

CLASSES = list(clf.classes_)
SCAM_IDX = CLASSES.index("scam") if "scam" in CLASSES else 1
# sign flips coefficient interpretation if 'scam' isn't classes_[1]
COEF_SIGN = 1 if SCAM_IDX == 1 else -1

# Classic advance-fee / 419-style fraud patterns a small generic
# SMS-spam training set usually doesn't cover well. Any match here
# forces the risk score to the "scam" band.
RED_FLAG_PATTERNS = [
    (r"\bpin\s*[:\-]?\s*\d{3,}", "contains a raw PIN number"),
    (r"\bacc(?:ount)?\.?\s*(?:no\.?|number)?\s*[:\-]?\s*\d{5,}", "contains a raw account number"),
    (r"\bwithdraw\b", "asks you to withdraw a sum of money"),
    (r"\bconsignment\b|\bcontainers?\b", "mentions clearing a consignment/containers"),
    (r"\bnext of kin\b", "uses \"next of kin\" (classic advance-fee scam phrase)"),
    (r"\bdiplomat(ic)?\b", "mentions a diplomatic shipment/bag (classic scam script)"),
    (r"\bbeneficiary\b", "refers to you as a \"beneficiary\""),
    (r"\bcustoms?\b.{0,20}\bclear", "mentions clearing something through customs"),
    (r"\bprocessing fee\b|\bclearance fee\b|\bactivation fee\b|\btransfer fee\b", "asks for an upfront fee"),
    (r"\burgent(ly)?\b.{0,20}\b(business|assistance|proposal)\b", "urgency + business proposal language"),
    (r"\binherit(ance)?\b", "mentions an unexpected inheritance"),
    (r"\bregistration bonus\b|\bshare (your |)user id\b", "unsolicited bonus + share your ID pattern"),
]


def check_red_flags(message: str):
    lowered = message.lower()
    return [reason for pattern, reason in RED_FLAG_PATTERNS if re.search(pattern, lowered)]


def risk_score(message: str):
    """Returns (risk 0-1 = probability of scam, sparse tfidf vector)."""
    vec = vectorizer.transform([message])
    proba = clf.predict_proba(vec)[0]
    return float(proba[SCAM_IDX]), vec


def classify(risk: float):
    if risk >= 0.70:
        return "scam"
    elif risk < 0.50:
        return "safe"
    else:
        return "verify"


def explain(vec, n=5):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0] * COEF_SIGN
    nonzero = vec.nonzero()[1]
    if len(nonzero) == 0:
        return []
    contributions = vec[0, nonzero].toarray().flatten() * coefs[nonzero]
    order = np.argsort(-np.abs(contributions))[:n]
    return [(feature_names[nonzero][i], contributions[i]) for i in order]


VERDICT_INFO = {
    "scam": {
        "word": "Scam",
        "advice": "Strong fraud indicators found. Do not click links, share codes, or send money.",
    },
    "verify": {
        "word": "Verify",
        "advice": "Risk is inconclusive. Call your service provider or bank directly, using their official number, before acting.",
    },
    "safe": {
        "word": "Safe",
        "advice": "No strong scam indicators found. Still stay cautious with unexpected requests for money or codes.",
    },
}

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

THEMES = {
    "dark": {
        "bg": "#0B0F1A", "surface": "#121826", "surface_alt": "#182035", "border": "#26304A",
        "text": "#E7EBF3", "text_muted": "#8C96AC",
        "accent": "#38BDF8", "accent_ink": "#06121C",
        "safe": "#34D399", "safe_soft": "rgba(52,211,153,0.14)",
        "danger": "#F87171", "danger_soft": "rgba(248,113,113,0.14)",
        "verify": "#FBBF24", "verify_soft": "rgba(251,191,36,0.14)",
    },
    "light": {
        "bg": "#F6F7FB", "surface": "#FFFFFF", "surface_alt": "#EEF1F6", "border": "#DCE1EC",
        "text": "#10162A", "text_muted": "#5B6478",
        "accent": "#0284C7", "accent_ink": "#FFFFFF",
        "safe": "#059669", "safe_soft": "rgba(5,150,105,0.10)",
        "danger": "#DC2626", "danger_soft": "rgba(220,38,38,0.10)",
        "verify": "#B45309", "verify_soft": "rgba(180,83,9,0.10)",
    },
}
t = THEMES[st.session_state.theme]

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">

<style>
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"] {{
    visibility: hidden; height: 0;
}}
[data-testid="stAppViewContainer"], .stApp {{ background: {t['bg']}; }}
.block-container {{ max-width: 640px; padding-top: 2.5rem; padding-bottom: 3rem; }}
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; color: {t['text']}; }}

.sc-topbar {{ display: flex; justify-content: space-between; align-items: center; }}
.sc-brand {{ display: flex; align-items: center; gap: 0.6rem; }}
.sc-brand h1 {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 600;
    font-size: 1.55rem; letter-spacing: -0.01em; margin: 0; color: {t['text']};
}}
.sc-subtitle {{
    color: {t['text_muted']}; font-size: 0.95rem;
    margin: 0.9rem 0 1.6rem 0; padding-bottom: 1.4rem;
    border-bottom: 1px solid {t['border']};
}}
.sc-field-label {{ font-size: 0.85rem; color: {t['text_muted']}; margin-bottom: 0.4rem; }}

[data-testid="stTextArea"] textarea {{
    background: {t['surface_alt']} !important; border: 1px solid {t['border']} !important;
    border-radius: 8px !important; color: {t['text']} !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.95rem !important;
}}
[data-testid="stTextArea"] textarea:focus {{
    border-color: {t['accent']} !important; box-shadow: 0 0 0 1px {t['accent']} !important;
}}
[data-testid="stTextArea"] label {{ display: none; }}

[data-testid="stButton"] button {{
    border-radius: 8px !important; font-weight: 600 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    transition: opacity 0.15s ease;
}}
div[data-testid="column"]:first-of-type [data-testid="stButton"] button {{
    background: {t['accent']} !important; color: {t['accent_ink']} !important;
    border: none !important; padding: 0.55rem 1.4rem !important;
}}
div[data-testid="column"]:last-of-type [data-testid="stButton"] button {{
    background: transparent !important; color: {t['text_muted']} !important;
    border: 1px solid {t['border']} !important; padding: 0.5rem 0.75rem !important;
}}
[data-testid="stButton"] button:hover {{ opacity: 0.8; }}

.sc-verdict {{
    border-radius: 6px; border-left: 4px solid var(--vc);
    background: var(--vs); padding: 1.1rem 1.3rem; margin: 1.6rem 0 0.4rem 0;
}}
.sc-verdict-label {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; color: var(--vc); margin: 0 0 0.15rem 0; }}
.sc-verdict-risk {{ font-size: 0.85rem; color: {t['text_muted']}; margin: 0 0 0.6rem 0; }}
.sc-verdict-advice {{ font-size: 0.88rem; color: {t['text']}; margin: 0; line-height: 1.45; }}

.sc-meter {{ position: relative; margin: 1.4rem 0 0.3rem 0; }}
.sc-meter-track {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; }}
.sc-meter-marker {{
    position: absolute; top: -4px; width: 2px; height: 16px;
    background: {t['text']}; border-radius: 1px; transform: translateX(-50%);
}}
.sc-meter-labels {{ display: flex; justify-content: space-between; font-size: 0.72rem; color: {t['text_muted']}; margin-top: 0.4rem; }}

.sc-why-label {{ font-size: 0.85rem; color: {t['text_muted']}; margin: 1.4rem 0 0.6rem 0; }}
.sc-reason-row {{
    display: flex; align-items: flex-start; gap: 0.55rem; padding: 0.4rem 0;
    font-size: 0.9rem; border-bottom: 1px solid {t['border']};
}}
.sc-reason-row:last-child {{ border-bottom: none; }}
.sc-dot {{ width: 6px; height: 6px; border-radius: 50%; margin-top: 0.5rem; flex-shrink: 0; background: var(--dc); }}
.sc-reason-text {{ color: {t['text']}; }}
.sc-tag {{ font-size: 0.78rem; padding: 0.1rem 0.5rem; border-radius: 4px; font-weight: 500; margin-right: 0.4rem; flex-shrink: 0; }}
.sc-tag-scam {{ background: {t['danger_soft']}; color: {t['danger']}; }}
.sc-tag-safe {{ background: {t['safe_soft']}; color: {t['safe']}; }}

.sc-footer {{ margin-top: 2.5rem; padding-top: 1.2rem; border-top: 1px solid {t['border']}; color: {t['text_muted']}; font-size: 0.8rem; }}

[data-testid="stExpander"] {{
    border: 1px solid {t['border']} !important;
    border-radius: 8px !important;
    background: {t['surface']} !important;
    margin-top: 1.8rem;
}}
[data-testid="stExpander"] summary {{
    color: {t['text']} !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
}}
.sc-tip-row {{ display: flex; align-items: flex-start; gap: 0.6rem; padding: 0.5rem 0; font-size: 0.9rem; border-bottom: 1px solid {t['border']}; }}
.sc-tip-row:last-child {{ border-bottom: none; }}
.sc-tip-num {{ color: {t['accent']}; font-family: 'Space Grotesk', sans-serif; font-weight: 600; flex-shrink: 0; }}
.sc-tip-text {{ color: {t['text']}; line-height: 1.45; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_brand, col_toggle = st.columns([5, 1])
with col_brand:
    st.markdown(f"""
    <div class="sc-brand">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
      <path d="M12 2L4 5v6c0 5 3.4 8.7 8 10 4.6-1.3 8-5 8-10V5l-8-3z" stroke="{t['accent']}" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M9 12l2 2 4-4" stroke="{t['accent']}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <h1>ScamShield</h1>
    </div>
    """, unsafe_allow_html=True)
with col_toggle:
    icon = "☀️" if st.session_state.theme == "dark" else "🌙"
    if st.button(icon, key="theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

st.markdown('<p class="sc-subtitle">Paste a message below to check whether it\'s a scam.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
st.markdown('<div class="sc-field-label">Message</div>', unsafe_allow_html=True)
message = st.text_area("Message", height=120, placeholder="Paste an SMS, WhatsApp, or email message...", label_visibility="collapsed")
check = st.button("Check message", type="primary")

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
if check:
    if not message.strip():
        st.warning("Type or paste a message first.")
    else:
        risk, vec = risk_score(message)
        red_flags = check_red_flags(message)
        if red_flags:
            risk = max(risk, 0.97)

        label = classify(risk)
        info = VERDICT_INFO[label]
        color_key = {"scam": "danger", "verify": "verify", "safe": "safe"}[label]
        vc, vs = t[color_key], t[f"{color_key}_soft"] if color_key != "verify" else t["verify_soft"]

        st.markdown(f"""
        <div class="sc-verdict" style="--vc:{vc}; --vs:{vs};">
            <p class="sc-verdict-label">{info['word']}</p>
            <p class="sc-verdict-risk">Risk score: {risk*100:.1f}%</p>
            <p class="sc-verdict-advice">{info['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

        marker_pos = min(max(risk * 100, 1), 99)
        st.markdown(f"""
        <div class="sc-meter">
            <div class="sc-meter-track">
                <div style="width:50%; background:{t['safe']};"></div>
                <div style="width:20%; background:{t['verify']};"></div>
                <div style="width:30%; background:{t['danger']};"></div>
            </div>
            <div class="sc-meter-marker" style="left:{marker_pos}%;"></div>
            <div class="sc-meter-labels"><span>Safe</span><span>Verify</span><span>Scam</span></div>
        </div>
        """, unsafe_allow_html=True)

        words = explain(vec)
        if red_flags or words:
            rows = ""
            for reason in red_flags:
                rows += f'<div class="sc-reason-row"><span class="sc-dot" style="--dc:{t["danger"]};"></span><span class="sc-reason-text">{reason}</span></div>'
            for word, weight in words:
                tag_class = "sc-tag-scam" if weight > 0 else "sc-tag-safe"
                tag_word = "scam signal" if weight > 0 else "safe signal"
                rows += f'<div class="sc-reason-row"><span class="sc-tag {tag_class}">{tag_word}</span><span class="sc-reason-text">"{word}"</span></div>'
            st.markdown(f'<p class="sc-why-label">Why</p>{rows}', unsafe_allow_html=True)

st.markdown('<div class="sc-footer">ScamShield &middot; trained on labeled scam/safe message data</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tips section
# ---------------------------------------------------------------------------
TIPS = [
    "Never share your M-Pesa PIN, PIN reset code, or one-time password (OTP) with anyone - not even someone claiming to be from Safaricom, your bank, or a delivery company.",
    "No legitimate company asks you to pay a \"fee\" to receive money, a prize, or a refund. If a message asks you to pay first to get a bigger payout, it's a scam.",
    "Verify any payment or bonus notification directly in your own M-Pesa app or bank app - never through a link inside the message itself.",
    "Be suspicious of urgency: \"act now,\" \"your account will be suspended,\" \"claim before it expires.\" Scammers rely on you not stopping to think.",
    "If a message tells you to call a number, don't use the number in the message. Look up the company's official customer care number yourself.",
    "Common patterns to watch for: fake prize or refund alerts, advance-fee schemes (\"send a small amount to receive a much bigger amount\"), job offers that need a registration fee, and messages using raw account/PIN numbers in the text itself.",
]

with st.expander("How to avoid getting scammed"):
    rows = "".join(
        f'<div class="sc-tip-row"><span class="sc-tip-num">{i+1}</span><span class="sc-tip-text">{tip}</span></div>'
        for i, tip in enumerate(TIPS)
    )
    st.markdown(rows, unsafe_allow_html=True)
