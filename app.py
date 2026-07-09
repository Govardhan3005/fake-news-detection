"""
============================================================
app.py — Fake News Detection using Machine Learning
============================================================
Advanced ML-powered news verification platform
Sky Blue + White corporate theme
Real-time text analytics, sentiment, readability scoring
3 Tabs: Text Analysis | Image OCR | URL Analysis
============================================================
"""

import os, re, time, math, warnings
warnings.filterwarnings("ignore")

import joblib
import streamlit as st
from PIL import Image

try:
    from preprocess import preprocess_text
except Exception:
    def preprocess_text(t): return t.lower()

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection using Machine Learning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Sky Blue + White Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #0f172a !important;
}
.stApp { background: #ffffff !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"]        { display: none; }
[data-testid="collapsedControl"] { display: none; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f0f9ff; }
::-webkit-scrollbar-thumb { background: #7dd3fc; border-radius: 99px; }

.block-container { padding: 1rem 2.5rem 2rem !important; max-width: 1240px !important; }

/* ═══════ HEADER ═══════ */
.header {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8, #7dd3fc);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 4px 24px rgba(14,165,233,.18);
}
.header-left { display: flex; align-items: center; gap: 16px; }
.header-icon {
    width: 52px; height: 52px; background: rgba(255,255,255,.2);
    backdrop-filter: blur(10px); border-radius: 14px; border: 1px solid rgba(255,255,255,.3);
    display: flex; align-items: center; justify-content: center; font-size: 1.5rem;
}
.header-title {
    font-size: 1.2rem; font-weight: 800; color: #fff;
    letter-spacing: -.3px; text-shadow: 0 1px 2px rgba(0,0,0,.1);
}
.header-sub { font-size: .72rem; color: rgba(255,255,255,.85); margin-top: 2px; font-weight: 400; }
.header-badges { display: flex; gap: 8px; align-items: center; }
.h-badge {
    font-size: .68rem; font-weight: 600; padding: 5px 14px; border-radius: 99px;
    backdrop-filter: blur(8px); letter-spacing: .3px;
}
.hb-ml { background: rgba(255,255,255,.2); color: #fff; border: 1px solid rgba(255,255,255,.3); }
.hb-live { background: rgba(16,185,129,.2); color: #ecfdf5; border: 1px solid rgba(16,185,129,.4); }
.hb-demo { background: rgba(251,191,36,.2); color: #fffbeb; border: 1px solid rgba(251,191,36,.4); }

/* ═══════ KPI CARDS ═══════ */
.kpi-row { display: flex; gap: 12px; margin-bottom: 22px; }
.kpi-card {
    flex: 1; background: #f0f9ff; border: 1px solid #e0f2fe;
    border-radius: 12px; padding: 16px 18px; text-align: center;
    transition: all .2s ease; cursor: default;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(14,165,233,.1); }
.kpi-val { font-size: 1.7rem; font-weight: 900; color: #0f172a; }
.kpi-lbl { font-size: .6rem; font-weight: 700; text-transform: uppercase;
           letter-spacing: 1px; color: #64748b; margin-top: 3px; }

/* ═══════ TABS ═══════ */
button[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #e2e8f0 !important;
    margin-bottom: 22px !important;
    padding-bottom: 0px !important;
    gap: 8px !important;
}
button[data-baseweb="tab"] {
    font-size: .85rem !important;
    font-weight: 600 !important; 
    color: #475569 !important;
    background: transparent !important; 
    border: none !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 24px !important; 
    transition: all .2s ease !important;
}
/* Force text color on inner elements */
button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
    color: #475569 !important;
}
button[data-baseweb="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] p {
    background: #0066cc !important; 
    color: #ffffff !important; 
    font-weight: 700 !important;
    border-radius: 6px 6px 0 0 !important;
}
button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: #f1f5f9 !important; 
    color: #1e293b !important;
}
button[data-baseweb="tab"]:hover:not([aria-selected="true"]) p {
    color: #1e293b !important;
}

/* ═══════ FORM CONTROLS ═══════ */
.stTextArea textarea {
    background: #f8fafc !important; border: 1.5px solid #e0f2fe !important;
    border-radius: 10px !important; color: #0f172a !important;
    font-family: 'Inter', sans-serif !important; font-size: .87rem !important;
    line-height: 1.7 !important; resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,.15) !important;
}
.stTextArea textarea::placeholder { color: #94a3b8 !important; }

.stTextInput input {
    background: #f8fafc !important; border: 1.5px solid #e0f2fe !important;
    border-radius: 10px !important; color: #0f172a !important; font-size: .87rem !important;
    padding: 10px 14px !important;
}
.stTextInput input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,.15) !important;
}
.stTextInput input::placeholder { color: #94a3b8 !important; }

/* ═══════ BUTTONS ═══════ */
.stButton > button {
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: .84rem !important; border-radius: 10px !important;
    transition: all .2s ease !important; height: 44px !important;
}
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8) !important;
    border: none !important; color: #fff !important;
    box-shadow: 0 2px 8px rgba(14,165,233,.25) !important;
}
[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(14,165,233,.3) !important;
}
[data-testid="baseButton-secondary"] {
    background: #f0f9ff !important; border: 1.5px solid #e0f2fe !important; color: #0284c7 !important;
}
[data-testid="baseButton-secondary"]:hover {
    background: #e0f2fe !important; border-color: #7dd3fc !important;
}

/* ═══════ FILE UPLOADER ═══════ */
[data-testid="stFileUploader"] {
    background: #f0f9ff !important; border: 2px dashed #bae6fd !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #38bdf8 !important; }

/* ═══════ SELECTBOX ═══════ */
[data-testid="stSelectbox"] > div > div {
    background: #f8fafc !important; border: 1.5px solid #e0f2fe !important;
    color: #0f172a !important; border-radius: 10px !important;
}

/* ═══════ RESULT CARD ═══════ */
.result-card { border-radius: 14px; padding: 22px 26px; margin-top: 16px; animation: pop .4s ease; }
@keyframes pop {
    from { opacity:0; transform:translateY(12px) scale(.98); }
    to   { opacity:1; transform:translateY(0) scale(1); }
}
.card-real { background: linear-gradient(135deg, #ecfdf5, #f0fdf4); border: 1.5px solid #a7f3d0; }
.card-fake { background: linear-gradient(135deg, #fef2f2, #fff5f5); border: 1.5px solid #fecaca; }
.result-verdict { font-size: 1.4rem; font-weight: 900; letter-spacing: .5px; margin-bottom: 6px; }
.verdict-real { color: #059669; }
.verdict-fake { color: #dc2626; }
.result-desc { font-size: .82rem; color: #475569; line-height: 1.6; margin-bottom: 14px; }
.conf-wrap { margin-bottom: 16px; }
.conf-label { display:flex; justify-content:space-between; font-size:.7rem; color:#64748b;
              margin-bottom:6px; font-weight:600; text-transform:uppercase; letter-spacing:.5px; }
.conf-track { background: #e2e8f0; border-radius: 99px; height: 8px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 99px; transition: width 1s cubic-bezier(.4,0,.2,1); }
.fill-real  { background: linear-gradient(90deg, #10b981, #6ee7b7); }
.fill-fake  { background: linear-gradient(90deg, #ef4444, #fca5a5); }
.prob-row   { display: flex; gap: 8px; }
.prob-pill  { flex:1; background:#fff; border:1.5px solid #e2e8f0;
              border-radius:10px; padding:9px 10px; text-align:center;
              transition: transform .15s; }
.prob-pill:hover { transform: translateY(-1px); }
.prob-pill .pl { font-size:.58rem; color:#64748b; text-transform:uppercase;
                 letter-spacing:.6px; font-weight:700; }
.prob-pill .pv { font-size:.92rem; font-weight:800; margin-top:2px; }
.pv-real{color:#059669}.pv-fake{color:#dc2626}.pv-mode{color:#0ea5e9;font-size:.72rem!important}

/* ═══════ ANALYTICS PANEL ═══════ */
.analytics-panel {
    background: #f0f9ff; border: 1.5px solid #e0f2fe; border-radius: 14px;
    padding: 20px 22px; margin-top: 16px;
}
.analytics-title {
    font-size: .72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #0284c7; margin-bottom: 14px;
    display: flex; align-items: center; gap: 6px;
}
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px; }
.metric-item {
    background: #fff; border: 1px solid #e0f2fe; border-radius: 10px;
    padding: 10px 14px; text-align: center; transition: all .15s;
}
.metric-item:hover { border-color: #7dd3fc; box-shadow: 0 2px 8px rgba(14,165,233,.08); }
.mi-val { font-size: 1.1rem; font-weight: 800; color: #0f172a; }
.mi-lbl { font-size: .58rem; font-weight: 600; text-transform: uppercase;
          letter-spacing: .5px; color: #64748b; margin-top: 2px; }

/* signal bars */
.signal-row { display: flex; flex-direction: column; gap: 8px; }
.signal-item { display: flex; align-items: center; gap: 10px; }
.signal-label { font-size: .72rem; font-weight: 500; color: #475569; width: 120px; flex-shrink: 0; }
.signal-bar-track { flex: 1; background: #e2e8f0; border-radius: 99px; height: 6px; overflow: hidden; }
.signal-bar-fill { height: 100%; border-radius: 99px; transition: width .8s ease; }
.sb-low  { background: linear-gradient(90deg, #10b981, #6ee7b7); }
.sb-mid  { background: linear-gradient(90deg, #f59e0b, #fcd34d); }
.sb-high { background: linear-gradient(90deg, #ef4444, #fca5a5); }
.signal-val { font-size: .7rem; font-weight: 700; width: 36px; text-align: right; }
.sv-low  { color: #059669; }
.sv-mid  { color: #d97706; }
.sv-high { color: #dc2626; }

/* ═══════ CHAT BUBBLES ═══════ */
.user-bub {
    max-width: 75%; background: linear-gradient(135deg, #0ea5e9, #38bdf8); color: #fff;
    border-radius: 18px 18px 4px 18px; padding: 13px 18px;
    font-size: .84rem; line-height: 1.6; box-shadow: 0 2px 8px rgba(14,165,233,.2);
}
.bot-bub {
    max-width: 78%; border-radius: 18px 18px 18px 4px; padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
}
.bb-real { background: #ecfdf5; border: 1.5px solid #a7f3d0; }
.bb-fake { background: #fef2f2; border: 1.5px solid #fecaca; }

/* ═══════ MISC ═══════ */
.section-label { font-size:.7rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.8px; color:#0284c7; margin-bottom:10px; }
.empty-state { text-align:center; padding:48px 20px; }
.empty-icon  { font-size:2.5rem; margin-bottom:10px; opacity:.4; }
.empty-text  { font-size:.88rem; color:#1e293b; line-height:1.7; }
.empty-text strong { color: #0066cc; }
.divider     { border:none; border-top:1.5px solid #e0f2fe; margin:20px 0; }
.ocr-output  { background:#f0f9ff; border:1.5px solid #e0f2fe; border-radius:10px;
  padding:14px 16px; font-size:.8rem; color:#475569; line-height:1.7;
  max-height:180px; overflow-y:auto; white-space:pre-wrap; margin-bottom:14px; }
.img-box { background:#f0f9ff; border:1.5px solid #e0f2fe; border-radius:12px; padding:14px; }
.footer { margin-top:40px; padding-top:20px; border-top:1.5px solid #e0f2fe;
  font-size:.72rem; color:#94a3b8; text-align:center; }
.footer strong { color:#64748b; }
.stSpinner > div { border-top-color: #0ea5e9 !important; }
[data-testid="stAlert"] { border-radius:10px !important; font-size:.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load ML
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    mp, vp = os.path.join("models","model.pkl"), os.path.join("models","vectorizer.pkl")
    if os.path.exists(mp) and os.path.exists(vp):
        return joblib.load(mp), joblib.load(vp), True
    return None, None, False

@st.cache_resource(show_spinner=False)
def load_ocr():
    try:
        import easyocr
        return easyocr.Reader(['en'], gpu=False, verbose=False), True
    except Exception:
        return None, False

model, vectorizer, IS_LIVE = load_artifacts()
ocr_reader, OCR_OK = load_ocr()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Engine
# ─────────────────────────────────────────────────────────────────────────────
_FK = [r'\bbreaking\b',r'\bshocking\b',r'\bmiracle cure\b',r'\bexposed\b',
       r'\bconspiracy\b',r'\bhoax\b',r'\bdeep state\b',r'\billuminati\b',
       r'\bwake up\b',r'\bshare before deleted\b',r'\bplandemic\b',r'\bmicrochip\b',
       r'\b5g\b',r'\bcover.?up\b',r'\bgovernment.?hiding\b',r'\bthey don.t want\b',
       r'\bsecret(ly)?\b',r'\bbanned\b',r'\bcure\b',r'\bmiracl\b']
_RK = [r'\baccording to\b',r'\bofficial\b',r'\bpublished\b',r'\bresearch\b',
       r'\bstudy\b',r'\bscientist\b',r'\buniversity\b',r'\breport\b',
       r'\bpress release\b',r'\bspokesperson\b',r'\bfact.?check\b',r'\bverif\b',
       r'\bpeer.?review\b',r'\bdata\b',r'\bevidence\b',r'\bjournal\b']

def heuristic(text):
    t = text.lower()
    fs = sum(bool(re.search(p,t)) for p in _FK)
    rs = sum(bool(re.search(p,t)) for p in _RK)
    pf = round(min(max((fs+.5)/(fs+rs+1.5),.1),.9),4); pr = round(1-pf,4)
    return {"label":"FAKE" if pf>=pr else "REAL","confidence":max(pf,pr),"proba_real":pr,"proba_fake":pf}

def ml_predict(text):
    cleaned = preprocess_text(text); features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    if hasattr(model,"predict_proba"):
        proba = model.predict_proba(features)[0]; classes = list(model.classes_)
        iF = next((i for i,c in enumerate(classes) if str(c).upper() in ("0","FAKE")),0)
        iR = next((i for i,c in enumerate(classes) if str(c).upper() in ("1","REAL")),1)
        pf,pr = float(proba[iF]),float(proba[iR])
    else:
        pf = 1.0 if str(pred).upper() in ("0","FAKE") else 0.0; pr = 1-pf
    label = "FAKE" if str(pred).upper() in ("0","FAKE") else "REAL"
    return {"label":label,"confidence":round(max(pf,pr),4),"proba_real":round(pr,4),"proba_fake":round(pf,4)}

def predict(text):
    return ml_predict(text) if (IS_LIVE and model and vectorizer) else heuristic(text)

def extract_ocr(pil_img):
    if not OCR_OK: return "",False
    import numpy as np
    return " ".join(ocr_reader.readtext(np.array(pil_img.convert("RGB")),detail=0,paragraph=True)).strip(), True

def fetch_url_text(url):
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            html = r.read().decode("utf-8",errors="ignore")
        text = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>',' ',html,flags=re.S)
        text = re.sub(r'<[^>]+>',' ',text); text = re.sub(r'&[a-z]+;',' ',text)
        return re.sub(r'\s+',' ',text).strip()[:3000], True
    except Exception as e: return str(e), False


# ─────────────────────────────────────────────────────────────────────────────
# ██  REAL-TIME TEXT ANALYTICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def analyze_text_signals(text: str) -> dict:
    """
    Compute real-time NLP signals from raw text.
    Returns metrics for: word count, sentence count, avg word length,
    caps ratio, exclamation density, question density,
    clickbait score, emotional tone, readability, source quality.
    """
    words    = text.split()
    wc       = len(words)
    sc       = max(len(re.split(r'[.!?]+', text)), 1)
    avg_wl   = round(sum(len(w) for w in words) / max(wc, 1), 1)
    caps     = sum(1 for c in text if c.isupper())
    caps_pct = round(caps / max(len(text), 1) * 100, 1)
    excl     = text.count('!')
    ques     = text.count('?')
    excl_d   = round(excl / max(sc, 1) * 100, 1)

    # Clickbait score (0-100)
    cb_signals = [
        bool(re.search(r'BREAKING|SHOCKING|EXPOSED|URGENT|ALERT', text, re.I)),
        bool(re.search(r'you won.t believe|they don.t want|secret(ly)?', text, re.I)),
        bool(re.search(r'share (before|now|this)', text, re.I)),
        caps_pct > 15,
        excl > 3,
        bool(re.search(r'!!+', text)),
        bool(re.search(r'\?\?+', text)),
        bool(re.search(r'miracle|cure|banned|conspiracy', text, re.I)),
    ]
    clickbait = min(round(sum(cb_signals) / len(cb_signals) * 100), 100)

    # Emotional manipulation score (0-100)
    emo_signals = [
        bool(re.search(r'fear|terrif|danger|threat|horror|scary|alarming', text, re.I)),
        bool(re.search(r'outrage|furious|angry|disgusting|unbelievable', text, re.I)),
        bool(re.search(r'heartbreak|tragic|devastating|victim|suffer', text, re.I)),
        bool(re.search(r'must (act|share|read|watch|see)', text, re.I)),
        bool(re.search(r'wake up|open your eyes|truth they', text, re.I)),
        excl > 2,
        caps_pct > 12,
    ]
    emotional = min(round(sum(emo_signals) / len(emo_signals) * 100), 100)

    # Source credibility (0-100, higher = better)
    src_signals = [
        bool(re.search(r'according to|stated|confirmed|reported', text, re.I)),
        bool(re.search(r'official|government|ministry|department', text, re.I)),
        bool(re.search(r'university|professor|dr\.|researcher|scientist', text, re.I)),
        bool(re.search(r'study|research|published|journal|peer.?review', text, re.I)),
        bool(re.search(r'reuters|ap |associated press|bbc|cnn', text, re.I)),
        bool(re.search(r'spokesperson|press (release|conference)', text, re.I)),
        bool(re.search(r'data|evidence|statistics|survey|poll', text, re.I)),
    ]
    source_q = min(round(sum(src_signals) / len(src_signals) * 100), 100)

    # Readability (Flesch-like approximation, 0-100)
    avg_syl = max(avg_wl * 0.6, 1)
    avg_sl  = wc / max(sc, 1)
    flesch  = max(0, min(100, round(206.835 - 1.015 * avg_sl - 84.6 * (avg_syl / max(avg_wl, 1)))))

    # Writing quality composite
    quality = max(0, min(100, round(
        source_q * 0.35 +
        flesch * 0.25 +
        (100 - clickbait) * 0.2 +
        (100 - emotional) * 0.2
    )))

    return {
        "word_count": wc, "sentence_count": sc, "avg_word_len": avg_wl,
        "caps_pct": caps_pct, "exclamation_count": excl, "question_count": ques,
        "excl_density": excl_d,
        "clickbait": clickbait, "emotional": emotional,
        "source_quality": source_q, "readability": flesch,
        "writing_quality": quality,
    }


def signal_bar(label: str, value: int, invert: bool = False) -> str:
    """Render a signal bar. If invert=True, high=good (green). Default: high=bad (red)."""
    if invert:
        cls = "sb-low" if value >= 60 else ("sb-mid" if value >= 30 else "sb-high")
        vcl = "sv-low" if value >= 60 else ("sv-mid" if value >= 30 else "sv-high")
    else:
        cls = "sb-high" if value >= 60 else ("sb-mid" if value >= 30 else "sb-low")
        vcl = "sv-high" if value >= 60 else ("sv-mid" if value >= 30 else "sv-low")
    return f"""<div class="signal-item">
      <span class="signal-label">{label}</span>
      <div class="signal-bar-track"><div class="signal-bar-fill {cls}" style="width:{value}%"></div></div>
      <span class="signal-val {vcl}">{value}%</span>
    </div>"""


def analytics_panel(signals: dict) -> str:
    """Render the full real-time analytics panel."""
    return f"""
<div class="analytics-panel">
  <div class="analytics-title">📊 Real-Time Text Analytics</div>
  <div class="metric-grid">
    <div class="metric-item"><div class="mi-val">{signals['word_count']}</div><div class="mi-lbl">Words</div></div>
    <div class="metric-item"><div class="mi-val">{signals['sentence_count']}</div><div class="mi-lbl">Sentences</div></div>
    <div class="metric-item"><div class="mi-val">{signals['avg_word_len']}</div><div class="mi-lbl">Avg Word Len</div></div>
    <div class="metric-item"><div class="mi-val">{signals['caps_pct']}%</div><div class="mi-lbl">Caps Ratio</div></div>
    <div class="metric-item"><div class="mi-val">{signals['exclamation_count']}</div><div class="mi-lbl">Exclamations</div></div>
    <div class="metric-item"><div class="mi-val">{signals['question_count']}</div><div class="mi-lbl">Questions</div></div>
  </div>
  <div class="analytics-title" style="margin-top:4px;">🔬 Credibility Signals</div>
  <div class="signal-row">
    {signal_bar("Clickbait", signals['clickbait'], invert=False)}
    {signal_bar("Emotional", signals['emotional'], invert=False)}
    {signal_bar("Source Quality", signals['source_quality'], invert=True)}
    {signal_bar("Readability", signals['readability'], invert=True)}
    {signal_bar("Writing Quality", signals['writing_quality'], invert=True)}
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Result card
# ─────────────────────────────────────────────────────────────────────────────
def result_card(res, source="text"):
    l=res["label"]; c=res["confidence"]*100; rp=res["proba_real"]*100; fp=res["proba_fake"]*100
    r=l=="REAL"; cc="card-real" if r else "card-fake"; vc="verdict-real" if r else "verdict-fake"
    fc="fill-real" if r else "fill-fake"; ic="✅" if r else "🚫"
    d = "Content consistent with credible, verified reporting." if r else "Content shows misinformation signals."
    sl={"text":"💬 Text","image":"📸 Image","url":"🔗 URL"}.get(source,source)
    en="ML Model" if IS_LIVE else "Heuristic"
    return f"""<div class="result-card {cc}">
  <div class="result-verdict {vc}">{ic} {l} NEWS</div>
  <p class="result-desc">{d}</p>
  <div class="conf-wrap"><div class="conf-label"><span>Confidence Score</span>
    <span style="color:{'#059669' if r else '#dc2626'};font-weight:700;">{c:.1f}%</span></div>
    <div class="conf-track"><div class="conf-fill {fc}" style="width:{c:.1f}%"></div></div></div>
  <div class="prob-row">
    <div class="prob-pill"><div class="pl">P(Real)</div><div class="pv pv-real">{rp:.1f}%</div></div>
    <div class="prob-pill"><div class="pl">P(Fake)</div><div class="pv pv-fake">{fp:.1f}%</div></div>
    <div class="prob-pill"><div class="pl">Source</div><div class="pv pv-mode">{sl}</div></div>
    <div class="prob-pill"><div class="pl">Engine</div><div class="pv pv-mode">{en}</div></div>
  </div></div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for k,v in [("history",[]),("total",0),("n_real",0),("n_fake",0),("input","")]:
    if k not in st.session_state: st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# ██  HEADER
# ─────────────────────────────────────────────────────────────────────────────
engine_badge = '<span class="h-badge hb-live">● Model Active</span>' if IS_LIVE else '<span class="h-badge hb-demo">◉ Demo Mode</span>'
st.markdown(f"""
<div class="header">
  <div class="header-left">
    <div class="header-icon">🛡️</div>
    <div>
      <div class="header-title">Fake News Detection using Machine Learning</div>
      <div class="header-sub">NLP · ML Classifier · OCR · Real-Time Analytics</div>
    </div>
  </div>
  <div class="header-badges">
    <span class="h-badge hb-ml">🧠 ML Powered</span>
    {engine_badge}
  </div>
</div>
""", unsafe_allow_html=True)

# KPIs
st.markdown(f"""<div class="kpi-row">
  <div class="kpi-card"><div class="kpi-val">{st.session_state.total}</div><div class="kpi-lbl">Articles Analyzed</div></div>
  <div class="kpi-card"><div class="kpi-val" style="color:#059669">{st.session_state.n_real}</div><div class="kpi-lbl">Real</div></div>
  <div class="kpi-card"><div class="kpi-val" style="color:#dc2626">{st.session_state.n_fake}</div><div class="kpi-lbl">Fake</div></div>
  <div class="kpi-card"><div class="kpi-val" style="color:#0ea5e9">{'ML' if IS_LIVE else 'Demo'}</div><div class="kpi-lbl">Engine</div></div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬  Text Analytics","📸  Image OCR","🔗  URL Analysis"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — TEXT CHATBOT + REAL-TIME ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    te = [h for h in st.session_state.history if h["source"]=="text"]
    if not te:
        st.markdown("""<div class="empty-state"><div class="empty-icon">💬</div>
          <div class="empty-text">Paste or type a news article below and click <strong>Analyze</strong><br>
          to get instant ML classification + real-time text analytics.</div></div>""", unsafe_allow_html=True)
    else:
        bbs = []
        for h in te:
            l=h["result"]["label"]; c=h["result"]["confidence"]*100; r=l=="REAL"
            vc="#059669" if r else "#dc2626"; bc="bb-real" if r else "bb-fake"
            fl="fill-real" if r else "fill-fake"; ic="✅" if r else "🚫"
            d=h["text"][:320]+("…" if len(h["text"])>320 else "")
            rp=h["result"]["proba_real"]*100; fp=h["result"]["proba_fake"]*100
            bbs.append(f"""<div style="margin-bottom:20px;">
              <div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
                <div class="user-bub">{d}</div>
                <div style="width:36px;height:36px;border-radius:50%;
                  background:linear-gradient(135deg,#0ea5e9,#38bdf8);
                  display:flex;align-items:center;justify-content:center;
                  font-size:.95rem;margin-left:10px;flex-shrink:0;color:#fff;">👤</div></div>
              <div style="display:flex;justify-content:flex-start;">
                <div style="width:36px;height:36px;border-radius:50%;background:#f0f9ff;
                  border:1.5px solid #e0f2fe;display:flex;align-items:center;justify-content:center;
                  font-size:.95rem;margin-right:10px;flex-shrink:0;">🛡️</div>
                <div class="bot-bub {bc}">
                  <div style="font-size:1.1rem;font-weight:900;color:{vc};margin-bottom:6px;">{ic} {l} NEWS</div>
                  <div style="font-size:.65rem;color:#64748b;text-transform:uppercase;font-weight:700;margin-bottom:5px;">Confidence</div>
                  <div style="background:#e2e8f0;border-radius:99px;height:7px;overflow:hidden;margin-bottom:12px;">
                    <div class="conf-fill {fl}" style="width:{c:.1f}%;height:100%;border-radius:99px;"></div></div>
                  <div style="display:flex;gap:8px;">
                    <div style="flex:1;background:#fff;border:1.5px solid #e2e8f0;border-radius:8px;padding:7px 10px;text-align:center;">
                      <div style="font-size:.55rem;color:#64748b;text-transform:uppercase;font-weight:700;">P(Real)</div>
                      <div style="font-size:.88rem;font-weight:800;color:#059669;">{rp:.1f}%</div></div>
                    <div style="flex:1;background:#fff;border:1.5px solid #e2e8f0;border-radius:8px;padding:7px 10px;text-align:center;">
                      <div style="font-size:.55rem;color:#64748b;text-transform:uppercase;font-weight:700;">P(Fake)</div>
                      <div style="font-size:.88rem;font-weight:800;color:#dc2626;">{fp:.1f}%</div></div>
                    <div style="flex:1;background:#fff;border:1.5px solid #e2e8f0;border-radius:8px;padding:7px 10px;text-align:center;">
                      <div style="font-size:.55rem;color:#64748b;text-transform:uppercase;font-weight:700;">Score</div>
                      <div style="font-size:.88rem;font-weight:800;color:{vc};">{c:.0f}%</div></div>
                  </div></div></div></div>""")
        st.markdown('<div style="max-height:460px;overflow-y:auto;padding:4px 2px;">'+"".join(bbs)+"</div>", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    SAMPLES = {"— Load a sample article —":"",
        "✅ Real — MIT Study":"Researchers at MIT published a new study in Nature journal showing a breakthrough in quantum error correction. The findings were independently verified by Stanford and Cambridge. According to the official press release, the team achieved 94% accuracy on standard benchmarks.",
        "✅ Real — RBI Policy":"The Reserve Bank of India issued an official statement confirming a 25 basis point rate cut. The Governor's press conference presented data showing inflation at 4.1% for the third consecutive quarter. Multiple agencies including Reuters and AP confirmed the report.",
        "🚫 Fake — Conspiracy":"SHOCKING: Government secretly putting microchips in vaccines EXPOSED!! Deep state hiding the truth from billions — wake up sheeple!! Share before they delete this! The illuminati are behind everything!!",
        "🚫 Fake — Health":"Doctors don't want you to know: turmeric water CURES cancer in 3 days! Big Pharma has been hiding this for 50 years! Share this miracle cure now before it gets banned!!"}
    pk = st.selectbox("sample", list(SAMPLES.keys()), label_visibility="collapsed")
    if pk != "— Load a sample article —" and SAMPLES[pk]: st.session_state.input = SAMPLES[pk]; st.rerun()

    text_input = st.text_area("text_in", value=st.session_state.input,
        placeholder="Paste or type a news article here for ML classification + real-time analytics…",
        height=140, label_visibility="collapsed")

    c1,c2 = st.columns([3,1])
    with c1: go = st.button("🔎  Analyze Article", type="primary", use_container_width=True)
    with c2:
        if st.button("Clear", type="secondary", use_container_width=True):
            st.session_state.history=[h for h in st.session_state.history if h["source"]!="text"]
            st.session_state.input=""; st.rerun()

    if go:
        txt = text_input.strip()
        if len(txt)<20: st.warning("Enter at least one full sentence.")
        else:
            with st.spinner("Running ML classification + analytics…"):
                time.sleep(0.3)
                res = predict(txt)
                signals = analyze_text_signals(txt)

            # Store with signals
            st.session_state.history.append({"text":txt,"result":res,"source":"text","signals":signals})
            st.session_state.total+=1
            if res["label"]=="REAL": st.session_state.n_real+=1
            else: st.session_state.n_fake+=1
            st.session_state.input=""
            st.rerun()

    # Show analytics for last analyzed article
    if te and "signals" in te[-1]:
        st.markdown(analytics_panel(te[-1]["signals"]), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE OCR
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    if not OCR_OK: st.error("OCR not available. Run: `pip install easyocr`")
    else:
        cu,cr = st.columns([1,1.2], gap="large")
        with cu:
            st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
            up = st.file_uploader("img", type=["jpg","jpeg","png","webp","bmp"], label_visibility="collapsed")
            if up:
                img=Image.open(up)
                st.markdown('<div class="img-box">', unsafe_allow_html=True)
                st.image(img, caption=up.name, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        with cr:
            st.markdown('<div class="section-label">Detection Result</div>', unsafe_allow_html=True)
            if up:
                if st.button("🔎  Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Extracting text via OCR…"): ex,ok=extract_ocr(img)
                    if not ok or len(ex.strip())<20: st.error("No readable text found.")
                    else:
                        with st.spinner("Classifying…"): res=predict(ex); sig=analyze_text_signals(ex)
                        st.markdown(f'<div class="ocr-output">{ex[:800]}</div>', unsafe_allow_html=True)
                        st.markdown(result_card(res,"image"), unsafe_allow_html=True)
                        st.markdown(analytics_panel(sig), unsafe_allow_html=True)
                        st.session_state.history.insert(0,{"text":ex,"result":res,"source":"image","signals":sig})
                        st.session_state.total+=1
                        if res["label"]=="REAL": st.session_state.n_real+=1
                        else: st.session_state.n_fake+=1
                        st.rerun()
            else:
                st.markdown("""<div class="empty-state" style="border:2px dashed #bae6fd;border-radius:12px;">
                  <div class="empty-icon">📸</div><div class="empty-text">Upload a newspaper clipping,<br>screenshot, or social media post</div></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — URL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">News Article URL</div>', unsafe_allow_html=True)
    uc,bc = st.columns([4,1])
    with uc: url_in = st.text_input("url", placeholder="https://example.com/news-article", label_visibility="collapsed")
    with bc: url_go = st.button("🔎  Fetch & Analyze", type="primary", use_container_width=True)
    if url_go:
        u=url_in.strip()
        if not u.startswith("http"): st.warning("Enter a valid URL.")
        else:
            with st.spinner("Fetching article…"): text,ok=fetch_url_text(u)
            if not ok or len(text.strip())<50: st.error(f"Failed: {text}")
            else:
                with st.spinner("Classifying…"): res=predict(text); sig=analyze_text_signals(text)
                ct,cr2 = st.columns([1,1], gap="large")
                with ct:
                    st.markdown('<div class="section-label">Extracted Content</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="ocr-output">{text[:700]}…</div>', unsafe_allow_html=True)
                with cr2:
                    st.markdown('<div class="section-label">Detection Result</div>', unsafe_allow_html=True)
                    st.markdown(result_card(res,"url"), unsafe_allow_html=True)
                st.markdown(analytics_panel(sig), unsafe_allow_html=True)
                st.session_state.history.insert(0,{"text":text,"result":res,"source":"url","signals":sig})
                st.session_state.total+=1
                if res["label"]=="REAL": st.session_state.n_real+=1
                else: st.session_state.n_fake+=1
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""<div class="footer">
  <strong>Fake News Detection using Machine Learning</strong> &nbsp;·&nbsp;
  Python &nbsp;·&nbsp; Scikit-learn &nbsp;·&nbsp; NLTK &nbsp;·&nbsp; Streamlit
</div>""", unsafe_allow_html=True)
