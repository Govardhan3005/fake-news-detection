"""
============================================================
app.py  ─  Fake News Detection using Machine Learning
============================================================
Author      : Govardhan N
Project     : Fake News Detection using Machine Learning
Description : Professional AI-powered chatbot + image OCR
              fake news detector. Supports:
                • Text / chat input
                • Image upload  (newspaper, screenshot, etc.)
              Runs in Demo Mode until a trained model is saved.
============================================================
"""

import os, re, time, io, warnings
warnings.filterwarnings("ignore")

import joblib
import streamlit as st
from PIL import Image
from preprocess import preprocess_text

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Fake News Detection using Machine Learning",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ██  CSS  ── dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg0:#07090f;--bg1:#0d1117;--bg2:#161b27;--bg3:#1e2538;
  --blue:#4f8ef7;--purple:#9b72f7;--green:#22d3a5;--red:#f05a6e;
  --yellow:#f5a623;--cyan:#38bdf8;
  --txt:#e8edf5;--muted:#6b7a99;
  --border:rgba(255,255,255,0.07);
  --glass:rgba(255,255,255,0.03);
  --shadow:0 8px 32px rgba(0,0,0,.55);
}

html,body,[class*="css"]{font-family:'Inter',sans-serif!important;
  background:var(--bg0)!important;color:var(--txt)!important;}

.stApp{background:radial-gradient(ellipse 120% 80% at 60% -10%,
  rgba(79,142,247,.12) 0%,transparent 55%),
  linear-gradient(160deg,#07090f 0%,#0d1117 60%,#07090f 100%)!important;}

#MainMenu,footer,header{visibility:hidden}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0d1117 0%,#07090f 100%)!important;
  border-right:1px solid var(--border)!important;}

/* scrollbar */
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--purple);border-radius:99px}

/* ── Hero banner ── */
.hero{
  background:linear-gradient(135deg,#0f1729 0%,#141d33 60%,#0c1220 100%);
  border:1px solid rgba(155,114,247,.25);
  border-radius:22px;padding:32px 40px 28px;
  margin-bottom:28px;position:relative;overflow:hidden;
  box-shadow:var(--shadow),0 0 60px rgba(155,114,247,.1);}
.hero::before{content:'';position:absolute;top:-40%;left:-20%;
  width:140%;height:180%;
  background:radial-gradient(ellipse,rgba(79,142,247,.06) 0%,transparent 60%);
  pointer-events:none;}
.hero-logo{font-size:2.6rem;font-weight:900;letter-spacing:-1px;
  background:linear-gradient(90deg,#c4b5fd,#60a5fa,#34d399,#f472b6);
  background-size:200%;animation:grad 6s linear infinite;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;margin:0 0 6px 0;}
@keyframes grad{0%{background-position:0%}100%{background-position:200%}}
.hero-sub{color:var(--muted);font-size:.95rem;margin:0;}
.hero-badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;}
.badge{display:inline-flex;align-items:center;gap:5px;
  padding:4px 12px;border-radius:20px;font-size:.72rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.6px;}
.badge-live{background:rgba(34,211,165,.12);border:1px solid rgba(34,211,165,.3);color:#34d399;}
.badge-demo{background:rgba(245,166,35,.12);border:1px solid rgba(245,166,35,.3);color:#fbbf24;}
.badge-ocr{background:rgba(56,189,248,.12);border:1px solid rgba(56,189,248,.3);color:#38bdf8;}
.badge-ai{background:rgba(155,114,247,.12);border:1px solid rgba(155,114,247,.3);color:#c4b5fd;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
  background:var(--bg2);border-radius:14px;padding:4px;gap:4px;
  border:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{
  border-radius:10px;font-weight:600;font-size:.85rem;
  color:var(--muted);padding:10px 24px;border:none;
  transition:all .2s ease;}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,var(--purple),var(--blue))!important;
  color:#fff!important;box-shadow:0 4px 16px rgba(155,114,247,.35);}

/* ── Chat bubbles ── */
.chat-wrap{max-height:520px;overflow-y:auto;padding:6px 2px 12px;
  scroll-behavior:smooth;}
.msg-row{display:flex;margin-bottom:20px;
  animation:popIn .3s cubic-bezier(.34,1.56,.64,1);}
@keyframes popIn{from{opacity:0;transform:scale(.92) translateY(10px)}
  to{opacity:1;transform:scale(1) translateY(0)}}
.msg-row.user{justify-content:flex-end;}
.msg-row.bot{justify-content:flex-start;}
.bubble{max-width:78%;padding:14px 18px;border-radius:18px;
  font-size:.88rem;line-height:1.65;word-break:break-word;}
.bubble-user{background:linear-gradient(135deg,#4f8ef7,#7c5bf7);
  color:#fff;border-bottom-right-radius:4px;
  box-shadow:0 4px 18px rgba(79,142,247,.35);}
.bubble-bot{background:var(--bg2);border:1px solid var(--border);
  color:var(--txt);border-bottom-left-radius:4px;
  box-shadow:0 4px 18px rgba(0,0,0,.3);}
.av{width:38px;height:38px;border-radius:50%;display:flex;
  align-items:center;justify-content:center;font-size:1.15rem;flex-shrink:0;}
.av-user{background:linear-gradient(135deg,#4f8ef7,#7c5bf7);margin-left:10px;order:2;}
.av-bot{background:var(--bg3);border:1px solid var(--border);margin-right:10px;}

/* ── Verdict card ── */
.vcard{border-radius:16px;padding:18px 22px;margin-top:10px;overflow:hidden;position:relative;}
.vcard-real{background:linear-gradient(135deg,rgba(34,211,165,.13),rgba(4,120,87,.08));
  border:1px solid rgba(34,211,165,.3);box-shadow:0 0 28px rgba(34,211,165,.15);}
.vcard-fake{background:linear-gradient(135deg,rgba(240,90,110,.13),rgba(159,18,57,.08));
  border:1px solid rgba(240,90,110,.3);box-shadow:0 0 28px rgba(240,90,110,.15);}
.vtitle{font-size:1.7rem;font-weight:900;letter-spacing:1.5px;margin-bottom:4px;}
.vcard-real .vtitle{color:#34d399}
.vcard-fake .vtitle{color:#f87171}
.vsub{font-size:.82rem;color:var(--muted);margin-bottom:12px;line-height:1.5;}

/* confidence bar */
.cbar-bg{background:rgba(255,255,255,.07);border-radius:99px;
  height:8px;margin:6px 0 4px;overflow:hidden;}
.cbar-fill{height:100%;border-radius:99px;transition:width .9s ease;}
.fill-real{background:linear-gradient(90deg,#059669,#34d399);}
.fill-fake{background:linear-gradient(90deg,#dc2626,#f87171);}

/* metric pills */
.mpills{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;}
.mpill{display:flex;flex-direction:column;align-items:center;
  background:rgba(255,255,255,.04);border:1px solid var(--border);
  border-radius:10px;padding:8px 14px;min-width:76px;}
.mpill .lbl{font-size:.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;}
.mpill .val{font-size:1rem;font-weight:700;margin-top:2px;}
.v-real{color:#34d399}.v-fake{color:#f87171}.v-purple{color:#c4b5fd}

/* ── Image result ── */
.img-preview-box{border:2px dashed rgba(155,114,247,.35);border-radius:16px;
  padding:12px;background:var(--bg2);margin-bottom:16px;text-align:center;}
.ocr-box{background:var(--bg3);border:1px solid var(--border);border-radius:12px;
  padding:14px 16px;font-family:'JetBrains Mono',monospace;font-size:.8rem;
  color:var(--muted);max-height:200px;overflow-y:auto;line-height:1.7;
  white-space:pre-wrap;margin-bottom:14px;}
.ocr-label{font-size:.7rem;text-transform:uppercase;letter-spacing:.7px;
  color:var(--cyan);font-weight:600;margin-bottom:6px;}

/* ── Inputs ── */
.stTextArea textarea{
  background:var(--bg2)!important;
  border:1px solid rgba(155,114,247,.25)!important;
  border-radius:12px!important;color:var(--txt)!important;
  font-size:.88rem!important;resize:none!important;
  transition:border-color .2s!important;}
.stTextArea textarea:focus{border-color:var(--blue)!important;
  box-shadow:0 0 0 3px rgba(79,142,247,.15)!important;}
.stTextArea textarea::placeholder{color:#3d4f6e!important;}

/* buttons */
.stButton>button{font-family:'Inter',sans-serif!important;font-weight:600!important;
  border-radius:10px!important;transition:all .2s ease!important;letter-spacing:.3px!important;}
.stButton>button:hover{transform:translateY(-2px)!important;
  box-shadow:0 8px 20px rgba(0,0,0,.3)!important;}

/* sidebar card */
.scard{background:rgba(255,255,255,.03);border:1px solid var(--border);
  border-radius:12px;padding:14px 16px;margin-bottom:14px;}
.scard h4{font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;
  color:var(--blue);margin:0 0 10px 0;}

/* fancy hr */
.fhr{border:none;height:1px;
  background:linear-gradient(90deg,transparent,var(--purple),var(--blue),transparent);
  margin:22px 0;}

/* file uploader */
[data-testid="stFileUploader"]{
  background:var(--bg2)!important;border-radius:14px!important;
  border:2px dashed rgba(155,114,247,.3)!important;padding:8px!important;}

/* selectbox */
[data-testid="stSelectbox"]>div>div{
  background:var(--bg2)!important;border:1px solid var(--border)!important;
  color:var(--txt)!important;border-radius:10px!important;}

/* success / error boxes */
[data-testid="stAlert"]{border-radius:10px!important;}

/* spinner */
.stSpinner>div{border-top-color:var(--purple)!important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached: load trained model (or stay in demo mode)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    mp = os.path.join("models","model.pkl")
    vp = os.path.join("models","vectorizer.pkl")
    if os.path.exists(mp) and os.path.exists(vp):
        return joblib.load(mp), joblib.load(vp), True
    return None, None, False

model, vectorizer, IS_LIVE = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# Cached: EasyOCR reader (heavy — load once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ocr():
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return reader, True
    except Exception:
        return None, False

ocr_reader, OCR_AVAILABLE = load_ocr()

# ─────────────────────────────────────────────────────────────────────────────
# Heuristic signals (demo mode)
# ─────────────────────────────────────────────────────────────────────────────
_FAKE = [
    r'\bbreaking\b',r'\bshocking\b',r'\bmiracle\b',r'\bexplosive\b',
    r'\bsecret\b',r'\bconspiracy\b',r'\bhoax\b',r'\bscam\b',r'\bexposed\b',
    r'\bsatan\b',r'\billuminati\b',r'\bdeep state\b',r'\bfake\b',r'\blie\b',
    r'\bcover.?up\b',r'\bplandemic\b',r'\b5g\b',r'\bwake up\b',
    r'\bthey.?don.t.?want\b',r'\bshare before deleted\b',r'\btruth revealed\b',
    r'\bmicrochip\b',r'\bbill gates\b.*\bvaccine\b',r'\bgovernment.?hiding\b',
]
_REAL = [
    r'\baccording to\b',r'\bofficial\b',r'\bstatement\b',r'\bresearch\b',
    r'\bstudy\b',r'\bscientist\b',r'\buniversity\b',r'\breport\b',
    r'\bconference\b',r'\bdata\b',r'\bstatistic\b',r'\bjournalist\b',
    r'\bnews agency\b',r'\bsource\b',r'\binvestigation\b',r'\bfact.?check\b',
    r'\bpublished\b',r'\bverif\b',r'\bpress release\b',r'\bspokesperson\b',
]

def heuristic(text:str)->dict:
    t = text.lower()
    fs = sum(bool(re.search(p,t)) for p in _FAKE)
    rs = sum(bool(re.search(p,t)) for p in _REAL)
    total = fs + rs + 1
    pf = min(max((fs+0.5)/(total+1), 0.10), 0.90)
    pr = round(1-pf, 4); pf = round(pf, 4)
    lbl = "FAKE" if pf >= pr else "REAL"
    return {"label":lbl,"confidence":max(pf,pr),"proba_real":pr,"proba_fake":pf}

def ml_predict(text:str)->dict:
    cleaned  = preprocess_text(text)
    features = vectorizer.transform([cleaned])
    pred     = model.predict(features)[0]
    if hasattr(model,"predict_proba"):
        proba   = model.predict_proba(features)[0]
        classes = list(model.classes_)
        iR,iF  = 1,0
        for i,c in enumerate(classes):
            if str(c).upper() in ("1","REAL"): iR=i
            elif str(c).upper() in ("0","FAKE"): iF=i
        pf,pr = float(proba[iF]), float(proba[iR])
    else:
        pf = 1.0 if str(pred).upper() in ("0","FAKE") else 0.0
        pr = 1-pf
    lbl = "FAKE" if str(pred).upper() in ("0","FAKE") else "REAL"
    return {"label":lbl,"confidence":round(max(pf,pr),4),
            "proba_real":round(pr,4),"proba_fake":round(pf,4)}

def run_prediction(text:str)->dict:
    if IS_LIVE and model and vectorizer:
        return ml_predict(text)
    return heuristic(text)

# ─────────────────────────────────────────────────────────────────────────────
# OCR: extract text from PIL image
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_image(pil_img: Image.Image) -> tuple[str, bool]:
    """
    Returns (extracted_text, success_flag).
    Tries EasyOCR first; falls back to a basic message if unavailable.
    """
    if not OCR_AVAILABLE or ocr_reader is None:
        return "", False
    import numpy as np
    img_array = np.array(pil_img.convert("RGB"))
    results   = ocr_reader.readtext(img_array, detail=0, paragraph=True)
    text      = " ".join(results).strip()
    return text, bool(text)

# ─────────────────────────────────────────────────────────────────────────────
# Build verdict HTML block
# ─────────────────────────────────────────────────────────────────────────────
def verdict_html(result:dict, mode:str, source:str="text")->str:
    lbl    = result["label"]
    cp     = result["confidence"]*100
    rp     = result["proba_real"]*100
    fp     = result["proba_fake"]*100
    real   = lbl=="REAL"
    icon   = "✅" if real else "🚫"
    cls    = "vcard-real" if real else "vcard-fake"
    fcls   = "fill-real"  if real else "fill-fake"
    vclr   = "#34d399"    if real else "#f87171"

    src_icon = "📸 Image" if source=="image" else "💬 Text"
    detail = (
        "Content shows <strong>credible, fact-based reporting</strong> consistent with verified news sources."
        if real else
        "Content exhibits <strong>characteristics of misinformation</strong> — exaggerated claims, emotive language, or unverified assertions."
    )

    tips = (
        "<li>Cross-check with <strong>Reuters, AP, BBC</strong></li>"
        "<li>Look for named sources and citations</li>"
        "<li>Check the publication date</li>"
        if real else
        "<li>Verify with <strong>Snopes, FactCheck.org</strong></li>"
        "<li>Search the headline in Google News</li>"
        "<li>Check the original source website</li>"
    )

    return f"""
<div class="vcard {cls}">
  <div class="vtitle">{icon} {lbl} NEWS</div>
  <p class="vsub">{detail}</p>

  <div style="font-size:.72rem;color:#64748b;margin-bottom:2px;">
    CONFIDENCE SCORE &mdash;
    <span style="color:{vclr};font-weight:700;">{cp:.1f}%</span>
  </div>
  <div class="cbar-bg">
    <div class="cbar-fill {fcls}" style="width:{cp:.1f}%"></div>
  </div>

  <div class="mpills">
    <div class="mpill"><span class="lbl">P(Real)</span>
      <span class="val v-real">{rp:.1f}%</span></div>
    <div class="mpill"><span class="lbl">P(Fake)</span>
      <span class="val v-fake">{fp:.1f}%</span></div>
    <div class="mpill"><span class="lbl">Engine</span>
      <span class="val v-purple" style="font-size:.72rem;">{mode}</span></div>
    <div class="mpill"><span class="lbl">Input</span>
      <span class="val" style="font-size:.72rem;color:#38bdf8;">{src_icon}</span></div>
  </div>

  <div style="margin-top:14px;font-size:.78rem;color:#94a3b8;">
    <strong style="color:{vclr};">💡 Next Steps:</strong>
    <ul style="margin:4px 0 0 14px;line-height:1.9;">{tips}</ul>
  </div>
</div>"""

# ─────────────────────────────────────────────────────────────────────────────
# Sample articles
# ─────────────────────────────────────────────────────────────────────────────
SAMPLES = {
    "🔬 Real — Scientific Report":(
        "Researchers at MIT's Computer Science and Artificial Intelligence Laboratory "
        "have developed a new machine learning algorithm that significantly improves "
        "natural language understanding. The study, published in Nature Machine Intelligence, "
        "was peer-reviewed by 12 independent experts. According to the official press release, "
        "the model achieved a 94.3% accuracy rate on standard benchmarks, verified across "
        "five international institutions including Stanford and Cambridge."
    ),
    "🏛️ Real — Government Policy":(
        "The Reserve Bank of India issued an official monetary policy statement on Thursday, "
        "announcing a 25 basis point reduction in the repo rate to 6.25%. According to the "
        "RBI Governor's press conference, this decision was unanimous among all six committee "
        "members. The data released shows inflation stabilised at 4.1% for the third "
        "consecutive quarter, well within the 2-6% tolerance band."
    ),
    "🚨 Fake — Conspiracy Theory":(
        "SHOCKING!! Government secretly putting MICROCHIPS in COVID vaccines exposed!! "
        "Deep state and Illuminati controlling world population through 5G towers!! "
        "Bill Gates ADMITTED it in this video they deleted!! Wake up sheeple — share "
        "before they take this down!! The truth they DON'T want you to see!! "
        "Thousands already CHIPPED without knowing! BREAKING bombshell revelation!!"
    ),
    "💊 Fake — Health Hoax":(
        "MIRACLE CURE doctors are hiding from you!! Drinking turmeric water cures cancer "
        "in 3 days — BIG PHARMA suppressing this secret for 50 YEARS!! Hospitals will "
        "lose billions if this goes viral! Share this explosive truth now before it gets "
        "BANNED! One weird trick eliminated diabetes completely — FDA doesn't want you knowing!"
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
def _init(k,v):
    if k not in st.session_state: st.session_state[k]=v

_init("chat_history", [])
_init("img_history",  [])
_init("input_text",   "")
_init("total",  0); _init("n_real", 0); _init("n_fake", 0)
_init("img_total",0); _init("img_real",0); _init("img_fake",0)

# ─────────────────────────────────────────────────────────────────────────────
# ██  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 10px;">
      <div style="font-size:2.8rem;">🔍</div>
      <div style="font-size:.9rem;font-weight:800;
        background:linear-gradient(90deg,#c4b5fd,#60a5fa,#34d399);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;letter-spacing:-.3px;">Fake News Detection</div>
      <div style="font-size:.65rem;color:#374151;margin-top:3px;letter-spacing:.5px;">
        USING MACHINE LEARNING
      </div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(255,255,255,.06);margin:6px 0 16px;">
    """, unsafe_allow_html=True)

    # mode badges
    mode_tag = "AI Model" if IS_LIVE else "Demo"
    if IS_LIVE:
        st.markdown('<span class="badge badge-live">● AI Model Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-demo">◉ Demo Mode</span>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:.72rem;color:#64748b;margin:8px 0 0;padding:10px;
          background:rgba(245,158,11,.05);border:1px solid rgba(245,158,11,.18);
          border-radius:8px;line-height:1.6;">
          ⚠️ <strong style="color:#fbbf24;">No trained model.</strong><br>
          Upload your dataset then run<br>
          <code style="color:#a78bfa;">python train.py</code>
        </div>""", unsafe_allow_html=True)

    if OCR_AVAILABLE:
        st.markdown('<br><span class="badge badge-ocr">📸 OCR Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<br><span class="badge" style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#f87171;">📸 OCR Unavailable</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stats ──
    st.markdown('<div class="scard"><h4>📊 Session Stats</h4>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Total", st.session_state.total + st.session_state.img_total)
    c2.metric("✅ Real", st.session_state.n_real + st.session_state.img_real)
    c3.metric("🚫 Fake", st.session_state.n_fake + st.session_state.img_fake)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="scard"><h4>💬 Text Checks</h4>', unsafe_allow_html=True)
    a1,a2,a3 = st.columns(3)
    a1.metric("Total", st.session_state.total)
    a2.metric("Real",  st.session_state.n_real)
    a3.metric("Fake",  st.session_state.n_fake)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="scard"><h4>📸 Image Checks</h4>', unsafe_allow_html=True)
    b1,b2,b3 = st.columns(3)
    b1.metric("Total", st.session_state.img_total)
    b2.metric("Real",  st.session_state.img_real)
    b3.metric("Fake",  st.session_state.img_fake)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Sample loader ──
    st.markdown('<div class="scard"><h4>📰 Sample Articles</h4>', unsafe_allow_html=True)
    pick = st.selectbox("Choose sample", ["— select —"]+list(SAMPLES.keys()),
                        label_visibility="collapsed")
    if pick != "— select —":
        if st.button("📋 Load Sample", use_container_width=True):
            st.session_state.input_text = SAMPLES[pick]
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Clear ──
    col_cc, col_ci = st.columns(2)
    with col_cc:
        if st.button("🗑️ Chat", use_container_width=True):
            st.session_state.chat_history=[]
            st.session_state.total=st.session_state.n_real=st.session_state.n_fake=0
            st.session_state.input_text=""
            st.rerun()
    with col_ci:
        if st.button("🗑️ Images", use_container_width=True):
            st.session_state.img_history=[]
            st.session_state.img_total=st.session_state.img_real=st.session_state.img_fake=0
            st.rerun()

    with st.expander("ℹ️ How It Works"):
        st.markdown("""
**Text Tab**
1. Type / paste any news article
2. Click **Analyze** → instant verdict

**Image Tab**
1. Upload a photo (newspaper, screenshot, WhatsApp forward, etc.)
2. OCR extracts all text automatically
3. AI analyzes extracted text → verdict

**Confidence Score**
- 90–100% → Very High certainty
- 70–89%  → High certainty
- 50–69%  → Moderate — verify manually
        """)

    st.markdown("""
    <div style="text-align:center;margin-top:20px;font-size:.65rem;color:#1f2937;">
    🔍 Fake News Detection using ML © 2025 &nbsp;|&nbsp; Built with Python & Streamlit
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██  MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

# Hero banner
mode_badge = (
    '<span class="badge badge-live" style="margin-left:12px;">● AI Model</span>'
    if IS_LIVE else
    '<span class="badge badge-demo" style="margin-left:12px;">◉ Demo</span>'
)
ocr_badge = (
    '<span class="badge badge-ocr">📸 OCR Active</span>'
    if OCR_AVAILABLE else
    '<span class="badge" style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.2);color:#f87171;">📸 OCR Offline</span>'
)
st.markdown(f"""
<div class="hero">
  <div class="hero-logo">🔍 Fake News Detection using Machine Learning</div>
  <p class="hero-sub">
    Analyze text or upload an image to detect misinformation instantly using AI &amp; NLP.
  </p>
  <div class="hero-badges">
    {mode_badge}
    {ocr_badge}
    <span class="badge badge-ai">🤖 NLP Powered</span>
    <span class="badge" style="background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.2);color:#fca5a5;">
      🛡️ Anti-Misinformation
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──
tab_text, tab_image = st.tabs(["💬  Text / Chat Analysis", "📸  Image Analysis (OCR)"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — TEXT CHAT
# ═══════════════════════════════════════════════════════════════════════════
with tab_text:

    # ── Render chat history ──
    chat_slot = st.empty()

    def render_chat():
        if not st.session_state.chat_history:
            chat_slot.markdown("""
            <div style="text-align:center;padding:52px 20px;color:#1f2937;">
              <div style="font-size:3rem;margin-bottom:10px;">💬</div>
              <div style="font-size:.95rem;font-weight:500;color:#4b5563;">
                Paste a news article below and click <strong>Analyze</strong>.
              </div>
              <div style="font-size:.8rem;color:#374151;margin-top:6px;">
                Or load a sample from the sidebar →
              </div>
            </div>""", unsafe_allow_html=True)
            return
        parts = ['<div class="chat-wrap">']
        for m in st.session_state.chat_history:
            if m["role"]=="user":
                disp = m["content"][:600]+"…" if len(m["content"])>600 else m["content"]
                disp = disp.replace("<","&lt;").replace(">","&gt;")
                parts.append(f"""
                <div class="msg-row user">
                  <div class="bubble bubble-user">{disp}</div>
                  <div class="av av-user">👤</div>
                </div>""")
            else:
                inner = m.get("html","") or m["content"]
                parts.append(f"""
                <div class="msg-row bot">
                  <div class="av av-bot">🔍</div>
                  <div class="bubble bubble-bot">{inner}</div>
                </div>""")
        parts.append("</div>")
        chat_slot.markdown("".join(parts), unsafe_allow_html=True)

    render_chat()
    st.markdown('<hr class="fhr">', unsafe_allow_html=True)

    # ── Input ──
    st.markdown('<div style="font-size:.75rem;color:#475569;text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px;font-weight:600;">📝 Enter News Article Text</div>', unsafe_allow_html=True)

    user_text = st.text_area(
        "news_text",
        value=st.session_state.input_text,
        placeholder=(
            "Paste or type a news article here…\n\n"
            "Example: 'According to the WHO report published today, scientists at…'"
        ),
        height=170,
        label_visibility="collapsed",
    )

    col_btn, col_clr, col_info = st.columns([2.5, 1, 2])
    with col_btn:
        clicked = st.button("🔍  Analyze Article", type="primary", use_container_width=True)
    with col_clr:
        if st.button("✕ Clear", use_container_width=True):
            st.session_state.input_text = ""
            st.rerun()
    with col_info:
        n = len(user_text)
        clr = "#22d3a5" if n>=50 else "#f5a623" if n>0 else "#374151"
        st.markdown(
            f'<div style="padding:9px 0;font-size:.78rem;color:{clr};text-align:right;">'
            f'{"✓" if n>=50 else "⚠"} {n} chars{"" if n>=50 else " — min 50 recommended"}</div>',
            unsafe_allow_html=True)

    if clicked:
        txt = user_text.strip()
        if not txt:
            st.warning("⚠️ Please enter some text first.")
        elif len(txt) < 15:
            st.warning("⚠️ Too short — please enter at least a sentence.")
        else:
            st.session_state.chat_history.append({"role":"user","content":txt})
            with st.spinner("🔍 Analyzing article…"):
                time.sleep(0.5)
                result = run_prediction(txt)
            html = verdict_html(result, mode_tag, "text")
            st.session_state.chat_history.append({"role":"bot","content":"","html":html})
            st.session_state.total += 1
            if result["label"]=="REAL": st.session_state.n_real += 1
            else: st.session_state.n_fake += 1
            st.session_state.input_text = ""
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE OCR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab_image:

    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(56,189,248,.08),rgba(155,114,247,.06));
      border:1px solid rgba(56,189,248,.2);border-radius:14px;
      padding:16px 20px;margin-bottom:20px;">
      <div style="font-size:.95rem;font-weight:600;color:#38bdf8;margin-bottom:4px;">
        📸 Image-Based Fake News Detection
      </div>
      <div style="font-size:.82rem;color:#64748b;line-height:1.6;">
        Upload a <strong style="color:#a5b4fc;">newspaper clipping</strong>,
        <strong style="color:#a5b4fc;">screenshot</strong>,
        <strong style="color:#a5b4fc;">WhatsApp forward</strong>, or any news image.
        Fake News Detection using Machine Learning will extract the text using OCR and instantly check for misinformation.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not OCR_AVAILABLE:
        st.error("""
        **📸 OCR engine not available.**
        Run the following to enable image analysis:
        ```
        pip install easyocr
        ```
        Then restart the app.
        """)
    else:
        uploaded = st.file_uploader(
            "Drop your image here or click to browse",
            type=["jpg","jpeg","png","bmp","tiff","webp"],
            label_visibility="visible",
            help="Supports: JPG, PNG, BMP, TIFF, WEBP — max 10 MB",
        )

        if uploaded:
            img = Image.open(uploaded)

            col_img, col_result = st.columns([1, 1.2], gap="large")

            with col_img:
                st.markdown('<div class="img-preview-box">', unsafe_allow_html=True)
                st.image(img, caption=f"📎 {uploaded.name}", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # image metadata
                w, h = img.size
                st.markdown(f"""
                <div style="font-size:.75rem;color:#475569;text-align:center;
                  background:var(--bg2);border:1px solid var(--border);
                  border-radius:8px;padding:8px;margin-top:-8px;">
                  📐 {w}×{h}px &nbsp;|&nbsp;
                  📁 {uploaded.type} &nbsp;|&nbsp;
                  💾 {uploaded.size/1024:.1f} KB
                </div>""", unsafe_allow_html=True)

            with col_result:
                analyze_img = st.button(
                    "🔍  Analyze Image",
                    type="primary",
                    use_container_width=True,
                    key="analyze_img_btn",
                )

                if analyze_img:
                    with st.spinner("📸 Extracting text via OCR…"):
                        extracted, ok = extract_text_from_image(img)

                    if not ok or len(extracted.strip()) < 15:
                        st.error("""
                        ❌ **Could not extract readable text from this image.**

                        **Tips for better results:**
                        - Use a clear, high-resolution image
                        - Ensure text is not rotated or blurry
                        - Screenshots work best; avoid artistic fonts
                        """)
                    else:
                        with st.spinner("🧠 Analyzing extracted text…"):
                            time.sleep(0.4)
                            result = run_prediction(extracted)

                        # Show extracted text
                        st.markdown('<div class="ocr-label">📄 Extracted Text (OCR)</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="ocr-box">{extracted[:1200]}{"…" if len(extracted)>1200 else ""}</div>',
                            unsafe_allow_html=True)

                        # Show verdict
                        st.markdown(
                            verdict_html(result, mode_tag, "image"),
                            unsafe_allow_html=True)

                        # Save to image history
                        import numpy as np
                        thumb_buf = io.BytesIO()
                        img.thumbnail((200, 200))
                        img.save(thumb_buf, format="PNG")

                        st.session_state.img_history.append({
                            "name"      : uploaded.name,
                            "extracted" : extracted,
                            "result"    : result,
                        })
                        st.session_state.img_total += 1
                        if result["label"]=="REAL": st.session_state.img_real += 1
                        else: st.session_state.img_fake += 1

                elif not analyze_img:
                    st.markdown("""
                    <div style="text-align:center;padding:32px 16px;color:#374151;">
                      <div style="font-size:2rem;margin-bottom:8px;">👆</div>
                      <div style="font-size:.85rem;color:#4b5563;">
                        Click <strong>Analyze Image</strong> to begin OCR extraction and detection.
                      </div>
                    </div>""", unsafe_allow_html=True)

        else:
            # Placeholder when no image uploaded
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;
              border:2px dashed rgba(155,114,247,.2);border-radius:18px;
              background:rgba(255,255,255,.01);margin-top:8px;">
              <div style="font-size:3.5rem;margin-bottom:12px;">📸</div>
              <div style="font-size:1rem;font-weight:600;color:#6b7a99;margin-bottom:6px;">
                Upload a News Image
              </div>
              <div style="font-size:.82rem;color:#374151;line-height:1.7;">
                Newspaper clipping &nbsp;·&nbsp; Screenshot &nbsp;·&nbsp; Social media post<br>
                WhatsApp forward &nbsp;·&nbsp; Web article snapshot
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Image History ──
        if st.session_state.img_history:
            st.markdown('<hr class="fhr">', unsafe_allow_html=True)
            st.markdown("#### 🕓 Image Analysis History")
            for i, h in enumerate(reversed(st.session_state.img_history)):
                lbl   = h["result"]["label"]
                conf  = h["result"]["confidence"]*100
                icon  = "✅" if lbl=="REAL" else "🚫"
                color = "#22d3a5" if lbl=="REAL" else "#f87171"
                with st.expander(f"{icon} {h['name']} — {lbl} ({conf:.0f}% confidence)"):
                    st.markdown(
                        f'<div style="font-size:.78rem;color:{color};font-weight:700;">'
                        f'{lbl} NEWS — {conf:.1f}% confidence</div>',
                        unsafe_allow_html=True)
                    st.text_area(
                        "Extracted Text",
                        h["extracted"][:800],
                        height=120,
                        disabled=True,
                        key=f"hist_{i}")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid rgba(255,255,255,.05);margin:28px 0 14px;">
<div style="text-align:center;font-size:.72rem;color:#1f2937;">
  🔍 <strong style="color:#4b5563;">Fake News Detection using Machine Learning</strong> &nbsp;—&nbsp;
  Powered by NLP &amp; Scikit-learn &nbsp;|&nbsp;
  Built with Python &amp; Streamlit &nbsp;|&nbsp;
  © 2025 Govardhan N
</div>""", unsafe_allow_html=True)
