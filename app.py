# app.py
import base64, json
from io import BytesIO
import mimetypes

import streamlit as st
import google.generativeai as genai
from gtts import gTTS

# ===== Config =====
FIXED_PROMPT = "Describe what is in front of me in few words."

st.set_page_config(page_title="Smart Glasses Assistant", page_icon="🕶️")
st.title("🕶️ Smart Glasses Assistant")

# ===== Helpers =====
def guess_mime(name: str, default="application/octet-stream") -> str:
    m, _ = mimetypes.guess_type(name); return m or default

def tts_bytes(text: str) -> bytes:
    buf = BytesIO(); gTTS(text).write_to_fp(buf); buf.seek(0); return buf.read()

def speak_autoplay(mp3: bytes):
    b64 = base64.b64encode(mp3).decode()
    st.markdown(f"""
    <audio autoplay>
      <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
    </audio>
    """, unsafe_allow_html=True)

def gen_gemini(parts):
    api_key = st.secrets.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel("gemini-2.5-flash")
    r = m.generate_content(parts, request_options={"timeout": 90})
    return (getattr(r, "text", "") or "").strip()

def run_analyze(img_bytes: bytes, img_mime: str):
    parts = [
        "You are an AI smart glasses assistant. "
        "Keep answers ≤2 sentences (≤40 words). No filler."
    ]
    parts.append({"mime_type": img_mime, "data": img_bytes})
    reply = gen_gemini(parts) or "I couldn't see enough."
    st.subheader("🧾 Response")
    st.write(reply)
    mp3 = tts_bytes(reply); speak_autoplay(mp3)

# ===== Hotword Mode =====
st.header("Browser Hotword Mode")
st.caption('Say **"capture"** → snapshot → auto analyze & speak.')

# Bridge input (hidden)
payload = st.text_area("bridge", key="bridge", label_visibility="collapsed")

# Inject JS: speech+camera → put JSON into hidden input
st.markdown(f"""
<video id="v" autoplay playsinline muted style="width:100%;max-width:400px;border:1px solid #555"></video>
<div id="status" style="color:#aaa;margin:6px 0;">idle</div>
<button onclick="takeAndSend()">Snap</button>
<script>
  const video = document.getElementById('v');
  navigator.mediaDevices.getUserMedia({{video:true}}).then(s=>video.srcObject=s);

  const SR = window.SpeechRecognition||window.webkitSpeechRecognition;
  let recog=new SR(); recog.continuous=true; recog.interimResults=true; recog.lang='en-US';
  recog.onresult=e=>{let t="";for(let i=e.resultIndex;i<e.results.length;i++)t+=e.results[i][0].transcript;
                     if(/\\bcapture\\b/i.test(t)) takeAndSend();};
  recog.start();

  function takeAndSend(){{
    if(!video.videoWidth) return;
    let c=document.createElement('canvas'); c.width=320; c.height=240;
    c.getContext('2d').drawImage(video,0,0,320,240);
    let dataURL=c.toDataURL('image/jpeg',0.5);
    let obj={{event:'capture', ts:Date.now(), image:dataURL, prompt:{json.dumps(FIXED_PROMPT)}}};
    let el=window.parent.document.querySelector('textarea#bridge');
    if(el){{el.value=JSON.stringify(obj); el.dispatchEvent(new Event('input',{{bubbles:true}}));}}
    document.getElementById('status').textContent="sent!";
  }}
</script>
""", unsafe_allow_html=True)

# ===== Consume payload =====
if payload:
    try:
        data = json.loads(payload)
    except Exception:
        data = {}
    if data.get("event") == "capture":
        b64 = data["image"].split(",",1)[1]
        img_bytes = base64.b64decode(b64)
        st.image(img_bytes, caption="Snapshot", use_container_width=True)
        run_analyze(img_bytes, "image/jpeg")
