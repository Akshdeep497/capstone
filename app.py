# app.py
import base64, json, mimetypes
from io import BytesIO

import streamlit as st
from PIL import Image
from gtts import gTTS
import google.generativeai as genai
import streamlit.components.v1 as components

# ---------- Page ----------
st.set_page_config(page_title="Smart Glasses Assistant", page_icon="🕶️", layout="centered")
st.markdown("""
<style>
section[data-testid="stSidebar"]{display:none!important;}
div[data-testid="stToolbar"]{display:none!important;}
</style>
""", unsafe_allow_html=True)
st.title("🕶️ Smart Glasses Assistant (Camera/Upload + Voice → Auto-Speak)")

# ---------- Helpers ----------
def guess_mime(name: str, default="application/octet-stream") -> str:
    m, _ = mimetypes.guess_type(name); return m or default

def tts_bytes(text: str) -> bytes:
    buf = BytesIO(); gTTS(text).write_to_fp(buf); buf.seek(0); return buf.read()

def speak_autoplay(mp3_bytes: bytes):
    if not mp3_bytes: return
    b64 = base64.b64encode(mp3_bytes).decode()
    st.session_state["audio_counter"] = st.session_state.get("audio_counter", 0) + 1
    aid = f"tts_{st.session_state['audio_counter']}"
    st.markdown(f"""
    <audio id="{aid}" autoplay>
      <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
    </audio>
    <script>document.getElementById("{aid}")?.play?.().catch(()=>{{}})</script>
    """, unsafe_allow_html=True)

def gen_gemini(parts, model="gemini-2.5-flash", timeout=90) -> str:
    m = genai.GenerativeModel(model)
    r = m.generate_content(parts, request_options={"timeout": timeout})
    return (getattr(r, "text", "") or "").strip()

# ---------- State ----------
st.session_state.setdefault("last_mp3", b"")
st.session_state.setdefault("auto_speak", True)

# ================= Manual Mode =================
st.header("Manual Mode")

# Camera element always active
html_cam = """
<video id="manualCam" autoplay playsinline muted style="width:100%;max-width:480px;border-radius:10px;background:#111"></video>
<script>
  const vid = document.getElementById("manualCam");
  navigator.mediaDevices.getUserMedia({video:true,audio:false}).then(s=>{vid.srcObject=s;});
  function snapManual(){
    if(!vid.videoWidth) return null;
    const c=document.createElement("canvas");
    c.width=vid.videoWidth; c.height=vid.videoHeight;
    c.getContext("2d").drawImage(vid,0,0,c.width,c.height);
    return c.toDataURL("image/jpeg",0.92);
  }
  window.snapManual = snapManual;
</script>
"""
components.html(html_cam, height=360)

text_prompt = st.text_input("🔤 Optional text prompt (short)", "")
btn_analyze = st.button("📸 Analyze Camera & Speak")

if btn_analyze:
    # Grab snapshot from camera
    js = """
    <script>
    const dataURL = window.snapManual ? window.snapManual() : null;
    if(dataURL){window.parent.postMessage(
      {isStreamlitMessage:true,type:'streamlit:setComponentValue',value:dataURL}, '*');}
    </script>
    """
    result = components.html(js, height=0)
    if result:
        try:
            dataURL = json.loads(result)
            img_bytes = base64.b64decode(dataURL.split(",",1)[1])
        except:
            img_bytes = None

        if img_bytes:
            api_key = st.secrets.get("GOOGLE_API_KEY")
            if not api_key:
                st.error("Add GOOGLE_API_KEY to .streamlit/secrets.toml")
                st.stop()
            genai.configure(api_key=api_key)

            parts = [
                "You are an AI smart glasses assistant. Describe in ≤2 sentences.",
                {"mime_type":"image/jpeg","data": img_bytes}
            ]
            if text_prompt.strip():
                parts.append(f"User note: {text_prompt.strip()}")

            with st.spinner("Analyzing snapshot..."):
                reply = gen_gemini(parts) or "I couldn't generate a response."

            st.subheader("🧾 Response"); st.write(reply)
            mp3 = tts_bytes(reply)
            st.session_state["last_mp3"] = mp3
            speak_autoplay(mp3)
