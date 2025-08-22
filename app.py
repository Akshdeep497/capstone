# app.py
import base64, mimetypes
from io import BytesIO
import streamlit as st
from gtts import gTTS
from PIL import Image
import google.generativeai as genai

# ---------- Page ----------
st.set_page_config(page_title="Smart Glasses Assistant", page_icon="🕶️", layout="centered")
st.title("🕶️ Smart Glasses Assistant")

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

def gen_gemini(parts, model="gemini-2.5-flash") -> str:
    m = genai.GenerativeModel(model)
    r = m.generate_content(parts)
    return (getattr(r, "text", "") or "").strip()

# ---------- State ----------
st.session_state.setdefault("last_mp3", b"")
st.session_state.setdefault("auto_speak", True)

# ================= Manual Capture =================
st.header("Manual Mode (Upload/Camera/Keypress)")

# File or camera
uploaded = st.file_uploader("📁 Upload image", type=["jpg","jpeg","png","webp"])
captured = st.camera_input("📸 Or take a photo")

if captured is not None:
    img_file = captured
else:
    img_file = uploaded

text_prompt = st.text_input("Optional extra text", "")

# Buttons
col1, col2, col3 = st.columns([1,1,1])
with col1: btn_analyze = st.button("🧠 Analyze & Speak")
with col2: btn_stop = st.button("🛑 Stop Audio")
with col3: btn_replay = st.button("🔁 Replay")

# Keyboard shortcut 'c'
st.markdown("""
<script>
document.addEventListener("keydown", function(e){
  if(e.key === "c" || e.key === "C"){
    const btns = window.parent.document.querySelectorAll('button');
    for (let b of btns){ if(b.innerText.includes("Analyze")){ b.click(); break; } }
  }
});
</script>
""", unsafe_allow_html=True)

if btn_analyze:
    if not img_file:
        st.error("Please upload or capture an image.")
    else:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("Add GOOGLE_API_KEY to .streamlit/secrets.toml")
            st.stop()
        genai.configure(api_key=api_key)

        # Read image
        if captured is not None:
            img_bytes, img_mime = captured.getvalue(), "image/jpeg"
        else:
            img_bytes, img_mime = img_file.read(), guess_mime(img_file.name, "image/jpeg")

        parts = [
            "Describe briefly what is in front of me (≤2 sentences).",
            {"mime_type": img_mime, "data": img_bytes}
        ]
        if text_prompt.strip():
            parts.append(f"User note: {text_prompt.strip()}")

        with st.spinner("Analyzing..."):
            reply = gen_gemini(parts) or "I couldn't generate a response."

        st.subheader("🧾 Response")
        st.write(reply)

        mp3 = tts_bytes(reply)
        st.session_state["last_mp3"] = mp3
        if st.session_state.get("auto_speak", True):
            speak_autoplay(mp3)

if btn_stop:
    st.session_state["auto_speak"] = False
if btn_replay and st.session_state["last_mp3"]:
    st.session_state["auto_speak"] = True
    speak_autoplay(st.session_state["last_mp3"])
