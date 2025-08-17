import base64
import mimetypes
from io import BytesIO
import streamlit as st
from gtts import gTTS
import google.generativeai as genai

# ----- Page (no sidebar) -----
st.set_page_config(page_title="Smart Glasses Assistant", page_icon="🕶️", layout="centered")
st.markdown("""
<style>
section[data-testid="stSidebar"] { display:none !important; }
div[data-testid="stToolbar"] { display:none !important; }
</style>
""", unsafe_allow_html=True)
st.title("🕶️ Smart Glasses Assistant (Image + Voice → Auto-Speak)")


# ----- Mic components (optional) -----
_MIC = None
try:
    from streamlit_mic_recorder import mic_recorder as _mic1
    _MIC = ("smr", _mic1)
except Exception:
    try:
        from audio_recorder_streamlit import audio_recorder as _mic2
        _MIC = ("ars", _mic2)
    except Exception:
        _MIC = None

# ----- Helpers -----
def tts_bytes(text: str) -> bytes:
    buf = BytesIO()
    gTTS(text).write_to_fp(buf)
    buf.seek(0)
    return buf.read()

def guess_mime(filename: str, default="application/octet-stream") -> str:
    m, _ = mimetypes.guess_type(filename)
    return m or default

def speak_autoplay(mp3_bytes: bytes):
    """Autoplay audio via HTML; stops only if user presses Stop or sends a new query."""
    if not mp3_bytes: return
    b64 = base64.b64encode(mp3_bytes).decode()
    # unique id forces replay on rerun
    st.session_state["audio_counter"] = st.session_state.get("audio_counter", 0) + 1
    aid = f"tts_audio_{st.session_state['audio_counter']}"
    st.markdown(f"""
    <audio id="{aid}" autoplay>
      <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
    </audio>
    <script>
    const a = document.getElementById("{aid}");
    if (a) {{
      a.play().catch(()=>{{}});
    }}
    </script>
    """, unsafe_allow_html=True)

def generate_with_gemini(parts, model_name="gemini-2.5-flash", timeout=90) -> str:
    model = genai.GenerativeModel(model_name)
    try:
        resp = model.generate_content(parts, request_options={"timeout": timeout})
        txt = (getattr(resp, "text", "") or "").strip()
        if txt: return txt
        raise RuntimeError("Empty response text.")
    except Exception:
        has_audio = any(isinstance(p, dict) and str(p.get("mime_type","")).startswith("audio/") for p in parts)
        if has_audio and model_name != "gemini-1.5-flash":
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(parts, request_options={"timeout": timeout})
            txt = (getattr(resp, "text", "") or "").strip()
            if txt: return txt
        raise

# ----- UI inputs -----
img_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png", "webp"])
text_fallback = st.text_input("🔤 Optional text prompt (short)", "")

st.markdown("### 🎙️ Voice Prompt")
audio_bytes, audio_mime = None, None
if _MIC:
    impl, mic_fn = _MIC
    if impl == "smr":
        st.caption("Click **Record**, speak, then **Stop**.")
        obj = mic_fn(start_prompt="🎙️ Record", stop_prompt="⏹️ Stop", just_once=False, key="voice_rec")
        if obj and "bytes" in obj and obj["bytes"]:
            audio_bytes = obj["bytes"]; audio_mime = "audio/wav"; st.audio(audio_bytes, format="audio/wav")
    else:
        st.caption("Click the mic button to record.")
        audio_bytes = mic_fn(pause_threshold=2.0)
        if audio_bytes:
            audio_mime = "audio/wav"; st.audio(audio_bytes, format="audio/wav")
else:
    up = st.file_uploader("Or upload voice (WAV/MP3/OGG/WEBM/M4A)", type=["wav","mp3","ogg","webm","m4a"], key="au")
    if up:
        audio_bytes = up.read(); audio_mime = guess_mime(up.name, "audio/wav"); st.audio(audio_bytes, format=audio_mime)

col1, col2, col3 = st.columns([1,1,1])
with col1: analyze = st.button("🧠 Analyze & Speak")
with col2: stop = st.button("🛑 Stop")
with col3: replay = st.button("🔁 Speak Again")

# ----- State -----
st.session_state.setdefault("auto_speak", True)        # keep speaking by default
st.session_state.setdefault("last_mp3", b"")           # last audio bytes

# ----- Main -----
if analyze:
    # new query -> (re)enable auto speak
    st.session_state["auto_speak"] = True
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        st.error("`GOOGLE_API_KEY` missing in secrets.")
        st.stop()

    if not img_file:
        st.error("Upload an image."); st.stop()
    if not audio_bytes and not text_fallback.strip():
        st.error("Give a voice prompt or short text."); st.stop()

    genai.configure(api_key=api_key)

    STYLE_BRIEF = (
        "STYLE:\n"
        "- Default: ≤2 sentences or ≤40 words.\n"
        "- Expand ONLY if the user asks for more details.\n"
        "- ≤3 bullets if listing. No filler.\n"
        "- Ask ONE short clarifying question if the image is ambiguous.\n"
        "- Transcribe audio internally; don't show transcript unless asked."
    )
    system_context = f"You are an AI smart glasses assistant. Use the image and the user's voice/text to answer.\n{STYLE_BRIEF}"
    user_hint = f"\nAdditional user text: {text_fallback.strip()}" if text_fallback.strip() else ""
    final_prompt = f"{system_context}{user_hint}"

    parts = [final_prompt]
    img_bytes = img_file.read()
    img_mime = guess_mime(img_file.name, "image/jpeg")
    if not img_mime.startswith("image/"): st.error(f"Unsupported image type: {img_mime}"); st.stop()
    parts.append({"mime_type": img_mime, "data": img_bytes})
    if audio_bytes and audio_mime:
        parts.append({"mime_type": audio_mime, "data": audio_bytes})

    with st.spinner("Thinking..."):
        try:
            reply = generate_with_gemini(parts)
        except Exception as e:
            st.exception(e); st.stop()

    st.markdown("### 🧾 Response")
    st.write(reply or "(No text)")
    try:
        mp3 = tts_bytes(reply or "I could not generate a response.")
        st.session_state["last_mp3"] = mp3
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        st.session_state["last_mp3"] = b""

if stop:
    st.session_state["auto_speak"] = False

if replay and st.session_state["last_mp3"]:
    st.session_state["auto_speak"] = True  # allow replay

# Auto-speak if enabled and we have audio
if st.session_state["auto_speak"] and st.session_state["last_mp3"]:
    speak_autoplay(st.session_state["last_mp3"])

