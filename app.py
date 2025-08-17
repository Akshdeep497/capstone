import os
import mimetypes
from io import BytesIO

import streamlit as st
from gtts import gTTS
import google.generativeai as genai

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Mic imports (choose what's available) ----------
_MIC_IMPL = None
try:
    # pip install streamlit-mic-recorder
    from streamlit_mic_recorder import mic_recorder as _mic1
    _MIC_IMPL = ("smr", _mic1)
except Exception:
    try:
        # pip install audio-recorder-streamlit
        from audio_recorder_streamlit import audio_recorder as _mic2
        _MIC_IMPL = ("ars", _mic2)
    except Exception:
        _MIC_IMPL = None

# ---------- Helpers ----------
def get_api_key() -> str:
    """
    Fetch API key from (in order):
      1) Streamlit secrets
      2) Environment variable
      3) Session (from sidebar input)
    """
    key = None
    try:
        key = st.secrets.get("GOOGLE_API_KEY", None)
    except Exception:
        pass
    if not key:
        key = os.environ.get("GOOGLE_API_KEY", None)
    if not key:
        key = st.session_state.get("GOOGLE_API_KEY", "")
    return key or ""

def ensure_api_key() -> str:
    key = get_api_key()
    if not key:
        st.sidebar.warning("Set your Google API key to proceed.")
    return key

def tts_bytes(text: str) -> bytes:
    """Convert text to MP3 bytes using gTTS."""
    buf = BytesIO()
    gTTS(text).write_to_fp(buf)
    buf.seek(0)
    return buf.read()

def guess_mime(filename: str, default: str = "application/octet-stream") -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or default

def generate_with_gemini(parts, model_name: str = "gemini-2.5-flash", timeout: int = 90) -> str:
    """
    Try Gemini 2.5 Flash first; if audio present and it fails/empty, fall back to 1.5 Flash.
    """
    model = genai.GenerativeModel(model_name)
    try:
        resp = model.generate_content(parts, request_options={"timeout": timeout})
        text = (getattr(resp, "text", "") or "").strip()
        if text:
            return text
        raise RuntimeError("Empty response text from model.")
    except Exception as e:
        has_audio = any(isinstance(p, dict) and str(p.get("mime_type", "")).startswith("audio/") for p in parts)
        if has_audio and model_name != "gemini-1.5-flash":
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(parts, request_options={"timeout": timeout})
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                return text
        raise e

# ---------- UI ----------
st.set_page_config(page_title="Smart Glasses Assistant", page_icon="🕶️", layout="centered")
st.title("🕶️ Smart Glasses Assistant (Image + Voice → Spoken Answer)")

with st.sidebar:
    st.subheader("API Key")
    st.caption("Use `.streamlit/secrets.toml`, an env var, or paste below.")
    api_key_input = st.text_input("GOOGLE_API_KEY", type="password", placeholder="Paste key (not stored permanently)")
    if api_key_input:
        st.session_state["GOOGLE_API_KEY"] = api_key_input

    st.markdown("---")
    st.write("**Tips**")
    st.markdown(
        "- Prefer JPEG/PNG/WebP images\n"
        "- Voice: record via mic or upload WAV/MP3/OGG/WEBM/M4A\n"
        "- If no mic component is installed, use audio upload or only text"
    )

# Inputs
img_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png", "webp"])
text_fallback = st.text_input("🔤 Optional text prompt (context or fallback)", "")

st.markdown("### 🎙️ Voice Prompt")
audio_bytes, audio_mime = None, None

if _MIC_IMPL:
    impl, mic_fn = _MIC_IMPL
    if impl == "smr":
        st.caption("Click **Record**, speak, then **Stop**.")
        audio_obj = mic_fn(
            start_prompt="🎙️ Record",
            stop_prompt="⏹️ Stop",
            just_once=False,
            use_container_width=True,
            key="voice_rec"
        )
        if audio_obj and "bytes" in audio_obj and audio_obj["bytes"]:
            audio_bytes = audio_obj["bytes"]
            audio_mime = "audio/wav"  # streamlit-mic-recorder returns WAV bytes
            st.audio(audio_bytes, format="audio/wav")
    else:  # "ars"
        st.caption("Click the mic button to record.")
        audio_bytes = mic_fn(pause_threshold=2.0)  # returns WAV bytes
        if audio_bytes:
            audio_mime = "audio/wav"
            st.audio(audio_bytes, format="audio/wav")
else:
    st.info("Mic component not installed. Upload a voice file instead.")
    up = st.file_uploader(
        "Upload voice file (WAV/MP3/OGG/WEBM/M4A)",
        type=["wav", "mp3", "ogg", "webm", "m4a"],
        key="audio_upl"
    )
    if up is not None:
        audio_bytes = up.read()
        audio_mime = guess_mime(up.name, default="audio/wav")
        st.audio(audio_bytes, format=audio_mime)

go = st.button("🧠 Analyze & Speak")

# ---------- Main ----------
if go:
    key = ensure_api_key()
    if not key:
        st.error("Please provide a valid GOOGLE_API_KEY.")
        st.stop()

    if not img_file:
        st.error("Please upload an image.")
        st.stop()

    if not audio_bytes and not text_fallback.strip():
        st.error("Provide a voice prompt (record/upload) or enter a text prompt.")
        st.stop()

    # Configure SDK
    genai.configure(api_key=key)

    # Build system + user prompt
    system_context = (
        "You are an AI smart glasses assistant.\n"
        "1) If audio is provided, FIRST transcribe the user's voice question.\n"
        "2) Then answer concisely using the image and the question.\n"
        "3) If something is unclear, state assumptions briefly.\n"
        "Respond in at most two short paragraphs."
    )
    user_hint = f"\nAdditional user text: {text_fallback.strip()}" if text_fallback.strip() else ""
    final_prompt = f"{system_context}{user_hint}"

    parts = [final_prompt]

    # Image part
    img_bytes = img_file.read()
    img_mime = guess_mime(img_file.name, default="image/jpeg")
    if not img_mime.startswith("image/"):
        st.error(f"Unsupported image type: {img_mime}")
        st.stop()
    parts.append({"mime_type": img_mime, "data": img_bytes})

    # Audio part (optional)
    if audio_bytes and audio_mime:
        parts.append({"mime_type": audio_mime, "data": audio_bytes})

    with st.spinner("Thinking..."):
        try:
            reply_text = generate_with_gemini(parts)
        except Exception as e:
            st.exception(e)
            st.stop()

    st.markdown("### 🧾 Response")
    st.write(reply_text or "(No text returned)")

    # TTS
    try:
        mp3_data = tts_bytes(reply_text or "I could not generate a response.")
        st.markdown("### 🔊 Audio Answer")
        st.audio(mp3_data, format="audio/mp3")
        st.download_button("⬇️ Download MP3", data=mp3_data, file_name="response.mp3", mime="audio/mpeg")
        st.success("Done!")
    except Exception as e:
        st.warning(f"Text-to-speech failed: {e}")
