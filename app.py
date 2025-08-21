import base64, mimetypes, time, wave
from collections import deque
from io import BytesIO

import numpy as np
from PIL import Image
import streamlit as st
from gtts import gTTS
import google.generativeai as genai

# Optional WebRTC (video+audio) for voice-triggered capture
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, AudioProcessorBase
    _VOICE_CAPTURE_AVAILABLE = True
except Exception:
    _VOICE_CAPTURE_AVAILABLE = False

from streamlit_autorefresh import st_autorefresh

# ---------- Page (no sidebar) ----------
st.set_page_config(page_title="Smart Glasses Assistant", page_icon="🕶️", layout="centered")
st.markdown("""
<style>
section[data-testid="stSidebar"] { display:none !important; }
div[data-testid="stToolbar"] { display:none !important; }
</style>
""", unsafe_allow_html=True)
st.title("🕶️ Smart Glasses Assistant (Camera/Upload + Voice → Auto-Speak)")

# ---------- Helpers ----------
def tts_bytes(text: str) -> bytes:
    buf = BytesIO()
    gTTS(text).write_to_fp(buf)
    buf.seek(0)
    return buf.read()

def guess_mime(name: str, default="application/octet-stream") -> str:
    m, _ = mimetypes.guess_type(name)
    return m or default

def speak_autoplay(mp3_bytes: bytes):
    if not mp3_bytes:
        return
    b64 = base64.b64encode(mp3_bytes).decode()
    st.session_state["audio_counter"] = st.session_state.get("audio_counter", 0) + 1
    aid = f"tts_audio_{st.session_state['audio_counter']}"
    st.markdown(f"""
    <audio id="{aid}" autoplay>
      <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
    </audio>
    <script>const a=document.getElementById("{aid}"); if(a){{a.play().catch(()=>{{}});}}</script>
    """, unsafe_allow_html=True)

def generate_with_gemini(parts, model_name="gemini-2.5-flash", timeout=90) -> str:
    model = genai.GenerativeModel(model_name)
    try:
        resp = model.generate_content(parts, request_options={"timeout": timeout})
        txt = (getattr(resp, "text", "") or "").strip()
        if txt:
            return txt
        raise RuntimeError("Empty response text.")
    except Exception:
        has_audio = any(isinstance(p, dict) and str(p.get("mime_type","")).startswith("audio/") for p in parts)
        if has_audio and model_name != "gemini-1.5-flash":
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(parts, request_options={"timeout": timeout})
            txt = (getattr(resp, "text", "") or "").strip()
            if txt:
                return txt
        raise

def wav_from_int16_mono(samples: np.ndarray, sample_rate: int) -> bytes:
    bio = BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return bio.getvalue()

# ---------- State ----------
st.session_state.setdefault("auto_speak", True)
st.session_state.setdefault("last_mp3", b"")
st.session_state.setdefault("last_trigger_ts", 0.0)
st.session_state.setdefault("voice_mode", False)

# ================= Manual Mode =================
st.header("Manual Mode")

# Image: upload or one-shot camera
img_file = st.file_uploader("📁 Upload image", type=["jpg", "jpeg", "png", "webp"])
cam = st.camera_input("Or take a photo")
if cam is not None:
    img_file = cam

# Optional text
text_fallback = st.text_input("🔤 Optional text prompt (short)", "")

# Voice (record/upload)
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

c1, c2, c3 = st.columns([1, 1, 1])
with c1: analyze = st.button("🧠 Analyze & Speak")
with c2: stop    = st.button("🛑 Stop")
with c3: replay  = st.button("🔁 Speak Again")

if analyze:
    st.session_state["auto_speak"] = True
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        st.error("Add GOOGLE_API_KEY to .streamlit/secrets.toml"); st.stop()

    if not img_file:
        st.error("Provide an image (upload or camera)."); st.stop()
    if not audio_bytes and not text_fallback.strip():
        st.error("Provide a voice prompt or short text."); st.stop()

    genai.configure(api_key=api_key)

    STYLE_BRIEF = (
        "STYLE:\n- Default: ≤2 sentences or ≤40 words.\n"
        "- Expand ONLY if user asks for more.\n"
        "- ≤3 bullets if listing. No filler.\n"
        "- Ask ONE short clarifying question if image is ambiguous.\n"
        "- Transcribe audio internally; don't show transcript unless asked."
    )
    system_context = f"You are an AI smart glasses assistant. Use the image and the user's voice/text to answer.\n{STYLE_BRIEF}"
    user_hint = f"\nAdditional user text: {text_fallback.strip()}" if text_fallback.strip() else ""
    final_prompt = f"{system_context}{user_hint}"

    parts = [final_prompt]
    if cam is not None:
        img_bytes = cam.getvalue(); img_mime = "image/jpeg"
    else:
        img_bytes = img_file.read()
        img_mime = guess_mime(getattr(img_file, "name", "image.jpg"), "image/jpeg")
    if not img_mime.startswith("image/"): st.error(f"Unsupported image type: {img_mime}"); st.stop()
    parts.append({"mime_type": img_mime, "data": img_bytes})
    if audio_bytes and audio_mime:
        parts.append({"mime_type": audio_mime, "data": audio_bytes})

    with st.spinner("Thinking..."):
        try:
            reply = generate_with_gemini(parts)
        except Exception as e:
            st.exception(e); st.stop()

    st.subheader("🧾 Response")
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
    st.session_state["auto_speak"] = True
if st.session_state["auto_speak"] and st.session_state["last_mp3"]:
    speak_autoplay(st.session_state["last_mp3"])

# ================= Voice Capture Mode (say "hey capture") =================
st.header("Voice Capture Mode (beta)")
st.caption('Say **"hey capture"** to snap & describe. Chrome recommended.')
st.session_state["voice_mode"] = st.toggle(
    "Enable voice-triggered capture", value=st.session_state["voice_mode"]
)

if not _VOICE_CAPTURE_AVAILABLE and st.session_state["voice_mode"]:
    st.info("Voice capture requires: streamlit-webrtc and aiortc. Install deps and restart.")
elif _VOICE_CAPTURE_AVAILABLE and st.session_state["voice_mode"]:

    class FrameGrabber(VideoTransformerBase):
        def __init__(self):
            self.last_rgb = None
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")[:, :, ::-1]  # BGR->RGB
            self.last_rgb = img
            return frame  # pass-through

    class VADBuffer(AudioProcessorBase):
        def __init__(self):
            self.sample_rate = 16000
            self.buffer = deque(maxlen=16000 * 4)  # ~4s
            self.last_rms = 0.0
            self.last_pull_ts = 0.0
        def recv(self, frame):
            arr = frame.to_ndarray()
            ch = arr[0] if arr.ndim == 2 else arr
            # normalize
            if ch.dtype == np.int16:
                f = ch.astype(np.float32) / 32768.0
            elif ch.dtype == np.int32:
                f = ch.astype(np.float32) / 2147483648.0
            else:
                f = ch.astype(np.float32)
            self.sample_rate = frame.sample_rate or 16000
            self.last_rms = float(np.sqrt(np.mean(f * f) + 1e-9))
            i16 = np.clip(f * 32767.0, -32768, 32767).astype(np.int16)
            self.buffer.extend(i16.tolist())
            return frame
        def get_recent_chunk(self, seconds: float = 1.6):
            now = time.time()
            if now - self.last_pull_ts < 2.0:
                return None, None
            n = int(seconds * self.sample_rate)
            if len(self.buffer) < int(0.6 * self.sample_rate):
                return None, None
            data = list(self.buffer)[-n:]
            self.last_pull_ts = now
            return np.array(data, dtype=np.int16), self.sample_rate

    webrtc_ctx = webrtc_streamer(
        key="voice-cam",
        video_transformer_factory=FrameGrabber,
        audio_processor_factory=VADBuffer,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )

    # Poll while playing
    if webrtc_ctx.state.playing:
        st_autorefresh(interval=1500, key="vc_poll")

        vad = webrtc_ctx.audio_processor
        grab = webrtc_ctx.video_transformer

        if vad is not None and grab is not None and vad.last_rms > 0.02:
            samples, sr = vad.get_recent_chunk(1.6)
            if samples is not None:
                # Transcribe small chunk; look for hotword
                try:
                    api_key = st.secrets["GOOGLE_API_KEY"]
                except Exception:
                    st.error("Add GOOGLE_API_KEY to .streamlit/secrets.toml"); st.stop()
                genai.configure(api_key=api_key)

                wav_bytes = wav_from_int16_mono(samples, sr)
                parts_t = [
                    "transcribe this audio; return the exact words in lowercase.",
                    {"mime_type": "audio/wav", "data": wav_bytes},
                ]
                try:
                    transcript = generate_with_gemini(parts_t, model_name="gemini-1.5-flash")
                except Exception:
                    transcript = ""
                phrase = (transcript or "").strip().lower()

                # Trigger on "hey capture"
                now = time.time()
                if "hey capture" in phrase and (now - st.session_state["last_trigger_ts"] > 2.5):
                    st.session_state["last_trigger_ts"] = now
                    if getattr(grab, "last_rgb", None) is not None:
                        buf = BytesIO()
                        Image.fromarray(grab.last_rgb).save(buf, format="JPEG", quality=90)
                        shot_bytes = buf.getvalue()

                        fixed_prompt = "Describe what is in front of me in few words."
                        parts2 = [fixed_prompt, {"mime_type": "image/jpeg", "data": shot_bytes}]
                        with st.spinner("Capturing & describing..."):
                            try:
                                reply = generate_with_gemini(parts2)
                            except Exception as e:
                                st.exception(e); reply = ""

                        st.subheader("🧾 Voice Capture Response")
                        st.write(reply or "(No text)")
                        try:
                            mp3 = tts_bytes(reply or "I could not generate a response.")
                            st.session_state["last_mp3"] = mp3
                            if st.session_state.get("auto_speak", True):
                                speak_autoplay(mp3)
                        except Exception as e:
                            st.warning(f"TTS failed: {e}")
                    else:
                        st.warning("Webcam not ready yet. Please allow camera and wait a moment.")
