# app.py
import base64, mimetypes, time, wave, re, threading
from collections import deque
from io import BytesIO

import numpy as np
from PIL import Image
import streamlit as st
from gtts import gTTS
import google.generativeai as genai

try:
    from streamlit_webrtc import webrtc_streamer
    _VOICE_OK = True
except Exception:
    _VOICE_OK = False

from streamlit_autorefresh import st_autorefresh

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
def tts_bytes(text: str) -> bytes:
    buf = BytesIO(); gTTS(text).write_to_fp(buf); buf.seek(0); return buf.read()

def guess_mime(name: str, default="application/octet-stream") -> str:
    m, _ = mimetypes.guess_type(name); return m or default

def speak_autoplay(mp3: bytes):
    if not mp3: return
    b64 = base64.b64encode(mp3).decode()
    st.session_state["audio_counter"] = st.session_state.get("audio_counter", 0) + 1
    aid = f"tts_{st.session_state['audio_counter']}"
    st.markdown(f"""
    <audio id="{aid}" autoplay>
      <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
    </audio>
    <script>document.getElementById("{aid}")?.play?.().catch(()=>{{}})</script>
    """, unsafe_allow_html=True)

def generate_with_gemini(parts, model="gemini-2.5-flash", timeout=90) -> str:
    m = genai.GenerativeModel(model)
    r = m.generate_content(parts, request_options={"timeout": timeout})
    t = (getattr(r, "text", "") or "").strip()
    if t: return t
    if any(isinstance(p, dict) and str(p.get("mime_type","")).startswith("audio/") for p in parts):
        m = genai.GenerativeModel("gemini-1.5-flash")
        r = m.generate_content(parts, request_options={"timeout": timeout})
        t = (getattr(r, "text", "") or "").strip()
    if not t: raise RuntimeError("Empty response text.")
    return t

def wav_from_int16(samples: np.ndarray, sr: int) -> bytes:
    bio = BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return bio.getvalue()

def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()

# ---------- State ----------
st.session_state.setdefault("auto_speak", True)
st.session_state.setdefault("last_mp3", b"")
st.session_state.setdefault("last_trigger_ts", 0.0)
st.session_state.setdefault("voice_mode", False)
st.session_state.setdefault("last_heard", "")
st.session_state.setdefault("last_rgb", None)

# ================= Manual Mode =================
st.header("Manual Mode")

img_file = st.file_uploader("📁 Upload image", type=["jpg","jpeg","png","webp"])
cam = st.camera_input("Or take a photo")
if cam is not None: img_file = cam

text_fallback = st.text_input("🔤 Optional text prompt (short)", "")

# Voice input (optional)
_MIC = None
try:
    from streamlit_mic_recorder import mic_recorder as _m1; _MIC = ("smr", _m1)
except Exception:
    try:
        from audio_recorder_streamlit import audio_recorder as _m2; _MIC = ("ars", _m2)
    except Exception:
        _MIC = None

st.markdown("### 🎙️ Voice Prompt")
audio_bytes, audio_mime = None, None
if _MIC:
    impl, mfn = _MIC
    if impl == "smr":
        st.caption("Click **Record**, speak, then **Stop**.")
        obj = mfn(start_prompt="🎙️ Record", stop_prompt="⏹️ Stop", just_once=False, key="vrec")
        if obj and obj.get("bytes"):
            audio_bytes = obj["bytes"]; audio_mime = "audio/wav"; st.audio(audio_bytes, format="audio/wav")
    else:
        st.caption("Click the mic button to record.")
        audio_bytes = mfn(pause_threshold=2.0)
        if audio_bytes: audio_mime = "audio/wav"; st.audio(audio_bytes, format="audio/wav")
else:
    up = st.file_uploader("Or upload voice (WAV/MP3/OGG/WEBM/M4A)", type=["wav","mp3","ogg","webm","m4a"], key="au")
    if up: audio_bytes = up.read(); audio_mime = guess_mime(up.name, "audio/wav"); st.audio(audio_bytes, format=audio_mime)

c1, c2, c3 = st.columns([1,1,1])
with c1: analyze = st.button("🧠 Analyze & Speak")
with c2: stop = st.button("🛑 Stop")
with c3: replay = st.button("🔁 Speak Again")

if analyze:
    st.session_state["auto_speak"] = True
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key: st.error("Add GOOGLE_API_KEY to .streamlit/secrets.toml"); st.stop()
    if not img_file: st.error("Provide an image (upload or camera)."); st.stop()
    if not audio_bytes and not text_fallback.strip(): st.error("Provide a voice prompt or short text."); st.stop()

    genai.configure(api_key=api_key)
    STYLE = ("STYLE:\n- ≤2 sentences or ≤40 words by default.\n- Expand only if asked.\n"
             "- ≤3 bullets if listing. No filler.\n- Ask one short clarifying question if image is ambiguous.")
    prompt = f"You are an AI smart glasses assistant.\n{STYLE}"
    if text_fallback.strip(): prompt += f"\nAdditional user text: {text_fallback.strip()}"

    parts = [prompt]
    if cam is not None: img_bytes = cam.getvalue(); img_mime = "image/jpeg"
    else: img_bytes = img_file.read(); img_mime = guess_mime(getattr(img_file, "name", "image.jpg"), "image/jpeg")
    if not img_mime.startswith("image/"): st.error(f"Unsupported image type: {img_mime}"); st.stop()
    parts.append({"mime_type": img_mime, "data": img_bytes})
    if audio_bytes and audio_mime: parts.append({"mime_type": audio_mime, "data": audio_bytes})

    with st.spinner("Thinking..."): reply = generate_with_gemini(parts)
    st.subheader("🧾 Response"); st.write(reply or "(No text)")
    st.session_state["last_mp3"] = tts_bytes(reply or "I could not generate a response.")

if stop: st.session_state["auto_speak"] = False
if replay and st.session_state["last_mp3"]: st.session_state["auto_speak"] = True
if st.session_state["auto_speak"] and st.session_state["last_mp3"]: speak_autoplay(st.session_state["last_mp3"])

# ================= Voice Capture Mode (hotword) =================
st.header("Voice Capture Mode (beta)")
st.caption('Say **"hey capture"** to snap & describe. Chrome recommended.')
st.session_state["voice_mode"] = st.toggle("Enable voice-triggered capture", value=st.session_state["voice_mode"])
show_heard = st.checkbox("Show heard words (debug)", value=True)

if not _VOICE_OK and st.session_state["voice_mode"]:
    st.info("Voice capture needs streamlit-webrtc + aiortc installed.")
elif _VOICE_OK and st.session_state["voice_mode"]:

    # ---- Thread-safe audio ring using new callbacks ----
    class AudioRing:
        def __init__(self):
            self.lock = threading.Lock()
            self.sample_rate = 16000
            self.buf = deque(maxlen=16000 * 6)   # ~6s
            self.frames = 0
            self.last_rms = 0.0
            self.last_chunk_len = 0
            self.last_pull_ts = 0.0
        def on_frame(self, frame):
            arr = frame.to_ndarray()
            ch = arr[0] if arr.ndim == 2 else arr
            if ch.dtype == np.int16: f = ch.astype(np.float32) / 32768.0
            elif ch.dtype == np.int32: f = ch.astype(np.float32) / 2147483648.0
            else: f = ch.astype(np.float32)
            sr = getattr(frame, "sample_rate", None) or self.sample_rate
            i16 = np.clip(f * 32767.0, -32768, 32767).astype(np.int16)
            with self.lock:
                self.sample_rate = sr
                self.last_rms = float(np.sqrt(np.mean(f * f) + 1e-12))
                self.frames += 1
                self.buf.extend(i16.tolist())
        def get_recent(self, seconds=2.0):
            now = time.time()
            with self.lock:
                if now - self.last_pull_ts < 1.2: return None, None
                if len(self.buf) < int(0.4 * self.sample_rate): return None, None
                n = int(seconds * self.sample_rate)
                data = list(self.buf)[-n:]
                self.last_pull_ts = now
                self.last_chunk_len = len(data)
            return np.array(data, dtype=np.int16), self.sample_rate

    if "audioring" not in st.session_state: st.session_state["audioring"] = AudioRing()
    ring: AudioRing = st.session_state["audioring"]

    def video_cb(frame):
        # store latest RGB for snapshot; show original frame
        bgr = frame.to_ndarray(format="bgr24")
        st.session_state["last_rgb"] = bgr[:, :, ::-1]
        return frame

    def audio_cb(frame):
        ring.on_frame(frame)
        return frame

    # ---- ICE ----
    stun = ["stun:stun.l.google.com:19302","stun:stun1.l.google.com:19302","stun:stun2.l.google.com:19302"]
    turn_urls = [u.strip() for u in st.secrets.get("TURN_URLS", "").split(",") if u.strip()]
    turn_user = st.secrets.get("TURN_USERNAME", ""); turn_pass = st.secrets.get("TURN_PASSWORD", "")
    HAS_TURN = bool(turn_urls and turn_user and turn_pass)
    force_turn = st.checkbox("Force TURN (relay only)", value=HAS_TURN, disabled=not HAS_TURN)

    ice_servers = [{"urls": stun}]
    if HAS_TURN:
        ice_servers.append({"urls": turn_urls, "username": turn_user, "credential": turn_pass})
    rtc_config = {"iceServers": ice_servers}
    if HAS_TURN and force_turn: rtc_config["iceTransportPolicy"] = "relay"

    ctx = webrtc_streamer(
        key="voice-cam",
        video_frame_callback=video_cb,      # new callback API
        audio_frame_callback=audio_cb,      # new callback API (reliable audio)
        media_stream_constraints={"video": True, "audio": True},
        rtc_configuration=rtc_config,
        async_processing=True,
        sendback_audio=False,
    )

    if ctx.state.playing:
        st_autorefresh(interval=1200, key="vc_poll2")

        if show_heard:
            st.caption(f"Frames: {ring.frames} | RMS: {ring.last_rms:.4f} | chunk: {ring.last_chunk_len}")

        # no mic track at all
        if ring.frames == 0:
            st.error("No microphone frames received. Click the lock icon → Microphone: Allow, then reload.")
        else:
            samples, sr = ring.get_recent(2.0)
            if samples is not None and (ring.last_rms > 0.001 or show_heard):
                api_key = st.secrets.get("GOOGLE_API_KEY")
                if not api_key: st.error("Add GOOGLE_API_KEY to .streamlit/secrets.toml"); st.stop()
                genai.configure(api_key=api_key)

                wav_bytes = wav_from_int16(samples, sr)
                parts_t = [
                    "transcribe this audio; return the exact words in lowercase.",
                    {"mime_type": "audio/wav", "data": wav_bytes},
                ]
                try: transcript = generate_with_gemini(parts_t, model="gemini-1.5-flash")
                except Exception: transcript = ""
                phrase = norm_text(transcript)
                st.session_state["last_heard"] = phrase
                if show_heard: st.caption("Heard: " + phrase)

                HOT = ["hey capture","capture","take picture","hey picture","hey capture now"]
                if any(h in phrase for h in HOT) and (time.time() - st.session_state["last_trigger_ts"] > 2.5):
                    st.session_state["last_trigger_ts"] = time.time()
                    if st.session_state["last_rgb"] is not None:
                        buf = BytesIO(); Image.fromarray(st.session_state["last_rgb"]).save(buf, format="JPEG", quality=90)
                        shot = buf.getvalue()
                        parts2 = ["Describe what is in front of me in few words.", {"mime_type": "image/jpeg", "data": shot}]
                        with st.spinner("Capturing & describing..."):
                            try: reply = generate_with_gemini(parts2)
                            except Exception as e: st.exception(e); reply = ""
                        st.subheader("🧾 Voice Capture Response"); st.write(reply or "(No text)")
                        try:
                            mp3 = tts_bytes(reply or "I could not generate a response.")
                            st.session_state["last_mp3"] = mp3
                            if st.session_state.get("auto_speak", True): speak_autoplay(mp3)
                        except Exception as e:
                            st.warning(f"TTS failed: {e}")
