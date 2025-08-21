import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import numpy as np
from PIL import Image
import io, os, base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from streamlit_autorefresh import st_autorefresh

# --- Config ---
st.set_page_config(page_title="Voice + Image Assistant", layout="centered")
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- Speak helper ---
def speak_text(text):
    tts = gTTS(text)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    st.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )

# --- Describe image ---
def describe_image(pil_img):
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(
        ["Describe what is in front of me in few words.", pil_img]
    )
    return resp.text.strip()

# --- Webcam capture transformer ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = Image.fromarray(img)
        return frame

# --- UI ---
st.title("📷 Voice + Image Assistant")
mode = st.radio("Choose mode:", ["Normal", "Voice Capture Mode (beta)"])

# ----------------- NORMAL MODE -----------------
if mode == "Normal":
    img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    text_in = st.text_input("Ask a question:")
    if st.button("Submit"):
        if img_file and text_in:
            pil_img = Image.open(img_file).convert("RGB")
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content([text_in, pil_img])
            st.write(resp.text)
            speak_text(resp.text)

# ----------------- VOICE MODE -----------------
else:
    st.markdown("Say **'hey capture'** to snap & describe. Chrome recommended.")

    st.session_state.setdefault("voice_mode", False)
    enable_voice = st.toggle("Enable voice-triggered capture", key="voice_mode")

    ctx = webrtc_streamer(key="vcam", video_transformer_factory=VideoTransformer)

    if ctx.video_transformer:
        # Poll every 1.5s
        st_autorefresh(interval=1500, key="vc_poll")

        if enable_voice and ctx.video_transformer.latest_frame:
            # Here you’d normally detect "hey capture" from mic
            # For demo: we trigger capture automatically
            pil_img = ctx.video_transformer.latest_frame
            st.image(pil_img, caption="Captured Frame", use_container_width=True)
            desc = describe_image(pil_img)
            st.success(desc)
            speak_text(desc)
