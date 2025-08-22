# app.py
import base64, json, mimetypes
from io import BytesIO

import streamlit as st
from gtts import gTTS
import google.generativeai as genai
import streamlit.components.v1 as components

# ===== Config =====
FIXED_PROMPT = "Describe what is in front of me in few words."

# ----- Page -----
st.set_page_config(page_title="Smart Glasses Assistant", page_icon="🕶️", layout="centered")
st.markdown("""
<style>
section[data-testid="stSidebar"]{display:none!important;}
div[data-testid="stToolbar"]{display:none!important;}
</style>
""", unsafe_allow_html=True)
st.title("🕶️ Smart Glasses Assistant (Camera/Upload + Voice → Auto-Speak)")

# ----- Helpers -----
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

# ----- State -----
st.session_state.setdefault("last_mp3", b"")
st.session_state.setdefault("auto_speak", True)

# ================= Manual Mode =================
st.header("Manual Mode")

img_file = st.file_uploader("📁 Upload image", type=["jpg","jpeg","png","webp"])
cam = st.camera_input("Or take a photo")
if cam is not None:
    img_file = cam

text_prompt = st.text_input("🔤 Optional text prompt (short)", "")

c1, c2, c3 = st.columns([1,1,1])
with c1: btn_analyze = st.button("🧠 Analyze & Speak")
with c2: btn_stop = st.button("🛑 Stop")
with c3: btn_replay = st.button("🔁 Speak Again")

if btn_analyze:
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Add GOOGLE_API_KEY to .streamlit/secrets.toml"); st.stop()
    genai.configure(api_key=api_key)

    if not img_file:
        st.error("Provide an image (upload or camera)."); st.stop()

    parts = [
        ("You are an AI smart glasses assistant. "
         "Default answers must be concise: ≤2 sentences (≤40 words). "
         "Expand ONLY if asked. Use ≤3 bullets when listing. "
         "Ask one short clarifying question only if the image is ambiguous.")
    ]
    if cam is not None:
        img_bytes, img_mime = cam.getvalue(), "image/jpeg"
    else:
        img_bytes, img_mime = img_file.read(), guess_mime(getattr(img_file, "name", "image.jpg"), "image/jpeg")

    parts.append({"mime_type": img_mime, "data": img_bytes})
    if text_prompt.strip():
        parts.append(f"Additional user text: {text_prompt.strip()}")

    with st.spinner("Thinking..."):
        reply = gen_gemini(parts) or "I couldn't generate a response."

    st.subheader("🧾 Response"); st.write(reply)
    st.session_state["last_mp3"] = tts_bytes(reply)

if btn_stop: st.session_state["auto_speak"] = False
if btn_replay and st.session_state["last_mp3"]: st.session_state["auto_speak"] = True
if st.session_state["auto_speak"] and st.session_state["last_mp3"]: speak_autoplay(st.session_state["last_mp3"])

# ================= Browser Hotword Mode (no WebRTC) =================
st.header("Browser Hotword Mode (no WebRTC)")
st.caption('Say **"capture"** to snap & describe. Chrome/Edge recommended.')

enable_hotword = st.toggle("Enable browser wake-word", value=False)
show_live = st.checkbox("Show live transcript", value=True)

if enable_hotword:
    live_div = "<div id='live' style='margin-top:6px;color:#bbb;font-family:monospace;white-space:pre-wrap;'></div>" if show_live else ""
    live_update = "const el=document.getElementById('live'); if(el) el.textContent = ('Final: '+lastFinal+'\\nInterim: '+interim);" if show_live else ""

    # HTML/JS with placeholders (avoid f-string brace issues)
    html_tpl = """
    <div style="display:flex;gap:10px;align-items:center;margin:8px 0;">
      <button id="startBtn" style="padding:6px 12px;border-radius:8px;">Start</button>
      <button id="stopBtn"  style="padding:6px 12px;border-radius:8px;">Stop</button>
      <span id="status" style="margin-left:8px;color:#aaa;">idle</span>
    </div>
    <video id="v" autoplay playsinline muted style="width:100%;max-width:640px;border-radius:10px;background:#111"></video>
    %%LIVE_DIV%%
    <script>
      const HOT = ['capture'];  // <-- single keyword
      function sendValue(val){window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setComponentValue',value:JSON.stringify(val)}, '*');}
      function norm(s){return (s||'').toLowerCase().replace(/[^a-z0-9 ]+/g,' ').trim();}

      // Camera
      const video = document.getElementById('v');
      let stream = null;
      async function startCam(){
        try{
          stream = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
          video.srcObject = stream;
        }catch(e){
          sendValue({event:'error', message:'camera error: '+e});
        }
      }

      // Speech
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      let recog = null; let listening=false; let lastFinal="";
      function startSR(){
        if(!SR){ sendValue({event:'error', message:'Web Speech API not supported'}); return; }
        recog = new SR();
        recog.continuous = true;
        recog.interimResults = true;
        recog.lang = 'en-US';
        recog.onstart = ()=>{listening=true; document.getElementById('status').textContent='listening…';};
        recog.onerror = e=>{sendValue({event:'error', message:'speech error: '+(e?.error||'unknown')});};
        recog.onend = ()=>{listening=false; document.getElementById('status').textContent='stopped';};
        recog.onresult = (ev)=>{
          let interim = '';
          for(let i=ev.resultIndex;i<ev.results.length;i++) {
            const t = ev.results[i][0].transcript;
            if(ev.results[i].isFinal) lastFinal += ' ' + t;
            else interim += t;
          }
          %%LIVE_UPDATE%%
          const test = norm((lastFinal + ' ' + interim));
          // exact keyword or word-boundary match
          const hit = HOT.some(hw => test.includes(hw)) || /\bcapture\b/.test(test);
          if(hit) takeAndSend();
        };
        try{recog.start();}catch(_){}
      }
      function stopSR(){
        if(recog) try{recog.stop();}catch(_){}
      }

      function takeAndSend(){
        if(!video.videoWidth) return;
        const c=document.createElement('canvas');
        c.width=video.videoWidth; c.height=video.videoHeight;
        const ctx=c.getContext('2d'); ctx.drawImage(video,0,0,c.width,c.height);
        const dataURL=c.toDataURL('image/jpeg',0.92);
        sendValue({event:'capture', image:dataURL});
        lastFinal='';
      }

      document.getElementById('startBtn').onclick=()=>{ startCam(); startSR(); };
      document.getElementById('stopBtn').onclick =()=>{ stopSR(); if(stream) stream.getTracks().forEach(t=>t.stop()); };
      // auto start
      startCam(); startSR();
    </script>
    """

    html = html_tpl.replace("%%LIVE_DIV%%", live_div).replace("%%LIVE_UPDATE%%", live_update)
    result = components.html(html, height=520 if show_live else 420, scrolling=False)

    if result:
        try:
            data = json.loads(result)
        except Exception:
            data = {}
        if data.get("event") == "error":
            st.warning(data.get("message","(unknown error)"))
        elif data.get("event") == "capture" and data.get("image"):
            # decode snapshot
            b64 = data["image"].split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
            st.image(img_bytes, caption="Snapshot", use_container_width=True)

            # always auto-speak for hotword captures
            st.session_state["auto_speak"] = True

            # Gemini with fixed prompt
            api_key = st.secrets.get("GOOGLE_API_KEY")
            if not api_key:
                st.error("Add GOOGLE_API_KEY to .streamlit/secrets.toml"); st.stop()
            genai.configure(api_key=api_key)

            parts = [FIXED_PROMPT, {"mime_type": "image/jpeg", "data": img_bytes}]
            with st.spinner("Analyzing snapshot..."):
                reply = gen_gemini(parts) or "I couldn't see enough to describe it."
            st.subheader("🧾 Response"); st.write(reply)

            # speak immediately
            try:
                mp3 = tts_bytes(reply)
                st.session_state["last_mp3"] = mp3
                speak_autoplay(mp3)
            except Exception as e:
                st.warning(f"TTS failed: {e}")
