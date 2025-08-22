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
        import base64, json
        try:
            dataURL = json.loads(result)
            img_bytes = base64.b64decode(dataURL.split(",",1)[1])
        except:
            img_bytes = None

        if img_bytes:
            api_key = st.secrets.get("GOOGLE_API_KEY")
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
