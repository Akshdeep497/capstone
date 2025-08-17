# requirements.txt:
# streamlit
# google-generativeai
# gtts
# Pillow
# streamlit-audiorec
# SpeechRecognition
# pydub

import streamlit as st
from PIL import Image
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
from streamlit_audiorec import st_audiorec
import os
import io

# --- App Configuration ---
st.set_page_config(
    page_title="Vision & Voice AI Assistant",
    page_icon="👓",
    layout="wide",
)

st.title("Vision & Voice AI Assistant 👓🎙️")
st.markdown("This app functions like AI smart glasses. Upload an image, ask a question with your voice, and get a spoken answer.")

# --- API Key Configuration ---
# For security, it's best to use Streamlit's secrets management
try:
    # Attempt to get the API key from Streamlit secrets
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=API_KEY)
except (KeyError, FileNotFoundError):
    st.sidebar.warning("API key not found in Streamlit secrets.")
    # Fallback to a text input if not found in secrets
    api_key_input = st.sidebar.text_input("Enter your Google API Key:", type="password")
    if api_key_input:
        genai.configure(api_key=api_key_input)
    else:
        st.error("Please provide your Google API Key in the sidebar to proceed.")
        st.stop()


# --- Function to Transcribe Audio ---
def transcribe_audio(audio_bytes):
    """
    Transcribes audio bytes into text using Google's Web Speech API.
    """
    recognizer = sr.Recognizer()
    try:
        # Convert audio bytes to AudioData
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        
        # Recognize speech using Google Web Speech API
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.warning("Could not understand the audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during audio transcription: {e}")
        return None

# --- Main Application Logic ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    image_display_area = st.empty()
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        image_display_area.image(image, caption="Uploaded Image", use_column_width=True)
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        image_bytes = img_byte_arr.getvalue()
    else:
        image_display_area.info("Please upload an image to get started.")


with col2:
    st.header("2. Record Your Question")
    
    # Audio recorder component
    wav_audio_data = st_audiorec()

    transcribed_text = ""
    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        # Transcribe the recorded audio
        with st.spinner("Transcribing your voice..."):
            transcribed_text = transcribe_audio(wav_audio_data)
            if transcribed_text:
                st.success(f"**Your Question:** \"{transcribed_text}\"")
            else:
                st.warning("Transcription failed. Please try recording again.")

st.divider()

# --- Generate Response ---
if st.button("Get Answer", disabled=(uploaded_file is None or not transcribed_text), use_container_width=True):
    if uploaded_file and transcribed_text:
        with st.spinner("Analyzing the image and generating a response..."):
            try:
                # Initialize the generative model
                # Use 'gemini-pro-vision' for tasks involving images
                model = genai.GenerativeModel('gemini-pro-vision')

                # Define the context prompt
                context_prompt = "You are an AI smart glasses assistant. You need to answer the user's questions according to the image provided."
                final_prompt = f"{context_prompt}\nUser Question: {transcribed_text}"

                # Prepare image parts for the model
                image_parts = [
                    {
                        "mime_type": uploaded_file.type,
                        "data": image_bytes
                    }
                ]

                # Generate content using the prompt and the image
                response = model.generate_content([final_prompt, image_parts[0]])

                # --- Process and Display Response ---
                st.header("AI Assistant's Response")

                if hasattr(response, 'text') and response.text.strip():
                    gemini_response_text = response.text.strip()
                    st.markdown(gemini_response_text)

                    # Convert text response to speech
                    with st.spinner("Generating audio response..."):
                        tts = gTTS(gemini_response_text)
                        audio_fp = io.BytesIO()
                        tts.write_to_fp(audio_fp)
                        audio_fp.seek(0)
                        
                        # Play the audio response automatically
                        st.audio(audio_fp, format='audio/mp3', start_time=0)
                else:
                    st.error("The model did not return a valid text response.")
                    st.write("Raw Response:", response)

            except genai.types.BlockedPromptException as e:
                st.error(f"Error: The prompt was blocked. Reason: {e.response.prompt_feedback.block_reason}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please upload an image and record your question first.")
