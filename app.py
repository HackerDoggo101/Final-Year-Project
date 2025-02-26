import streamlit as st  
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
import json
import numpy as np
import librosa
from io import BytesIO
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

# Set Streamlit page config
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")

# Load model and processor
@st.cache_resource
def load_model():
    processor = Wav2Vec2FeatureExtractor.from_pretrained("./saved_hubert_model", local_files_only=True)
    model = HubertForSequenceClassification.from_pretrained("./saved_hubert_model", local_files_only=True)
    model.eval()
    return processor, model

processor, model = load_model()

# Load emotions map
with open("./emotions_map.json", "r") as f:
    emotions_map = json.load(f)
emotions_map = {int(k): v for k, v in emotions_map.items()}

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")
st.write("Choose between **uploading a file** or **live recording** to detect emotions.")

# Choose Input Method
option = st.radio("Select an input method:", ["📂 Upload Audio File", "🎤 Live Recording"])

# Process and Predict Emotion Function
def predict_emotion(audio, sample_rate):
    # Resample if needed
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    
    # Preprocess the audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # Predict emotion
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = F.softmax(logits, dim=-1).squeeze()

    # Get predicted emotion
    predicted_class_id = torch.argmax(probabilities).item()
    predicted_emotion = emotions_map[predicted_class_id]

    # Display results
    st.markdown(f"## 🎯 Detected Emotion: **{predicted_emotion}**")
    st.subheader("📊 Emotion Probabilities:")
    for class_id, prob in enumerate(probabilities):
        emotion = emotions_map[class_id]
        st.write(f"{emotion}: **{prob.item() * 100:.2f}%**")
        st.progress(prob.item())

# Upload Audio File
if option == "📂 Upload Audio File":
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        speech, sample_rate = sf.read(uploaded_file)
        if len(speech.shape) == 2:  # Convert stereo to mono
            speech = np.mean(speech, axis=1)
        predict_emotion(speech, sample_rate)

# Live Recording
elif option == "🎤 Live Recording":
    audio_data = mic_recorder(start_prompt="🎙️ Click to Record", key="recorder")

    if audio_data is not None:
        if isinstance(audio_data, dict) and "bytes" in audio_data:
            audio_bytes = audio_data["bytes"]  # Extract actual bytes
            
            # Convert WebM/Opus to WAV
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format="webm").set_frame_rate(16000).set_channels(1)
            
            # Save to a BytesIO buffer
            wav_io = BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)

            # Display the recorded audio
            st.audio(wav_io, format="audio/wav")

            # Convert WAV to NumPy array
            with sf.SoundFile(wav_io) as f:
                speech = f.read(dtype="float32")
                sample_rate = f.samplerate

            predict_emotion(speech, sample_rate)
        else:
            st.error("⚠️ No valid audio recorded. Please try again.")
