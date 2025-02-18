import streamlit as st  
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
import json
import librosa
import io  # Added for handling in-memory audio
#test
# ✅ Set page configuration
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")

# ✅ Load model and processor
@st.cache_resource
def load_model():
    processor = Wav2Vec2FeatureExtractor.from_pretrained("C:/Users/Jason/Desktop/New Folder/University/FYP/saved_hubert_model", local_files_only=True)
    model = HubertForSequenceClassification.from_pretrained("C:/Users/Jason/Desktop/New Folder/University/FYP/saved_hubert_model", local_files_only=True)
    model.eval()
    return processor, model

processor, model = load_model()

# ✅ Load emotions map
with open("C:/Users/Jason/Desktop/New Folder/University/FYP/emotions_map.json", "r") as f:
    emotions_map = json.load(f)
emotions_map = {int(k): v for k, v in emotions_map.items()}

# 🎵 Process and Predict Emotion Function
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

# ✅ Streamlit UI
st.title("🎙️ Speech Emotion Recognition")
st.write("Choose between **uploading a file** or **live recording** to detect emotions.")

# 🎤 Choose Input Method
option = st.radio("Select an input method:", ["📂 Upload Audio File", "🎤 Live Recording"])

# 🎤 Recording settings
duration = 5  # Record for 5 seconds
sampling_rate = 16000  # Required for the model

# 🎤 Function to Record Audio and Save as WAV
def record_audio(duration=5, sampling_rate=16000):
    st.write("🔴 Recording... Speak now!")
    audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished
    st.write("✅ Recording complete!")

    # Save as in-memory WAV file
    wav_io = io.BytesIO()
    sf.write(wav_io, audio, samplerate=sampling_rate, format='wav')
    wav_io.seek(0)

    return audio.flatten(), wav_io

# 📂 Upload Audio File
if option == "📂 Upload Audio File":
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        speech, sample_rate = sf.read(uploaded_file)
        if len(speech.shape) == 2:  # Convert stereo to mono
            speech = np.mean(speech, axis=1)
        predict_emotion(speech, sample_rate)

# 🎤 Live Recording
elif option == "🎤 Live Recording":
    if st.button("🎤 Start Recording"):
        recorded_audio, wav_file = record_audio(duration, sampling_rate)
        
        # 🎵 Play back the recorded audio
        st.audio(wav_file, format="audio/wav")
        
        # 🎵 Process for emotion detection
        predict_emotion(recorded_audio, sampling_rate)
