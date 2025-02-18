import streamlit as st
import gdown
import os
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
import json
import librosa
import numpy as np
import soundfile as sf

# ✅ Set page config
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")

# ✅ Define Google Drive model folder link (Replace with your link ID)
GOOGLE_DRIVE_MODEL_LINK = "https://drive.google.com/uc?id=1LXL3F0oNbc3oBGVpJWVt4yrx3g9nqQu9"
MODEL_DIR = "./saved_hubert_model"

# ✅ Function to download and extract the model
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with st.spinner("Downloading model..."):
            output_path = os.path.join(MODEL_DIR, "model.zip")
            gdown.download(GOOGLE_DRIVE_MODEL_LINK, output_path, quiet=False)

            # Extract if it's a ZIP file
            if output_path.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(output_path, "r") as zip_ref:
                    zip_ref.extractall(MODEL_DIR)
                os.remove(output_path)  # Remove ZIP file after extraction

    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = HubertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    model.eval()
    return processor, model

# ✅ Load the model
processor, model = download_and_load_model()

# ✅ Load emotions map
with open(os.path.join(MODEL_DIR, "emotions_map.json"), "r") as f:
    emotions_map = json.load(f)
emotions_map = {int(k): v for k, v in emotions_map.items()}

# 🎵 Predict Emotion Function
def predict_emotion(audio, sample_rate):
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = F.softmax(logits, dim=-1).squeeze()

    predicted_class_id = torch.argmax(probabilities).item()
    predicted_emotion = emotions_map[predicted_class_id]

    st.markdown(f"## 🎯 Detected Emotion: **{predicted_emotion}**")
    st.subheader("📊 Emotion Probabilities:")
    for class_id, prob in enumerate(probabilities):
        emotion = emotions_map[class_id]
        st.write(f"{emotion}: **{prob.item() * 100:.2f}%**")
        st.progress(prob.item())

# ✅ Streamlit UI
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a **.wav file** or **record live audio** to detect emotions.")

# 📂 Upload Audio File
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    speech, sample_rate = sf.read(uploaded_file)
    if len(speech.shape) == 2:
        speech = np.mean(speech, axis=1)
    predict_emotion(speech, sample_rate)
