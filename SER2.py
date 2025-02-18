import streamlit as st
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
import json
import librosa
import base64
from io import BytesIO

# ✅ Set page config
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")

# ✅ Load model
@st.cache_resource
def load_model():
    processor = Wav2Vec2FeatureExtractor.from_pretrained("saved_hubert_model", local_files_only=True)
    model = HubertForSequenceClassification.from_pretrained("saved_hubert_model", local_files_only=True)
    model.eval()
    return processor, model

processor, model = load_model()

# ✅ Load emotions map
with open("emotions_map.json", "r") as f:
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

# 🎤 Live Recording with JavaScript
st.markdown("---")
st.subheader("🎤 Record Live Audio")

# JavaScript to record audio in the browser
st.markdown(
    """
    <script>
        let mediaRecorder;
        let audioChunks = [];

        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                audioChunks = [];

                mediaRecorder.addEventListener("dataavailable", event => {
                    audioChunks.push(event.data);
                });

                mediaRecorder.addEventListener("stop", () => {
                    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        const base64String = reader.result.split(',')[1];
                        fetch('/recorded_audio', {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ audio: base64String })
                        });
                    };
                });
            });
        }

        function stopRecording() {
            if (mediaRecorder) {
                mediaRecorder.stop();
            }
        }
    </script>
    <button onclick="startRecording()">🎙️ Start Recording</button>
    <button onclick="stopRecording()">⏹️ Stop Recording</button>
    """,
    unsafe_allow_html=True
)

# 🔄 Handle recorded audio
recorded_audio = st.experimental_get_query_params().get("recorded_audio")
if recorded_audio:
    audio_bytes = base64.b64decode(recorded_audio[0])
    st.audio(audio_bytes, format="audio/wav")

    # Process audio
    audio_io = BytesIO(audio_bytes)
    speech, sample_rate = sf.read(audio_io)
    if len(speech.shape) == 2:
        speech = np.mean(speech, axis=1)
    
    predict_emotion(speech, sample_rate)
