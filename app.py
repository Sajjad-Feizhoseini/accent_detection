import streamlit as st
from pydub import AudioSegment
import speechbrain as sb
from speechbrain.inference import EncoderClassifier  # Updated import
import librosa
import numpy as np
import os
import requests
import tempfile
import torchaudio

# Function to download video and extract audio
def extract_audio_from_video(video_url):
    try:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        with open(temp_video.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        video_audio = AudioSegment.from_file(temp_video.name, format="mp4")
        video_audio.export(temp_audio.name, format="wav")
        
        temp_video.close()
        return temp_audio.name
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None

# Function to analyze accent
def analyze_accent(audio_path):
    try:
        # Load pre-trained accent classification model
        classifier = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",  # Updated repo_id
            savedir="pretrained_models/accent-id-commonaccent_ecapa"
        )
        
        # Load audio using torchaudio with ffmpeg backend
        signal, sr = torchaudio.load(audio_path, backend="ffmpeg")
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            signal = resampler(signal)
        
        signal = signal.squeeze().numpy()
        
        # Perform accent classification
        output = classifier.classify_batch(signal)
        accent_probs = output[0].exp().numpy()
        accent_labels = classifier.hparams.label_encoder
        
        english_accents = ["British", "American", "Australian"]
        accent_scores = {label: prob for label, prob in zip(accent_labels, accent_probs) if label in english_accents}
        
        if not accent_scores:
            return "Non-English or unsupported accent", 0.0, "No English accent detected."
        
        top_accent = max(accent_scores, key=accent_scores.get)
        confidence = accent_scores[top_accent] * 100
        
        summary = f"Detected {top_accent} accent with {confidence:.2f}% confidence. "
        summary += f"Other probabilities: {', '.join([f'{k}: {v*100:.2f}%' for k, v in accent_scores.items()])}"
        
        return top_accent, confidence, summary
    except Exception as e:
        st.error(f"Error analyzing accent: {str(e)}")
        return "Error", 0.0, str(e)

# Streamlit UI
st.title("REM Waste Accent Analyzer")
st.write("Enter a public video URL to analyze the speaker's English accent.")

video_url = st.text_input("Video URL (e.g., Loom or MP4 link)")
if st.button("Analyze"):
    if video_url:
        with st.spinner("Extracting audio..."):
            audio_path = extract_audio_from_video(video_url)
            if audio_path:
                with st.spinner("Analyzing accent..."):
                    accent, confidence, summary = analyze_accent(audio_path)
                    st.success("Analysis complete!")
                    st.write(f"**Detected Accent**: {accent}")
                    st.write(f"**Confidence Score**: {confidence:.2f}%")
                    st.write(f"**Summary**: {summary}")
                    
                    os.unlink(audio_path)
                    if os.path.exists(audio_path.replace(".wav", ".mp4")):
                        os.unlink(audio_path.replace(".wav", ".mp4"))
    else:
        st.error("Please provide a valid video URL.")

st.write("Note: This tool supports British, American, and Australian English accents.")
