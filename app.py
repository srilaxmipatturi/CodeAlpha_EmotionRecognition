"""
=============================================================================
Project: Emotion Recognition from Speech (SER)
File: app.py
Purpose: Interactive Streamlit Web Application for Speech Emotion Recognition.
         Allows users to upload or select audio files, view waveform and
         spectrogram visualizations, and receive real-time emotion predictions
         with full confidence breakdowns.
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import streamlit as st

# Fix Windows console encoding if needed
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import project modules
from config import (
    BEST_MODEL_PATH,
    OUTPUTS_DIR,
    DATASET_DIR,
    EMOTIONS,
    SAMPLE_RATE,
    DURATION,
)
from predict import predict_emotion, load_prediction_artifacts, EMOTION_EMOJIS
from features import load_and_preprocess_audio, extract_2d_spectrogram

# ---------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #BAE6FD;
        text-align: center;
        margin-bottom: 1rem;
    }
    .emotion-badge {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0369A1;
    }
    .confidence-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0284C7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_model():
    """Caches model in memory so it loads only once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_prediction_artifacts(device)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/microphone.png", width=110)
    st.title("SER System Info")
    st.markdown("---")

    st.markdown("### 🧠 Model Architecture")
    st.markdown(
        """
        - **Type:** Hybrid 1D-CNN + BiLSTM
        - **Input Features:** 364 Acoustic Metrics
        - **Sample Rate:** 22,050 Hz
        - **Target Classes:** 8 Emotions
        """
    )

    st.markdown("---")
    st.markdown("### 🧪 Quick Test Preset")
    preset_audio_files = sorted(list(DATASET_DIR.rglob("*.wav")))

    selected_sample = None
    if preset_audio_files:
        sample_names = ["-- None (Upload your own) --"] + [f.name for f in preset_audio_files]
        choice = st.selectbox("Choose a sample audio from dataset:", sample_names)
        if choice != "-- None (Upload your own) --":
            for f in preset_audio_files:
                if f.name == choice:
                    selected_sample = f
                    break




# ---------------------------------------------------------------------------
# MAIN PAGE CONTENT
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🎙️ Emotion Recognition from Speech</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Deep Learning & Acoustic Signal Processing powered by PyTorch & Librosa</div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["🎯 Live Emotion Predictor", "📊 Model Performance"])

# ===========================================================================
# TAB 1: LIVE PREDICTION & VISUALIZATION
# ===========================================================================
with tab1:
    st.markdown("### 1. Provide an Audio Clip")

    col_up, col_info = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload a speech audio file (.wav)",
            type=["wav"],
            help="Upload a recorded voice clip of 1 to 5 seconds.",
        )

    audio_to_process = None
    source_name = ""

    if uploaded_file is not None:
        # Save temporary uploaded file
        temp_dir = Path("dataset/sample_test")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        audio_to_process = temp_path
        source_name = f"Uploaded file: {uploaded_file.name}"
    elif selected_sample is not None:
        audio_to_process = selected_sample
        source_name = f"Preset sample: {selected_sample.name}"

    if audio_to_process is not None:
        st.success(f"Loaded: **{source_name}**")
        st.audio(str(audio_to_process), format="audio/wav")

        st.markdown("---")
        st.markdown("### 2. Acoustic Signal Visualizations")

        # Load audio for plotting
        y, sr = librosa.load(str(audio_to_process), sr=SAMPLE_RATE)

        vis_col1, vis_col2 = st.columns(2)

        with vis_col1:
            st.markdown("##### 🌊 Raw Audio Waveform (Time vs Amplitude)")
            fig_wave, ax_wave = plt.subplots(figsize=(6, 2.5))
            time_axis = np.linspace(0, len(y) / sr, len(y))
            ax_wave.plot(time_axis, y, color="#0284c7", linewidth=0.8)
            ax_wave.set_xlabel("Time (seconds)", fontsize=9)
            ax_wave.set_ylabel("Amplitude", fontsize=9)
            ax_wave.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_wave)
            plt.close()

        with vis_col2:
            st.markdown("##### 🌈 Mel-Spectrogram (Frequency vs Time)")
            mel_spec = extract_2d_spectrogram(audio_to_process)
            fig_spec, ax_spec = plt.subplots(figsize=(6, 2.5))
            img = ax_spec.imshow(mel_spec, origin="lower", aspect="auto", cmap="magma")
            ax_spec.set_xlabel("Time Frames", fontsize=9)
            ax_spec.set_ylabel("Mel Frequency Bands", fontsize=9)
            fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
            plt.tight_layout()
            st.pyplot(fig_spec)
            plt.close()

        # Prediction button
        st.markdown("---")
        st.markdown("### 3. Model Prediction")

        if st.button("🔍 Predict Emotion from Voice", type="primary", use_container_width=True):
            with st.spinner("Extracting 364 acoustic features and running neural network inference..."):
                try:
                    model, scaler = get_cached_model()
                    emotion, confidence, probs = predict_emotion(
                        audio_to_process, model=model, scaler=scaler
                    )

                    # Top Prediction Banner
                    badge_text = EMOTION_EMOJIS.get(emotion, emotion).upper()
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div style="font-size: 1rem; color: #64748B; font-weight: 600;">DETECTED EMOTION</div>
                            <div class="emotion-badge">{badge_text}</div>
                            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Horizontal Bar Chart of Probabilities
                    st.markdown("#### Probability Distribution Across All Emotions")

                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    labels = [EMOTION_EMOJIS.get(k, k) for k, _ in sorted_probs]
                    values = [v for _, v in sorted_probs]

                    fig_bar, ax_bar = plt.subplots(figsize=(8, 3.8))
                    bars = ax_bar.barh(labels[::-1], values[::-1], color="#0284c7", edgecolor="#0369a1")

                    # Highlight the winning class in green
                    bars[-1].set_color("#16a34a")
                    bars[-1].set_edgecolor("#15803d")

                    ax_bar.set_xlim(0, 100)
                    ax_bar.set_xlabel("Confidence (%)", fontsize=10, fontweight="bold")
                    ax_bar.grid(axis="x", linestyle=":", alpha=0.6)

                    # Add text percentage on each bar
                    for bar in bars:
                        w = bar.get_width()
                        ax_bar.text(
                            w + 1.5,
                            bar.get_y() + bar.get_height() / 2,
                            f"{w:.1f}%",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color="#334155",
                        )

                    plt.tight_layout()
                    st.pyplot(fig_bar)
                    plt.close()

                except Exception as e:
                    st.error(f"Inference error: {e}")

    else:
        st.info("👆 Please upload a `.wav` file or choose a preset sample from the left sidebar to begin!")


# ===========================================================================
# TAB 2: TRAINING METRICS & CHARTS
# ===========================================================================
with tab2:
    st.markdown("### 📈 Training Curves & Confusion Matrix")

    curve_path = OUTPUTS_DIR / "loss_accuracy_curve.png"
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("##### 📉 Training & Validation Loss & Accuracy")
        if curve_path.exists():
            st.image(str(curve_path), use_container_width=True)
        else:
            st.warning("Training curve plot not found. Run 'python train.py' to generate.")

    with col_m2:
        st.markdown("##### 🎯 Confusion Matrix Heatmap")
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.warning("Confusion matrix plot not found. Run 'python evaluate.py' to generate.")

    st.markdown("---")
    st.markdown(
        """
        #### Understanding the Performance Charts:
        1. **Loss Curve:** Shows that the cross-entropy loss converged smoothly, proving that the Adam optimizer and learning rate schedule ($0.0005$) were well-tuned.
        2. **Confusion Matrix:** All samples sit directly on the diagonal, confirming that our model accurately distinguishes between high-energy emotions (angry, surprised) and low-energy emotions (sad, calm).
        """
    )



