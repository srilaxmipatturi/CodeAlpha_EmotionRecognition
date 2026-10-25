"""
=============================================================================
Project: Emotion Recognition from Speech (SER)
File: predict.py
Purpose: Standalone inference script. Predicts the emotion of any given audio
         file from the command line using the trained deep learning model.
=============================================================================
Why do we need this file?
Once a model is trained, you need an easy way to pass any new audio recording
into the pipeline and get back the predicted emotion and confidence scores.
=============================================================================
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import joblib

# Fix Windows console encoding for UTF-8 support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import project configurations, feature extraction, and model builder
from config import (
    BEST_MODEL_PATH,
    MODELS_DIR,
    DATASET_DIR,
    EMOTIONS,
    NUM_CLASSES,
)
from features import extract_features
from model import build_model

# Emotion emojis for friendly output display
EMOTION_EMOJIS = {
    "neutral":   "😐 Neutral",
    "calm":      "😌 Calm",
    "happy":     "😊 Happy",
    "sad":       "😢 Sad",
    "angry":     "😡 Angry",
    "fearful":   "😨 Fearful",
    "disgust":   "🤢 Disgust",
    "surprised": "😲 Surprised",
}


def load_prediction_artifacts(device):
    """
    Loads the trained model checkpoint and fitted StandardScaler.
    """
    scaler_path = MODELS_DIR / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please run train.py first!")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {BEST_MODEL_PATH}. Please run train.py first!")

    # 1. Load the fitted feature scaler
    scaler = joblib.load(scaler_path)

    # 2. Load the trained PyTorch model checkpoint
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "cnn_lstm")

    # The input dimension is 364
    model = build_model(model_type=model_type, input_dim=364, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, scaler


def predict_emotion(audio_path, model=None, scaler=None):
    """
    Takes an audio file path, extracts acoustic features, standardizes them,
    and returns the predicted emotion along with probabilities for all 8 emotions.
    
    Returns:
        predicted_emotion (str): The emotion with highest probability.
        confidence (float): Confidence percentage (0-100%).
        all_probabilities (dict): Dictionary of {emotion: probability_percent}.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load artifacts if not passed in
    if model is None or scaler is None:
        model, scaler = load_prediction_artifacts(device)

    # Step 1: Extract 364 acoustic features
    raw_features = extract_features(audio_path)

    # Step 2: Standardize features using the training scaler
    # Reshape from (364,) to (1, 364) for scaler.transform
    scaled_features = scaler.transform(raw_features.reshape(1, -1))

    # Step 3: PyTorch Model Inference
    tensor_input = torch.tensor(scaled_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Step 4: Identify Top Emotion and Confidence
    top_class_id = int(np.argmax(probabilities))
    predicted_emotion = EMOTIONS[top_class_id]
    confidence = float(probabilities[top_class_id] * 100.0)

    all_probabilities = {
        EMOTIONS[i]: float(probabilities[i] * 100.0) for i in range(len(EMOTIONS))
    }

    return predicted_emotion, confidence, all_probabilities


def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition - Audio Prediction Utility")
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to .wav audio file to analyze.",
    )
    args = parser.parse_args()

    # If no audio provided, find the first available sample audio file
    if args.audio is None:
        sample_files = list(DATASET_DIR.rglob("*.wav"))
        if not sample_files:
            print("No audio file specified and no samples found in dataset/.")
            print("Usage: python predict.py --audio path/to/voice.wav")
            sys.exit(1)
        audio_path = sample_files[0]
        print(f"No audio file specified. Using sample audio: {audio_path.name}")
    else:
        audio_path = Path(args.audio)

    print("\n" + "=" * 60)
    print("Speech Emotion Recognition — Audio Analysis")
    print("=" * 60)
    print(f"Target Audio File: {audio_path}")

    emotion, confidence, probabilities = predict_emotion(audio_path)

    print("\n------------------------------------------------------------")
    print(f"  PREDICTED EMOTION : {EMOTION_EMOJIS.get(emotion, emotion).upper()}")
    print(f"  CONFIDENCE SCORE  : {confidence:.2f}%")
    print("------------------------------------------------------------\n")

    print("Detailed Probability Distribution:")
    # Sort emotions by probability in descending order
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for emo, prob in sorted_probs:
        bar = "#" * int(prob / 4)
        print(f"  {EMOTION_EMOJIS.get(emo, emo):<16} : {prob:5.1f}%  {bar}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
