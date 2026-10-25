"""
=============================================================================
Project: Emotion Recognition from Speech (SER)
File: dataset_loader.py
Purpose: Scans audio datasets (RAVDESS / TESS), extracts emotion labels,
         manages train/test splits, caches extracted features for fast loading,
         and provides a synthetic starter dataset for instant testing.
=============================================================================
Why do we need this file?
Before training a neural network, we need organized pairs of:
  Input (X) = 364 audio features
  Target (y) = Emotion label (0 for neutral, 1 for calm, 2 for happy, etc.)
This file connects raw audio files from the disk to the training engine.
=============================================================================
"""

import os
import re
from pathlib import Path
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Import project settings and feature extraction functions
from config import (
    DATASET_DIR,
    MODELS_DIR,
    EMOTIONS,
    EMOTION_TO_ID,
    ID_TO_EMOTION,
    RAVDESS_EMOTION_MAP,
    SAMPLE_RATE,
    DURATION,
    RANDOM_SEED,
)
from features import extract_features, extract_2d_spectrogram


# ---------------------------------------------------------------------------
# 1. EMOTION PARSER: Detecting emotion from filename or folder
# ---------------------------------------------------------------------------
def parse_emotion_from_path(file_path):
    """
    Identifies the emotion of an audio file using 3 common conventions:
    
    1. RAVDESS Convention: '03-01-05-01-01-01-01.wav'
       The 3rd hyphen-separated number indicates emotion:
       01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
       
    2. TESS Convention: 'OAF_back_angry.wav' or 'YAF_happy.wav'
       Emotion name is at the end of the filename.
       
    3. Folder Convention: 'dataset/happy/speech_01.wav'
       Parent folder name is the emotion.
       
    Returns:
        str: Emotion label (e.g. 'happy') or None if not recognized.
    """
    path = Path(file_path)
    filename = path.stem.lower()

    # Case 1: Check RAVDESS pattern (e.g., 03-01-05-...)
    parts = filename.split("-")
    if len(parts) >= 3 and parts[2] in RAVDESS_EMOTION_MAP:
        return RAVDESS_EMOTION_MAP[parts[2]]

    # Case 2: Check TESS pattern (e.g., OAF_back_angry or YAF_fear)
    # TESS uses 'fear' for fearful and 'ps' for pleasant surprise
    tess_map = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fearful",
        "fearful": "fearful",
        "happy": "happy",
        "neutral": "neutral",
        "ps": "surprised",
        "surprise": "surprised",
        "surprised": "surprised",
        "sad": "sad",
        "calm": "calm",
    }
    for key, emotion_name in tess_map.items():
        if filename.endswith(f"_{key}") or filename.endswith(f" {key}") or f"_{key}_" in filename:
            return emotion_name

    # Case 3: Check if parent folder name is an emotion
    parent_folder = path.parent.name.lower()
    if parent_folder in EMOTION_TO_ID:
        return parent_folder

    # Case 4: Check if any emotion keyword is inside the filename
    for emotion in EMOTIONS:
        if emotion in filename:
            return emotion

    return None


def find_audio_files(directory=DATASET_DIR):
    """
    Recursively scans the dataset directory for all .wav files and parses their emotions.
    
    Returns:
        list of tuples: [(file_path, emotion_label), ...]
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    matched_files = []
    # Search for all .wav and .WAV files
    for file_path in directory.rglob("*.wav"):
        # Ignore sample test directory to avoid mixing test tones with training data
        if "sample_test" in str(file_path):
            continue
            
        emotion = parse_emotion_from_path(file_path)
        if emotion and emotion in EMOTION_TO_ID:
            matched_files.append((file_path, emotion))

    return matched_files


# ---------------------------------------------------------------------------
# 2. SYNTHETIC STARTER DATASET GENERATOR
# ---------------------------------------------------------------------------
def generate_starter_dataset(samples_per_emotion=15):
    """
    Generates an acoustic starter dataset with distinct acoustic profiles
    for each of the 8 emotions.
    
    Why is this useful?
    - If you don't have the 1.4 GB RAVDESS dataset downloaded yet, this allows you
      to test the entire machine learning pipeline (features -> train -> evaluate -> app)
      immediately in 15 seconds!
    - When real RAVDESS or TESS files are added later, they will be loaded automatically.
    """
    actors_dir = DATASET_DIR / "starter_speech"
    actors_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic starter speech dataset ({samples_per_emotion} samples x 8 emotions)...")

    # Acoustic characteristics for each emotion:
    # (base_pitch_hz, energy_amplitude, frequency_modulation_rate, harmonic_richness)
    profiles = {
        "neutral":   (200, 0.40, 0.5, 2),  # Steady mid-pitch, normal energy
        "calm":      (160, 0.25, 0.2, 1),  # Low pitch, quiet, slow
        "happy":     (320, 0.65, 3.5, 4),  # High pitch, loud, energetic vibrato
        "sad":       (140, 0.20, 0.1, 1),  # Very low pitch, quiet, flat monotone
        "angry":     (380, 0.90, 5.0, 6),  # Piercing high pitch, very loud, harsh harmonics
        "fearful":   (350, 0.60, 8.0, 3),  # High pitch, jittery rapid modulation
        "disgust":   (175, 0.35, 1.5, 5),  # Low-mid pitch with rasping harmonics
        "surprised": (420, 0.75, 2.0, 3),  # High soaring pitch glissando
    }

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)

    for emotion, (base_pitch, amp, mod_rate, harmonics) in profiles.items():
        emotion_dir = actors_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_emotion):
            # Add minor individual variation to each sample (jitter and volume variance)
            pitch = base_pitch + np.random.uniform(-15, 15)
            # Acoustic sound synthesis: fundamental wave + harmonics + pitch modulation
            modulation = 1.0 + 0.15 * np.sin(2 * np.pi * mod_rate * t)
            signal = np.zeros_like(t)

            for h in range(1, harmonics + 1):
                harmonic_amp = (amp / h) * np.random.uniform(0.8, 1.2)
                signal += harmonic_amp * np.sin(2 * np.pi * (pitch * h) * t * modulation)

            # Apply a speech amplitude envelope (fade in at start, fade out at end)
            envelope = np.sin(np.pi * np.linspace(0, 1, len(t)))
            signal = signal * envelope

            # Add gentle natural acoustic noise
            signal += np.random.normal(0, 0.005, len(t))

            # Normalize to avoid audio clipping
            signal = signal / (np.max(np.abs(signal)) + 1e-6) * amp

            file_path = emotion_dir / f"{emotion}_sample_{i+1:02d}.wav"
            sf.write(file_path, signal.astype(np.float32), SAMPLE_RATE)

    print(f"Starter dataset successfully created at: {actors_dir}")


# ---------------------------------------------------------------------------
# 3. FEATURE EXTRACTION & CACHING PIPELINE
# ---------------------------------------------------------------------------
def load_or_extract_dataset(cache_path=None, force_reextract=False):
    """
    Extracts 1D feature vectors for all audio files in the dataset.
    Caches the results to a compressed .npz file for instant reloading in the future.
    
    Returns:
        X (np.ndarray): Feature matrix of shape (num_samples, 364)
        y (np.ndarray): Integer label array of shape (num_samples,)
        file_paths (list): List of file paths
    """
    if cache_path is None:
        cache_path = DATASET_DIR / "features_cache.npz"
    else:
        cache_path = Path(cache_path)

    # 1. If cache exists and re-extraction is not forced, load instantly!
    if cache_path.exists() and not force_reextract:
        print(f"Loading cached features from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], list(data["file_paths"])

    # 2. Check for audio files
    audio_files = find_audio_files(DATASET_DIR)

    # If no audio files found, generate the starter dataset automatically
    if len(audio_files) == 0:
        print("No audio files found in dataset directory. Creating starter speech dataset...")
        generate_starter_dataset(samples_per_emotion=15)
        audio_files = find_audio_files(DATASET_DIR)

    print(f"Found {len(audio_files)} audio files across classes. Extracting features...")

    X_list = []
    y_list = []
    valid_paths = []

    for idx, (fpath, emotion) in enumerate(audio_files):
        try:
            feat = extract_features(fpath)
            label = EMOTION_TO_ID[emotion]
            X_list.append(feat)
            y_list.append(label)
            valid_paths.append(str(fpath))

            if (idx + 1) % 25 == 0 or (idx + 1) == len(audio_files):
                print(f"  Processed {idx + 1}/{len(audio_files)} files...")
        except Exception as e:
            print(f"  [Warning] Failed to process {fpath}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Save to compressed cache
    np.savez_compressed(cache_path, X=X, y=y, file_paths=valid_paths)
    print(f"Features saved to cache: {cache_path} (Shape: {X.shape})")

    return X, y, valid_paths


def get_train_test_data(test_size=0.20, random_state=RANDOM_SEED):
    """
    Loads dataset, splits into 80% train and 20% test sets, and standardizes
    features using StandardScaler (zero mean, unit variance).
    
    Returns:
        X_train (np.ndarray): Normalized training features
        X_test (np.ndarray): Normalized testing features
        y_train (np.ndarray): Training integer labels
        y_test (np.ndarray): Testing integer labels
        scaler (StandardScaler): The fitted scaler object (saved to models/scaler.joblib)
    """
    X, y, _ = load_or_extract_dataset()

    # Stratified split ensures each emotion is equally represented in both train & test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature Scaling: Neural networks converge much faster when features have mean=0 and std=1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler so future single-audio predictions in app.py can use the exact same scaling!
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"StandardScaler saved to: {scaler_path}")

    return X_train, X_test, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# Quick Verification when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Stage 3: Dataset Loading, Splitting & Feature Caching")
    print("=" * 60)

    X_train, X_test, y_train, y_test, scaler = get_train_test_data()

    print(f"Training set features shape: {X_train.shape} (Samples x Features)")
    print(f"Testing set features shape:  {X_test.shape}")
    print(f"Unique classes present:      {len(np.unique(y_train))} emotions")
    print("Stage 3 Dataset Pipeline verified successfully!")
    print("=" * 60)
