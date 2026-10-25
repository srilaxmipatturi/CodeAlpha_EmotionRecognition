"""
=============================================================================
Project: Emotion Recognition from Speech (SER)
File: config.py
Purpose: Central configuration file storing all project settings, paths,
         audio processing parameters, and deep learning hyperparameters.
=============================================================================
Why do we have a config.py file?
Instead of hardcoding numbers like 22050 (sample rate) or 40 (MFCCs) in every
file, we keep them in one place. If you want to change the audio duration
or batch size, you only change it here!
=============================================================================
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. DIRECTORY PATHS (Where files are stored)
# ---------------------------------------------------------------------------
# BASE_DIR is the root folder where this project lives.
BASE_DIR = Path(__file__).resolve().parent

# DATASET_DIR: Where speech datasets (like RAVDESS or TESS) are downloaded/stored.
DATASET_DIR = BASE_DIR / "dataset"

# MODELS_DIR: Where trained model weights (.pth) are saved.
MODELS_DIR = BASE_DIR / "models"

# OUTPUTS_DIR: Where plots (loss curve, confusion matrix) are saved.
OUTPUTS_DIR = BASE_DIR / "outputs"

# Path for the best trained model file
BEST_MODEL_PATH = MODELS_DIR / "best_ser_model.pth"

# Ensure all these folders exist automatically
for folder in [DATASET_DIR, MODELS_DIR, OUTPUTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. AUDIO SIGNAL PROCESSING PARAMETERS
# ---------------------------------------------------------------------------
# SAMPLE_RATE (Hz): The number of audio samples recorded per second.
# 22,050 Hz is the standard in speech recognition because human vocal frequencies
# comfortably fall well below 11,000 Hz (Nyquist theorem: 22050 / 2 = 11025 Hz).
SAMPLE_RATE = 22050

# DURATION (seconds): Each audio sample is standardized to this duration.
# If an audio clip is 2.5 seconds, we pad it with silence to 3.0 seconds.
# If an audio clip is 4.0 seconds, we trim it to 3.0 seconds.
# Total audio samples per file = 22050 * 3.0 = 66,150 numbers.
DURATION = 3.0
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION)

# N_MFCC: Number of Mel-Frequency Cepstral Coefficients to extract per time frame.
# 40 coefficients capture both low-pitch pitch formants and higher harmonics.
N_MFCC = 40

# N_MELS: Number of Mel filter banks used when computing Mel-Spectrograms.
N_MELS = 128

# N_FFT: Fast Fourier Transform window size (in samples).
# A window of 2048 samples at 22050 Hz corresponds to ~93 milliseconds of speech.
N_FFT = 2048

# HOP_LENGTH: The number of audio samples we slide the window forward each time.
# 512 samples means an overlap of 75% between windows, giving smooth temporal transitions.
HOP_LENGTH = 512


# ---------------------------------------------------------------------------
# 3. EMOTION CLASSES AND LABEL ENCODING
# ---------------------------------------------------------------------------
# The 8 standard emotions in the RAVDESS dataset:
# Filename format in RAVDESS: 03-01-XX-01-01-01-01.wav
# The 3rd number (XX) represents the emotion:
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad,
# 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Ordered list of unique emotion names for training & evaluation
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
NUM_CLASSES = len(EMOTIONS)

# Fast lookups: emotion name -> integer ID, and integer ID -> emotion name
EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
ID_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}


# ---------------------------------------------------------------------------
# 4. DEEP LEARNING MODEL HYPERPARAMETERS
# ---------------------------------------------------------------------------
# BATCH_SIZE: Number of audio samples processed simultaneously during one training step.
BATCH_SIZE = 32

# LEARNING_RATE: How large of a step the optimizer takes when adjusting model weights.
LEARNING_RATE = 0.0005

# EPOCHS: The number of full passes through the training dataset.
EPOCHS = 40

# RANDOM_SEED: Ensures reproducible train/test splits and weight initialization.
RANDOM_SEED = 42
