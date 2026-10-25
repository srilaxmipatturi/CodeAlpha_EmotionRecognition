"""
=============================================================================
Project: Emotion Recognition from Speech (SER)
File: features.py
Purpose: Audio signal preprocessing and feature extraction.
         Converts raw sound vibrations into mathematical feature vectors
         (MFCC, Chroma, Mel-Spectrogram, RMS, ZCR).
=============================================================================
Why do we need this file?
A raw audio file has 66,150 raw numbers with immense noise. This file acts
as a "translator" that translates noisy sound waves into clean, meaningful
musical and acoustic characteristics that a machine learning model can understand.
=============================================================================
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Import our project settings from config.py
from config import (
    SAMPLE_RATE,
    DURATION,
    TOTAL_SAMPLES,
    N_MFCC,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
)


def load_and_preprocess_audio(audio_input, sr=SAMPLE_RATE, duration=DURATION):
    """
    Loads an audio file and standardizes it to a fixed length and sample rate.
    
    Parameters:
        audio_input (str, Path, or np.ndarray): File path to .wav file OR raw numpy array.
        sr (int): Target sampling rate (default: 22,050 Hz).
        duration (float): Target duration in seconds (default: 3.0s).
        
    Returns:
        np.ndarray: A standardized 1D numpy array of audio samples, exactly
                    TOTAL_SAMPLES (66,150) in length.
    """
    # 1. If audio_input is a file path (string or Path object), load it using librosa.
    if isinstance(audio_input, (str, Path)):
        # librosa.load does 2 things automatically:
        # a) Converts multi-channel (stereo) sound to single-channel (mono).
        # b) Resamples the audio to our target sample rate (22,050 Hz).
        y, _ = librosa.load(audio_input, sr=sr, mono=True)
    elif isinstance(audio_input, np.ndarray):
        # If it's already an array (e.g. from a microphone recording), use it directly.
        y = audio_input.astype(np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=1)  # Convert stereo to mono by averaging channels
    else:
        raise ValueError("audio_input must be a file path string or a numpy array.")

    # 2. Trim leading and trailing silence (quiet pauses before and after speaking).
    # top_db=25 means any sound 25 decibels quieter than the loudest peak is considered silence.
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    # If trimming removed everything (e.g. an entirely silent audio file), revert to original.
    if len(y_trimmed) > 0:
        y = y_trimmed

    # 3. Standardize duration to exactly TOTAL_SAMPLES (66,150 numbers).
    # Why? Neural networks require all inputs to have the exact same shape!
    target_len = int(sr * duration)
    
    if len(y) < target_len:
        # Audio is shorter than 3 seconds: Add silence (zeros) to the end (Padding).
        padding = target_len - len(y)
        y = np.pad(y, (0, padding), mode="constant")
    else:
        # Audio is longer than 3 seconds: Cut off the extra sound (Trimming).
        y = y[:target_len]

    return y


def extract_features(audio_input, sr=SAMPLE_RATE):
    """
    Extracts a comprehensive 1D feature vector from speech audio.
    
    We extract 5 distinct acoustic features:
    1. MFCCs (40 bands): Captures vocal tract shape & timbre.
    2. Chroma (12 semitones): Captures musical pitch and harmonic melody.
    3. Mel-Spectrogram (128 bands): Captures energy across human-audible frequencies.
    4. RMS Energy (1 band): Captures loudness and vocal power.
    5. Zero Crossing Rate (1 band): Captures harshness/unvoiced hissing sounds.
    
    For each feature across time, we compute both the MEAN (average value)
    and STANDARD DEVIATION (how much it changes over time).
    
    Total Feature Vector Size:
    - MFCC: 40 mean + 40 std = 80
    - Chroma: 12 mean + 12 std = 24
    - Mel: 128 mean + 128 std = 256
    - RMS: 1 mean + 1 std = 2
    - ZCR: 1 mean + 1 std = 2
    Total = 80 + 24 + 256 + 2 + 2 = 364 numerical values per audio file!
    
    Returns:
        np.ndarray: 1D array of shape (364,)
    """
    # Step 1: Preprocess audio (load, resample, trim silence, pad to 3 seconds)
    y = load_and_preprocess_audio(audio_input, sr=sr)

    # Step 2: Compute MFCCs (Mel-Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc_mean = np.mean(mfcc, axis=1)  # 40 average values
    mfcc_std = np.std(mfcc, axis=1)    # 40 variance/spread values

    # Step 3: Compute Chroma (12 musical pitch classes)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    chroma_mean = np.mean(chroma, axis=1)  # 12 values
    chroma_std = np.std(chroma, axis=1)    # 12 values

    # Step 4: Compute Mel-Spectrogram (128 frequency bands)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    # Convert amplitude to decibels (log-scale) to match human hearing perception
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)  # 128 values
    mel_std = np.std(mel_db, axis=1)    # 128 values

    # Step 5: Compute RMS (Root Mean Square) Energy - Loudness
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    rms_mean = np.mean(rms, axis=1)  # 1 value
    rms_std = np.std(rms, axis=1)    # 1 value

    # Step 6: Compute Zero-Crossing Rate (ZCR) - Sound wave zero intersections
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    zcr_mean = np.mean(zcr, axis=1)  # 1 value
    zcr_std = np.std(zcr, axis=1)    # 1 value

    # Step 7: Concatenate all features into a single 1D array
    feature_vector = np.concatenate([
        mfcc_mean, mfcc_std,       # 80 values
        chroma_mean, chroma_std,   # 24 values
        mel_mean, mel_std,         # 256 values
        rms_mean, rms_std,         # 2 values
        zcr_mean, zcr_std          # 2 values
    ])

    return feature_vector.astype(np.float32)


def extract_2d_spectrogram(audio_input, sr=SAMPLE_RATE):
    """
    Extracts a 2D Mel-Spectrogram matrix (image of sound) for 2D-CNN models.
    
    Returns:
        np.ndarray: 2D array of shape (N_MELS, TIME_FRAMES), e.g. (128, 130).
    """
    y = load_and_preprocess_audio(audio_input, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)
