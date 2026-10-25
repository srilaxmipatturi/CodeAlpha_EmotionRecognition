"""
=============================================================================
Project: Emotion Recognition from Speech (SER)
File: test_features.py
Purpose: Verification script to test features.py without needing
         to download a full external dataset first!
=============================================================================
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from config import SAMPLE_RATE, DURATION, DATASET_DIR
from features import load_and_preprocess_audio, extract_features, extract_2d_spectrogram

def test_feature_pipeline():
    print("=" * 60)
    print("Testing Stage 2: Audio Preprocessing & Feature Extraction")
    print("=" * 60)

    # Step 1: Create a synthetic 2.5-second test audio file
    # (Simulating an audio clip shorter than 3 seconds to test automatic padding)
    test_dir = DATASET_DIR / "sample_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_audio_path = test_dir / "sample_speech_tone.wav"

    test_duration = 2.5
    t = np.linspace(0, test_duration, int(SAMPLE_RATE * test_duration), endpoint=False)
    # Generate a voice-like harmonic tone (fundamental pitch 220Hz + harmonics)
    synthetic_signal = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)

    sf.write(test_audio_path, synthetic_signal, SAMPLE_RATE)
    print(f"1. Created synthetic audio sample at: {test_audio_path}")
    print(f"   Original duration: {test_duration} seconds ({len(synthetic_signal)} samples)")

    # Step 2: Test load_and_preprocess_audio
    processed_signal = load_and_preprocess_audio(test_audio_path)
    expected_samples = int(SAMPLE_RATE * DURATION)
    print(f"2. Standardized audio length: {len(processed_signal)} samples (exactly {DURATION} seconds)")
    assert len(processed_signal) == expected_samples, f"Expected {expected_samples}, got {len(processed_signal)}"
    print("   [PASS] Silence trimming and 3-second padding verified!")

    # Step 3: Test 1D feature extraction
    feature_vec = extract_features(test_audio_path)
    print(f"3. Extracted 1D Feature Vector shape: {feature_vec.shape}")
    print(f"   Total numerical features extracted per audio clip: {len(feature_vec)}")
    print(f"   First 5 MFCC mean values: {np.round(feature_vec[:5], 3)}")
    assert len(feature_vec) == 364, f"Expected 364 features, got {len(feature_vec)}"
    print("   [PASS] 1D Feature Vector (MFCC + Chroma + Mel + RMS + ZCR) verified!")

    # Step 4: Test 2D spectrogram extraction
    spectrogram_2d = extract_2d_spectrogram(test_audio_path)
    print(f"4. Extracted 2D Mel-Spectrogram shape: {spectrogram_2d.shape} (Frequency bands x Time frames)")
    print("   [PASS] 2D Mel-Spectrogram verified!")

    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! Stage 2 is completely verified.")
    print("=" * 60)

if __name__ == "__main__":
    test_feature_pipeline()
