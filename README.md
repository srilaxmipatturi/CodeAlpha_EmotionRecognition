# 🎙️ Emotion Recognition from Speech (SER)
### Deep Learning and Speech Signal Processing using PyTorch & Librosa

Recognize human emotions (e.g., *Happy, Sad, Angry, Fearful, Calm, Neutral, Disgust, Surprised*) directly from human voice recordings using modern acoustic signal processing and deep neural networks.

---

## 📌 Project Overview
* **Objective:** Classify human emotional state from raw `.wav` speech audio.
* **Techniques:**
  * **Speech Signal Processing:** Mel-Frequency Cepstral Coefficients (MFCCs), Chroma, Mel-Spectrogram, RMS Energy, Zero-Crossing Rate (ZCR).
  * **Deep Learning:** Hybrid 1D-CNN + Bidirectional LSTM (BiLSTM) with Batch Normalization and Dropout.
  * **Web Application:** Modern interactive Streamlit GUI with live audio waveform and spectrogram visualization.

---

## 📂 Project Directory Structure

```text
CodeAlpha_Emotion Recognition from Speech/
│
├── dataset/                  # Speech datasets and feature cache
│   ├── starter_speech/       # Built-in acoustic starter samples (120 clips across 8 emotions)
│   ├── sample_test/          # Quick test audio files
│   └── features_cache.npz    # Compressed cache for instant reloading
│
├── models/                   # Saved model artifacts
│   ├── best_ser_model.pth    # Trained PyTorch neural network checkpoint
│   └── scaler.joblib         # Fitted StandardScaler for single-audio normalization
│
├── outputs/                  # High-resolution evaluation plots
│   ├── loss_accuracy_curve.png # Training & validation loss/accuracy across epochs
│   └── confusion_matrix.png    # 8x8 Confusion Matrix heatmap
│
├── config.py                 # Central configurations (sample rate, hop length, emotions)
├── features.py               # Audio preprocessing & 364-dimensional feature extraction
├── test_features.py          # Self-contained feature pipeline verification test
├── dataset_loader.py         # Dataset scanner, label parser, and stratified train/test splitter
├── model.py                  # PyTorch 1D-CNN and Hybrid CNN-LSTM architectures
├── train.py                  # Training pipeline with validation tracking & checkpointing
├── evaluate.py               # Performance evaluation, classification report & confusion matrix
├── predict.py                # Command-line single-audio prediction utility
├── app.py                    # Streamlit interactive Web Application
├── requirements.txt          # Python library dependencies
└── README.md                 # Complete documentation & project guide
```

---

## 🚀 Step-by-Step Execution Guide

### 1. Open the Project in VS Code
Open VS Code, click **File > Open Folder**, and select:
```text
C:\Users\ankit\OneDrive\Desktop\CodeAlpha_Emotion Recognition from Speech
```
Open the integrated terminal with ``Ctrl + ` ``.

### 2. Verify Audio Feature Extraction
Test the signal processing pipeline with the built-in test script:
```powershell
python test_features.py
```

### 3. Load Dataset & Cache Features
Scan and cache the dataset features:
```powershell
python dataset_loader.py
```

### 4. Train the Neural Network
Train the Hybrid CNN-LSTM model for 40 epochs:
```powershell
python train.py
```
*Outputs: Saves best weights to `models/best_ser_model.pth` and curves to `outputs/loss_accuracy_curve.png`.*

### 5. Evaluate the Model
Generate the Classification Report and Confusion Matrix:
```powershell
python evaluate.py
```
*Outputs: Saves heatmap to `outputs/confusion_matrix.png`.*

### 6. Predict Emotion for Any Audio (CLI)
Run inference from the command line:
```powershell
python predict.py --audio "dataset/starter_speech/happy/happy_sample_01.wav"
python predict.py --audio "dataset/starter_speech/angry/angry_sample_01.wav"
```

### 7. Launch the Interactive Web Application
Start the Streamlit Web UI:
```powershell
streamlit run app.py
```
*Your browser will automatically open to `http://localhost:8501`!*

---

## 🎧 Adding External Datasets (RAVDESS or TESS)

You can drop real human actor audio files into the `dataset/` folder:
1. **RAVDESS Dataset:** Download from Kaggle or Zenodo (`Audio_Speech_Actors_01-24.zip`). Extract the `.wav` files into `dataset/` (e.g., `dataset/Actor_01/03-01-05-01-01-01-01.wav`).
2. **TESS Dataset:** Extract the emotion folders into `dataset/` (e.g., `dataset/OAF_angry/`).
3. Delete `dataset/features_cache.npz` so the loader knows to process the new files:
   ```powershell
   python -c "import os; os.remove('dataset/features_cache.npz') if os.path.exists('dataset/features_cache.npz') else None"
   python train.py
   ```
The loader will automatically detect the RAVDESS/TESS naming patterns and train on the real human speech data!


