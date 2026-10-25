"""
=============================================================================
Project: Emotion Recognition from Speech (SER)
File: model.py
Purpose: Deep Learning Architectures in PyTorch for Speech Emotion Recognition.
         Contains:
         1. SpeechEmotionCNN1D (1D Convolutional Neural Network)
         2. SpeechEmotionCNNLSTM (Hybrid CNN + Bidirectional LSTM)
=============================================================================
Why do we need this file?
This file contains the "brain" of our system. It defines the neural network
layers that take the 364 audio features and learn to recognize patterns
unique to each human emotion.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES


# ===========================================================================
# 1. 1D CONVOLUTIONAL NEURAL NETWORK (CNN-1D)
# ===========================================================================
class SpeechEmotionCNN1D(nn.Module):
    """
    A 1D-CNN designed for audio feature vectors.
    
    How it works:
    - Input shape: (Batch_Size, 1, 364)
      Imagine a strip of 364 numbers representing the audio clip.
    - Conv1D layers slide filters across the strip to detect patterns
      (e.g., correlations between pitch, energy, and timbre).
    - MaxPool reduces dimension while preserving the strongest signals.
    - Dropout prevents overfitting by randomly deactivating neurons.
    - Output: Raw class logits of shape (Batch_Size, 8).
    """

    def __init__(self, input_dim=364, num_classes=NUM_CLASSES, dropout_rate=0.3):
        super(SpeechEmotionCNN1D, self).__init__()

        # Block 1: 1 input channel -> 64 filters
        # Kernel size 5 means the filter looks at 5 neighboring features at once
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)       # Stabilizes training by normalizing layer outputs
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Halves the length: 364 -> 182
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: 64 filters -> 128 filters
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # Halves the length: 182 -> 91
        self.drop2 = nn.Dropout(dropout_rate)

        # Block 3: 128 filters -> 256 filters
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        # Adaptive pooling summarizes any remaining length into exactly 1 summary value per filter
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Shape becomes: (Batch, 256, 1)
        self.drop3 = nn.Dropout(dropout_rate)

        # Fully Connected (Dense) Classification Head
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc_drop = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)  # Final 8 emotion output scores

    def forward(self, x):
        # Ensure input has channel dimension: (Batch, 364) -> (Batch, 1, 364)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Forward pass through Conv Blocks
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.global_pool(F.relu(self.bn3(self.conv3(x)))))

        # Flatten from (Batch, 256, 1) -> (Batch, 256)
        x = x.squeeze(-1)

        # Fully Connected Layers
        x = self.fc_drop(self.relu(self.fc1(x)))
        logits = self.fc2(x)
        return logits


# ===========================================================================
# 2. HYBRID CNN + BIDIRECTIONAL LSTM (CNN-LSTM)
# ===========================================================================
class SpeechEmotionCNNLSTM(nn.Module):
    """
    A state-of-the-art Hybrid Model combining CNN and Bidirectional LSTM.
    
    Why this hybrid architecture?
    1. CNN Layers: Extract rich spatial acoustic feature patterns from the input.
    2. Bidirectional LSTM: Scans the sequence in both directions (forward & backward)
       to understand temporal dependencies in emotional speech.
    3. Dense Head: Maps the learned representation into 8 emotion classes.
    """

    def __init__(self, input_dim=364, num_classes=NUM_CLASSES, lstm_hidden=64, lstm_layers=2, dropout_rate=0.3):
        super(SpeechEmotionCNNLSTM, self).__init__()

        # 1. CNN Feature Extractor
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 364 -> 182
        self.drop1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 182 -> 91
        self.drop2 = nn.Dropout(dropout_rate)

        # 2. Bidirectional LSTM
        # Input to LSTM will have shape: (Batch_Size, Sequence_Length=91, Channels=128)
        # bidirectional=True means 2 LSTMs run in parallel (one forward, one backward).
        # Output feature size = lstm_hidden * 2 = 64 * 2 = 128.
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
        )

        # 3. Dense Classifier Head
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.relu = nn.ReLU()
        self.fc_drop = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Ensure shape: (Batch, 1, 364)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Conv feature extraction: (Batch, 1, 364) -> (Batch, 128, 91)
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))

        # Permute dimensions for LSTM: PyTorch LSTM expects (Batch, Seq_Len, Features)
        # (Batch, 128, 91) -> (Batch, 91, 128)
        x = x.permute(0, 2, 1)

        # Pass through Bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # Shape: (Batch, 91, 128)

        # We take the mean across the sequence length (temporal average pooling)
        # Shape becomes: (Batch, 128)
        x = torch.mean(lstm_out, dim=1)

        # Fully connected classification
        x = self.fc_drop(self.relu(self.fc1(x)))
        logits = self.fc2(x)  # Shape: (Batch, 8)
        return logits


# ---------------------------------------------------------------------------
# 3. HELPER FUNCTION: Model Builder & Summary
# ---------------------------------------------------------------------------
def build_model(model_type="cnn_lstm", input_dim=364, num_classes=NUM_CLASSES):
    """
    Factory function to instantiate the chosen model architecture.
    """
    if model_type.lower() == "cnn":
        model = SpeechEmotionCNN1D(input_dim=input_dim, num_classes=num_classes)
    elif model_type.lower() == "cnn_lstm":
        model = SpeechEmotionCNNLSTM(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'cnn' or 'cnn_lstm'.")
        
    return model


# ---------------------------------------------------------------------------
# Quick Verification when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("Testing Stage 4: Deep Learning Model Architecture Verification")
    print("=" * 65)

    # Simulate a mini-batch of 4 audio samples with 364 features each
    dummy_input = torch.randn(4, 364)
    print(f"Input batch shape: {dummy_input.shape} (4 audio samples x 364 features)")

    # 1. Test 1D-CNN
    cnn_model = build_model("cnn")
    cnn_output = cnn_model(dummy_input)
    cnn_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
    print(f"\n1. SpeechEmotionCNN1D:")
    print(f"   Output logits shape: {cnn_output.shape} (4 samples x 8 classes)")
    print(f"   Trainable parameters: {cnn_params:,}")

    # 2. Test Hybrid CNN-LSTM
    hybrid_model = build_model("cnn_lstm")
    hybrid_output = hybrid_model(dummy_input)
    hybrid_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    print(f"\n2. SpeechEmotionCNNLSTM (Hybrid):")
    print(f"   Output logits shape: {hybrid_output.shape} (4 samples x 8 classes)")
    print(f"   Trainable parameters: {hybrid_params:,}")

    # 3. Verify Softmax Probabilities
    probs = F.softmax(hybrid_output, dim=1)
    sample_probs = probs[0].detach().numpy()
    print(f"\n3. Probability Verification (Sample 1 across 8 emotions):")
    print(f"   Percentages sum to: {sample_probs.sum():.4f} (100%)")
    print(f"   Sample probabilities: {torch.round(probs[0] * 100, decimals=1).tolist()}%")

    print("=" * 65)
    print("ALL MODEL ARCHITECTURES VERIFIED AND WORKING PROPERLY!")
    print("=" * 65)
