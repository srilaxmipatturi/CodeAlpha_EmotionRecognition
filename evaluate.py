"""
=============================================================================
Project: Emotion Recognition from Speech (SER)
File: evaluate.py
Purpose: Independent evaluation module for the Speech Emotion Recognition model.
         Calculates Accuracy, Precision, Recall, F1-score, and generates
         a high-resolution Confusion Matrix heatmap.
=============================================================================
Why do we need this file?
Training is only half the battle. We need to critically evaluate where our model
excels and where it makes mistakes. This file generates the exact charts and
metrics you will present in your project report and viva.
=============================================================================
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import configuration and loader
from config import (
    BEST_MODEL_PATH,
    OUTPUTS_DIR,
    EMOTIONS,
    NUM_CLASSES,
)
from dataset_loader import get_train_test_data
from model import build_model


def evaluate_model():
    """
    Loads the best saved model checkpoint and evaluates it on the held-out test set.
    """
    print("=" * 65)
    print("Evaluating Speech Emotion Recognition Model")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Check if trained model weights exist
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {BEST_MODEL_PATH}. Please run 'python train.py' first!"
        )

    # Step 2: Load the test set
    print("\n1. Loading held-out test dataset...")
    _, X_test, _, y_test, _ = get_train_test_data()

    # Step 3: Load the trained model checkpoint
    print(f"2. Loading model weights from: {BEST_MODEL_PATH}")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "cnn_lstm")
    
    model = build_model(model_type=model_type, input_dim=X_test.shape[1], num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Step 4: Perform Inference on the Test Set
    print("3. Performing inference across test samples...")
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

    # Step 5: Compute Evaluation Metrics
    accuracy = accuracy_score(y_test, predictions) * 100.0
    print(f"\n========================================================")
    print(f"  OVERALL TEST ACCURACY: {accuracy:.2f}%")
    print(f"========================================================\n")

    # Step 6: Detailed Classification Report
    present_classes = sorted(list(set(y_test) | set(predictions)))
    target_names = [EMOTIONS[i] for i in present_classes]

    report = classification_report(
        y_test, predictions, labels=present_classes, target_names=target_names, digits=3
    )
    print("Detailed Classification Report (Precision, Recall, F1-Score):")
    print("-" * 65)
    print(report)
    print("-" * 65)

    # Step 7: Confusion Matrix Visualization
    plot_confusion_matrix(y_test, predictions, present_classes, target_names)

    return accuracy, predictions


def plot_confusion_matrix(y_true, y_pred, present_classes, target_names):
    """
    Computes, visualizes, and saves a Confusion Matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True,
        linewidths=1,
        linecolor="#cccccc",
        annot_kws={"size": 11, "weight": "bold"},
    )

    plt.title("Speech Emotion Recognition — Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Emotion Label", fontsize=11, fontweight="bold")
    plt.ylabel("Actual True Emotion Label", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion Matrix plot successfully saved to: {cm_path}")


if __name__ == "__main__":
    evaluate_model()
