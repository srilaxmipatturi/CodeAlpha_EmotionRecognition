"""
=============================================================================
Project: Emotion Recognition from Speech (SER)
File: train.py
Purpose: Complete training pipeline for the Speech Emotion Recognition model.
         Handles data batching, forward pass, loss calculation, backpropagation,
         validation tracking, model checkpointing, and performance curve plotting.
=============================================================================
Why do we need this file?
Before an AI model can recognize emotions, it must be trained. This file
teaches the neural network by showing it examples, measuring its mistakes
(loss), and nudging its weights (optimization) until accuracy reaches its peak.
=============================================================================
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import project configuration, dataset loader, and model architecture
from config import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    BEST_MODEL_PATH,
    OUTPUTS_DIR,
    RANDOM_SEED,
)
from dataset_loader import get_train_test_data
from model import build_model


def set_seed(seed=RANDOM_SEED):
    """Sets random seeds for complete reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_speech_emotion_model(model_type="cnn_lstm", epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    """
    Executes the complete training and validation pipeline.
    
    Parameters:
        model_type (str): 'cnn_lstm' or 'cnn'.
        epochs (int): Number of training passes.
        lr (float): Initial learning rate for Adam optimizer.
        batch_size (int): Mini-batch size.
    """
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # -----------------------------------------------------------------------
    # Step 1: Load and prepare dataset splits
    # -----------------------------------------------------------------------
    print("\n--- Step 1: Loading Processed Dataset ---")
    X_train, X_test, y_train, y_test, scaler = get_train_test_data()

    # Convert NumPy arrays to PyTorch Tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    # PyTorch DataLoaders handle mini-batch shuffling and batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)} | Testing samples: {len(test_dataset)}")

    # -----------------------------------------------------------------------
    # Step 2: Instantiate Model, Loss Function & Optimizer
    # -----------------------------------------------------------------------
    print(f"\n--- Step 2: Initializing Model ({model_type.upper()}) ---")
    model = build_model(model_type=model_type, input_dim=X_train.shape[1]).to(device)

    # CrossEntropyLoss: Combines LogSoftmax and Negative Log-Likelihood Loss.
    # It penalizes confident wrong predictions and rewards confident correct ones.
    criterion = nn.CrossEntropyLoss()

    # Adam Optimizer: Adapts learning rate for each weight parameter.
    # weight_decay (L2 regularization) prevents individual weights from becoming excessively large.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning Rate Scheduler: Reduces learning rate by half if validation loss stalls for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # -----------------------------------------------------------------------
    # Step 3: The Training Loop
    # -----------------------------------------------------------------------
    print(f"\n--- Step 3: Starting Training for {epochs} Epochs ---")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # A) TRAINING PHASE
        model.train()  # Enables Dropout and BatchNorm update
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 1. Zero out leftover gradients from previous step
            optimizer.zero_grad()

            # 2. Forward pass: Model makes predictions
            outputs = model(X_batch)

            # 3. Compute loss: How far was the prediction from ground truth?
            loss = criterion(outputs, y_batch)

            # 4. Backward pass: Compute gradients with respect to each weight
            loss.backward()

            # 5. Optimizer step: Nudge weights in the direction that lowers loss
            optimizer.step()

            # Track training metrics
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # B) VALIDATION PHASE (Evaluation without updating weights)
        model.eval()  # Disables Dropout for deterministic evaluation
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():  # Disables gradient computation to save RAM & CPU
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == y_batch).sum().item()
                total_val += y_batch.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        # Adjust learning rate based on validation loss
        scheduler.step(epoch_val_loss)

        # Record metrics for plotting
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # C) CHECKPOINTING: Save the model if it achieves the highest validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "model_type": model_type,
                },
                BEST_MODEL_PATH,
            )
            saved_indicator = " * [BEST MODEL SAVED]"
        else:
            saved_indicator = ""

        # Print progress every 5 epochs and the final epoch
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or saved_indicator:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"| Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:5.1f}% "
                f"| Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:5.1f}%"
                f"{saved_indicator}"
            )

    print(f"\nTraining Complete! Peak Validation Accuracy: {best_val_acc:.1f}%")
    print(f"Best model weights saved to: {BEST_MODEL_PATH}")

    # -----------------------------------------------------------------------
    # Step 4: Plot and Save Training Curves (Loss & Accuracy)
    # -----------------------------------------------------------------------
    plot_training_curves(history)

    return model, history


def plot_training_curves(history):
    """
    Generates and saves a dual-panel plot showing Loss and Accuracy over Epochs.
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Loss Curve
    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title("Training & Validation Loss Over Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Accuracy Curve
    axes[1].plot(epochs_range, history["train_acc"], label="Train Accuracy", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs_range, history["val_acc"], label="Val Accuracy", color="#d62728", linewidth=2, linestyle="--")
    axes[1].set_title("Training & Validation Accuracy Over Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    curve_path = OUTPUTS_DIR / "loss_accuracy_curve.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Training performance curves saved to: {curve_path}")


if __name__ == "__main__":
    train_speech_emotion_model()
