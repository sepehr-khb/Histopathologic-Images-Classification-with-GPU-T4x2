# src/utils.py

"""
Utility functions for training, evaluation, and visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def save_history(history, filepath):
    """
    Save training history (loss/accuracy curves) to a compressed npz file.

    Args:
        history (keras.callbacks.History): History object from model.fit()
        filepath (str): Path to save history (.npz)
    """
    np.savez_compressed(filepath, **history.history)
    print(f"✅ Training history saved to {filepath}")


def load_history(filepath):
    """
    Load training history from npz file.

    Args:
        filepath (str): Path to saved history
    Returns:
        dict: Dictionary with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")
    data = np.load(filepath, allow_pickle=True)
    return dict(data.items())


def plot_history(history_dict, save_path=None):
    """
    Plot loss and accuracy curves from training history.

    Args:
        history_dict (dict): Loaded history dictionary
        save_path (str, optional): If provided, saves the plot instead of showing
    """
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_dict["loss"], label="train_loss")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    if "accuracy" in history_dict:
        plt.plot(history_dict["accuracy"], label="train_acc")
    if "val_accuracy" in history_dict:
        plt.plot(history_dict["val_accuracy"], label="val_acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()


def save_predictions_to_csv(ids, predictions, filepath):
    """
    Save predictions to CSV file with columns [id, label].

    Args:
        ids (list/np.array): Sample IDs
        predictions (np.array): Predicted labels (0/1)
        filepath (str): Path to save CSV
    """
    df = pd.DataFrame({"id": ids, "label": predictions.astype(int)})
    df.to_csv(filepath, index=False)
    print(f"✅ Predictions saved to {filepath}")
