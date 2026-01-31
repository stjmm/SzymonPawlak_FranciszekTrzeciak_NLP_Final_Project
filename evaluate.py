import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MODELS, Config
from data import TextDataset, load_data
from model import create_model_and_tokenizer


def evaluate_model(model_key: str):
    config = Config()
    output_dir = Path(config.output_dir) / model_key

    # Load OOF predictions
    oof_probs = np.load(output_dir / "oof_probs.npy")

    # Load labels
    _, labels = load_data(config.data_dir)
    labels = np.array(labels)

    # Calculate metrics
    oof_preds = np.argmax(oof_probs, axis=1)
    accuracy = np.mean(oof_preds == labels)

    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"Evaluation: {model_key}")
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(
        classification_report(labels, oof_preds, target_names=["Human (0)", "LLM (1)"])
    )
    print(f"Confusion Matrix:")
    print(confusion_matrix(labels, oof_preds))

    # Load fold results
    with open(output_dir / "results.json") as f:
        results = json.load(f)
    print(f"\nFold Accuracies: {results['fold_accuracies']}")


if __name__ == "__main__":
    import sys

    model_key = sys.argv[1] if len(sys.argv) > 1 else "polish-roberta"
    evaluate_model(model_key)
