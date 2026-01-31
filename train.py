import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import MODELS, Config
from data import TextDataset, create_folds, load_data
from ensemble import ensemble_predict, optimize_weights
from model import create_model_and_tokenizer
from trainer import Trainer


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_model(
    model_key: str,
    model_info: dict,
    texts: list,
    labels: list,
    folds: list,
    config: Config,
) -> None:
    """Trains a single model with cross-validation and saves results"""

    print(f"Training Task: {model_key}")
    print(f"Model Architecture: {model_info['name']}")

    output_dir = Path(config.output_dir) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # Storage for OOF predictions
    # Shape: (n_samples, n_classes)
    oof_probs = np.zeros((len(texts), 2))
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{len(folds)} ---")

        # Initialize model fresh for each fold
        model, tokenizer = create_model_and_tokenizer(
            model_name=model_info["name"],
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            use_4bit=True,
        )

        # Prepare data subsets
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_dataset = TextDataset(
            train_texts, train_labels, tokenizer, config.max_length
        )
        val_dataset = TextDataset(val_texts, val_labels, tokenizer, config.max_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=str(output_dir),
            fold_idx=fold_idx,
        )

        results = trainer.train()

        # Store OOF predictions
        oof_probs[val_idx] = results["final_probs"]
        fold_accuracies.append(results["best_accuracy"])

        print(f"Fold {fold_idx + 1} Best Accuracy: {results['best_accuracy']:.4f}")

        # Cleanup to free VRAM for next fold
        del model, trainer, tokenizer
        torch.cuda.empty_cache()

    # Calculate Overall Metrics
    oof_preds = np.argmax(oof_probs, axis=1)
    overall_accuracy = np.mean(oof_preds == labels)

    print(f"\n{model_key} Completed:")
    print(f"  CV Accuracy: {overall_accuracy:.4f} (+/- {np.std(fold_accuracies):.4f})")

    np.save(output_dir / "oof_probs.npy", oof_probs)

    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "model": model_info["name"],
                "cv_accuracy": overall_accuracy,
                "fold_accuracies": fold_accuracies,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SMIGIEL Training")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        help="Name of the specific model to train (see config.py)",
    )
    args = parser.parse_args()

    # Setup
    config = Config()

    # Ensure strict reproducibility across different runs
    set_seed(config.seed)

    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load Data & Recreate Folds
    print("\nLoading data and verifying split consistency...")
    texts, labels = load_data(config.data_dir)
    labels = np.array(labels)
    folds = create_folds(labels.tolist(), config.n_folds, config.seed)

    if args.model:
        # Train specific model
        model_info = MODELS[args.model]
        train_single_model(
            model_key=args.model,
            model_info=model_info,
            texts=texts,
            labels=labels.tolist(),
            folds=folds,
            config=config,
        )

    else:
        print("\nSpecify an action:")
        print(f"  Train a model:   python train.py --model [{'|'.join(MODELS.keys())}]")
        sys.exit(1)


if __name__ == "__main__":
    main()
