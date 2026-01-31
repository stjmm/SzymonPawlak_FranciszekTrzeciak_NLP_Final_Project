from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


def load_data(data_dir: str) -> Tuple[List[str], List[int]]:
    """Loads training data"""
    data_dir = Path(data_dir)

    with open(data_dir / "train" / "data.tsv", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    with open(data_dir / "train" / "labels.tsv", "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: {np.bincount(labels)}")
    return texts, labels


def load_test_data(data_dir: str, test_name: str = "test_A") -> List[str]:
    """Loads test data"""
    data_dir = Path(data_dir)

    with open(data_dir / "test" / test_name / "data.tsv", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(texts)} test samples")
    return texts


def create_folds(labels: List[int], n_folds: int = 5, seed: int = 42):
    """Stratified k-fold splits"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(range(len(labels)), labels))


class TextDataset(Dataset):
    """Text dataset class"""

    def __init__(
        self, texts: List[str], labels: Optional[List[int]], tokenizer, max_length: int
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
