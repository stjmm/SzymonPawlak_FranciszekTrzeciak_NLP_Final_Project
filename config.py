from dataclasses import dataclass
from typing import List


# This one was used for Herbert-base-cased
@dataclass
class Config:
    # Data
    data_dir: str = "data"
    max_length: int = 256

    # Training
    num_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    label_smoothing: float = 0.05

    # LoRA settings
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Early stopping
    early_stopping_patience: int = 2

    # Cross-validation
    n_folds: int = 3
    seed: int = 42

    # Logging
    output_dir: str = "outputs"
    logging_steps: int = 25
    save_checkpoints: bool = True


MODELS = {
    "polish-roberta": {
        "name": "sdadas/polish-roberta-base-v2",
        "type": "roberta",
    },
    "herbert": {
        "name": "allegro/herbert-base-cased",
        "type": "bert",
    },
    "deberta": {
        "name": "microsoft/deberta-v3-base",
        "type": "deberta",
    },
}

# THESE WERE THE SETTINGS FOR POLISH_ROBERTA MODEL
# # Data
# data_dir: str = "data"
# max_length: int = 256
#
# # Training
# num_epochs: int = 1
# batch_size: int = 128
# learning_rate: float = 2e-4
# weight_decay: float = 0.01
# warmup_ratio: float = 0.1
# label_smoothing: float = 0.1
#
# # LoRA settings
# lora_r: int = 16
# lora_alpha: int = 32
# lora_dropout: float = 0.1
#
# # Early stopping
# early_stopping_patience: int = 3
#
# # Cross-validation
# n_folds: int = 5
# seed: int = 42
#
# # Logging
# output_dir: str = "outputs"
# logging_steps: int = 50
# save_checkpoints: bool = True


# CONFIG FOR DEBERTAV3-BASE
# # Data
# data_dir: str = "data"
# max_length: int = 256
#
# # Training
# num_epochs: int = 1
# batch_size: int = 32
# learning_rate: float = 2e-4
# weight_decay: float = 0.01
# warmup_ratio: float = 0.06
# label_smoothing: float = 0.05
#
# # LoRA settings
# lora_r: int = 32
# lora_alpha: int = 64
# lora_dropout: float = 0.05
#
# # Early stopping
# early_stopping_patience: int = 2
#
# # Cross-validation
# n_folds: int = 5
# seed: int = 42
#
# # Logging
# output_dir: str = "outputs"
# logging_steps: int = 25
# save_checkpoints: bool = True
