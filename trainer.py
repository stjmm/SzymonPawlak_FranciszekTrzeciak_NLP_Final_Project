import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from pathlib import Path
import numpy as np
import json
import csv
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.counter >= self.patience


class Trainer:
    """Trainer with detailed logging for nice plots."""
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        output_dir: str,
        fold_idx: int = 0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.fold_idx = fold_idx
        
        # Setup directories
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints" / f"fold_{fold_idx}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard - log more frequently
        log_dir = self.output_dir / "logs" / f"fold_{fold_idx}"
        self.writer = SummaryWriter(log_dir)
        
        # CSV logger for detailed step-by-step metrics
        self.log_file = self.output_dir / "logs" / f"fold_{fold_idx}_metrics.csv"
        self.step_log_file = self.output_dir / "logs" / f"fold_{fold_idx}_steps.csv"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_csv_loggers()
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        
        # Tracking
        self.global_step = 0
        self.best_accuracy = 0
        self.history = {
            "train_loss": [], 
            "val_loss": [], 
            "val_accuracy": [], 
            "lr": [],
            "steps": [],
            "step_losses": [],
        }
    
    def _init_csv_loggers(self):
        """Initialize CSV log files."""
        # Epoch-level metrics
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "train_loss", "val_loss", "val_accuracy", "lr"])
        
        # Step-level metrics (for detailed curves)
        with open(self.step_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "lr"])
    
    def _log_step(self, step: int, loss: float, lr: float):
        """Log every N steps for smooth curves."""
        # CSV
        with open(self.step_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, loss, lr])
        
        # TensorBoard
        self.writer.add_scalar("Train/loss_step", loss, step)
        self.writer.add_scalar("Train/lr", lr, step)
    
    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float, val_acc: float, lr: float):
        """Log epoch-level metrics."""
        # CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, self.global_step, train_loss, val_loss, val_acc, lr])
        
        # TensorBoard - epoch metrics
        self.writer.add_scalar("Epoch/train_loss", train_loss, epoch)
        self.writer.add_scalar("Epoch/val_loss", val_loss, epoch)
        self.writer.add_scalar("Epoch/val_accuracy", val_acc, epoch)
        self.writer.add_scalar("Epoch/learning_rate", lr, epoch)
        
        # Also add to combined graph
        self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        
        # History
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_acc)
        self.history["lr"].append(lr)
    
    def train_epoch(self) -> float:
        """Train for one epoch with detailed logging."""
        self.model.train()
        total_loss = 0
        step_losses = []
        
        progress = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress):
            # Move to device
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Forward
            outputs = self.model(**batch)
            
            # Label smoothing
            if self.config.label_smoothing > 0:
                logits = outputs.logits
                labels = batch["labels"]
                loss = self._label_smoothing_loss(logits, labels, self.config.label_smoothing)
            else:
                loss = outputs.loss
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            loss_val = loss.item()
            total_loss += loss_val
            step_losses.append(loss_val)
            self.global_step += 1
            
            # Log every N steps for smooth curves
            lr = self.scheduler.get_last_lr()[0]
            if self.global_step % self.config.logging_steps == 0:
                avg_recent_loss = np.mean(step_losses[-self.config.logging_steps:])
                self._log_step(self.global_step, avg_recent_loss, lr)
            
            progress.set_postfix({
                "loss": f"{loss_val:.4f}", 
                "avg": f"{np.mean(step_losses[-100:]):.4f}",
                "lr": f"{lr:.2e}"
            })
        
        return total_loss / len(self.train_loader)
    
    def _label_smoothing_loss(self, logits, labels, smoothing: float):
        """Cross entropy with label smoothing."""
        n_classes = logits.size(-1)
        one_hot = torch.zeros_like(logits).scatter(1, labels.unsqueeze(1), 1)
        one_hot = one_hot * (1 - smoothing) + smoothing / n_classes
        log_probs = F.log_softmax(logits, dim=-1)
        return -(one_hot * log_probs).sum(dim=-1).mean()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = {k: v.cuda() for k, v in batch.items()}
            
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            
            probs = F.softmax(outputs.logits, dim=-1)
            preds = outputs.logits.argmax(dim=-1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
        
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        
        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": accuracy,
            "probs": all_probs,
            "preds": all_preds,
            "labels": all_labels,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.config.save_checkpoints:
            return
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_accuracy": self.best_accuracy,
            "history": self.history,
        }
        
        # Save LoRA adapter weights only
        self.model.save_pretrained(self.checkpoint_dir / f"epoch_{epoch}")
        torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch}" / "training_state.pt")
        
        if is_best:
            self.model.save_pretrained(self.checkpoint_dir / "best")
            torch.save(checkpoint, self.checkpoint_dir / "best" / "training_state.pt")
    
    def train(self) -> Dict:
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Training Fold {self.fold_idx + 1}")
        print(f"{'='*60}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Total steps: {len(self.train_loader) * self.config.num_epochs}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            val_results = self.evaluate()
            val_loss = val_results["loss"]
            val_acc = val_results["accuracy"]
            
            # Log
            lr = self.scheduler.get_last_lr()[0]
            self._log_epoch(epoch, train_loss, val_loss, val_acc, lr)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_acc:.4f}")
            
            # Checkpoint
            is_best = val_acc > self.best_accuracy
            if is_best:
                self.best_accuracy = val_acc
                print(f"  ✓ New best accuracy!")
            
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.early_stopping(val_acc):
                print(f"\n⚠️ Early stopping at epoch {epoch + 1}")
                break
        
        self.writer.close()
        
        return {
            "best_accuracy": self.best_accuracy,
            "history": self.history,
            "final_probs": val_results["probs"],
            "final_preds": val_results["preds"],
        }
