"""
Project Nexus — RefLens DANN Training Script
=============================================
Full training loop: DANN domain adaptation + event classification.
Includes: LR scheduling, gradient clipping, early stopping,
          checkpoint saving, per-class F1, confusion matrix.

Usage
-----
  # Train on synthetic data (no GPU required for testing)
  python train_reflens.py --source_dir dataset/synthetic_sequences \\
                          --target_dir dataset/synthetic_sequences \\
                          --epochs 30 --batch_size 32 --demo

  # Train on real data with GPU
  python train_reflens.py --source_dir dataset/stadium_a \\
                          --target_dir dataset/stadium_b \\
                          --epochs 50 --lr 3e-4 --device cuda

Author  : Sanchayan (Unwilling-mcu)
GitHub  : github.com/Unwilling-mcu/ProjectNexus
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("RefLensTrainer")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from reflens_pipeline import RefLensModel, EVENT_CLASSES
from dataset_pipeline import (
    NexusDataset, NexusDataModule, SyntheticDataGenerator,
    LABEL_EVENT_MAP, report_dataset_stats
)


# ─────────────────────────────────────────────────────────────────
# 1. METRICS
# ─────────────────────────────────────────────────────────────────

def compute_metrics(all_preds: list, all_labels: list,
                    n_classes: int = 5) -> dict:
    """
    Returns dict with per-class precision, recall, F1, and macro averages.
    Pure numpy — no sklearn dependency required.
    """
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    acc    = float((preds == labels).mean())

    per_class = {}
    macro_p = macro_r = macro_f1 = 0.0

    for c in range(n_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        per_class[LABEL_EVENT_MAP[c]] = {"precision": round(p,3),
                                          "recall":    round(r,3),
                                          "f1":        round(f1,3),
                                          "support":   int((labels==c).sum())}
        macro_p  += p
        macro_r  += r
        macro_f1 += f1

    return {
        "accuracy":     round(acc,   4),
        "macro_precision": round(macro_p  / n_classes, 4),
        "macro_recall":    round(macro_r  / n_classes, 4),
        "macro_f1":        round(macro_f1 / n_classes, 4),
        "per_class":    per_class,
    }


def confusion_matrix_str(all_preds: list, all_labels: list,
                          n_classes: int = 5) -> str:
    """Returns ASCII confusion matrix string."""
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    cm     = np.zeros((n_classes, n_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1

    names = [LABEL_EVENT_MAP[i] for i in range(n_classes)]
    col_w = max(len(n) for n in names) + 2

    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in names)
    lines  = [header, "-" * len(header)]
    for i, name in enumerate(names):
        row = f"{name:>{col_w}}" + "".join(f"{cm[i,j]:>{col_w}}" for j in range(n_classes))
        lines.append(row)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 2. CHECKPOINT MANAGER
# ─────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Saves model checkpoints. Keeps only the best and last.
    Best is determined by validation macro-F1.
    """

    def __init__(self, checkpoint_dir: str):
        self.dir      = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_f1  = -1.0
        self.best_path: Optional[Path] = None

    def save(self, model, optimizer, epoch: int, metrics: dict,
             is_last: bool = False) -> str:
        state = {
            "epoch":     epoch,
            "metrics":   metrics,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        f1 = metrics.get("macro_f1", 0.0)
        saved = []

        if f1 > self.best_f1:
            self.best_f1  = f1
            best = self.dir / "reflens_best.pt"
            if TORCH_OK:
                torch.save(state, best)
            self.best_path = best
            saved.append(f"best (F1={f1:.4f})")

        if is_last:
            last = self.dir / "reflens_last.pt"
            if TORCH_OK:
                torch.save(state, last)
            saved.append("last")

        return ", ".join(saved) if saved else ""


# ─────────────────────────────────────────────────────────────────
# 3. TRAINER
# ─────────────────────────────────────────────────────────────────

class RefLensTrainer:
    """
    Full DANN training loop for RefLens.

    Parameters
    ----------
    model           : RefLensModel instance
    source_loader   : DataLoader for labelled source-domain data
    target_loader   : DataLoader for unlabelled target-domain data
    val_loader      : DataLoader for validation (source domain held-out)
    device          : 'cuda' | 'mps' | 'cpu'
    lr              : initial learning rate
    weight_decay    : AdamW weight decay
    max_epochs      : training budget
    patience        : early-stopping patience (epochs without val F1 improvement)
    checkpoint_dir  : where to save checkpoints
    lam_max         : maximum DANN lambda (annealed from 0 → lam_max)
    class_weights   : optional per-class loss weights (Tensor, shape [5])
    """

    def __init__(self, model, source_loader, target_loader, val_loader,
                 device: str = "cpu", lr: float = 3e-4,
                 weight_decay: float = 1e-4, max_epochs: int = 50,
                 patience: int = 8, checkpoint_dir: str = "checkpoints",
                 lam_max: float = 1.0, class_weights=None):

        if not TORCH_OK:
            raise RuntimeError("PyTorch required for training.")

        self.model          = model.to(device)
        self.source_loader  = source_loader
        self.target_loader  = target_loader
        self.val_loader     = val_loader
        self.device         = device
        self.max_epochs     = max_epochs
        self.patience       = patience
        self.lam_max        = lam_max
        self.ckpt           = CheckpointManager(checkpoint_dir)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=lr * 0.01
        )

        self.event_criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
        self.domain_criterion = nn.CrossEntropyLoss()

        self.history: list[dict] = []

    # ── Lambda schedule (Ganin et al.) ────────────────────────────
    def _lam(self, epoch: int) -> float:
        p = epoch / max(self.max_epochs - 1, 1)
        return self.lam_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)

    # ── Training epoch ────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        lam = self._lam(epoch)

        task_losses, domain_losses, total_losses = [], [], []
        target_iter = iter(self.target_loader)

        for x_src, y_ev, y_dom_src in self.source_loader:
            x_src   = x_src.to(self.device)
            y_ev    = y_ev.to(self.device)
            y_dom_src = y_dom_src.to(self.device)

            try:
                x_tgt, _, y_dom_tgt = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader)
                x_tgt, _, y_dom_tgt = next(target_iter)
            x_tgt     = x_tgt.to(self.device)
            y_dom_tgt = y_dom_tgt.to(self.device)

            self.optimizer.zero_grad()

            # Source: event + domain
            ev_logits, dom_logits_src, _ = self.model(x_src, lam)
            # Target: domain only (no event supervision)
            _, dom_logits_tgt, _         = self.model(x_tgt, lam)

            task_loss   = self.event_criterion(ev_logits, y_ev)
            domain_loss = self.domain_criterion(
                torch.cat([dom_logits_src, dom_logits_tgt]),
                torch.cat([y_dom_src, y_dom_tgt])
            )
            loss = task_loss + lam * 0.3 * domain_loss

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            task_losses.append(task_loss.item())
            domain_losses.append(domain_loss.item())
            total_losses.append(loss.item())

        return {
            "task_loss":   round(float(np.mean(task_losses)),   4),
            "domain_loss": round(float(np.mean(domain_losses)), 4),
            "total_loss":  round(float(np.mean(total_losses)),  4),
            "lam":         round(lam, 4),
        }

    # ── Validation epoch ──────────────────────────────────────────
    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.model.eval()
        all_preds, all_labels, losses = [], [], []

        for x, y_ev, _ in self.val_loader:
            x    = x.to(self.device)
            y_ev = y_ev.to(self.device)

            logits, _, _ = self.model(x, lam=0.0)
            loss = self.event_criterion(logits, y_ev)
            losses.append(loss.item())

            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y_ev.cpu().tolist())

        metrics = compute_metrics(all_preds, all_labels)
        metrics["val_loss"] = round(float(np.mean(losses)), 4)
        return metrics

    # ── Full training run ─────────────────────────────────────────
    def fit(self):
        log.info(f"[Trainer] Starting training: {self.max_epochs} epochs "
                 f"on {self.device}")
        no_improve = 0

        for epoch in range(self.max_epochs):
            t0 = time.time()

            train_stats = self._train_epoch(epoch)
            val_stats   = self._val_epoch()

            self.scheduler.step()
            elapsed = time.time() - t0

            row = {"epoch": epoch+1, **train_stats, **val_stats,
                   "lr": self.scheduler.get_last_lr()[0]}
            self.history.append(row)

            is_last = (epoch == self.max_epochs - 1)
            saved   = self.ckpt.save(self.model, self.optimizer,
                                     epoch, val_stats, is_last)

            log.info(
                f"Epoch {epoch+1:>3}/{self.max_epochs} | "
                f"loss={train_stats['total_loss']:.4f} | "
                f"val_loss={val_stats['val_loss']:.4f} | "
                f"val_F1={val_stats['macro_f1']:.4f} | "
                f"acc={val_stats['accuracy']:.4f} | "
                f"λ={train_stats['lam']:.3f} | "
                f"{elapsed:.1f}s"
                + (f" → saved {saved}" if saved else "")
            )

            # Early stopping
            if saved and "best" in saved:
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                log.info(f"[Trainer] Early stopping at epoch {epoch+1} "
                         f"(no improvement for {self.patience} epochs)")
                break

        return self.finalize()

    def finalize(self) -> dict:
        """Print final confusion matrix and return training history."""
        self.model.eval()
        all_preds, all_labels = [], []

        if not TORCH_OK:
            return {}

        with torch.no_grad():
            for x, y_ev, _ in self.val_loader:
                x    = x.to(self.device)
                logits, _, _ = self.model(x, lam=0.0)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(y_ev.tolist())

        cm_str  = confusion_matrix_str(all_preds, all_labels)
        metrics = compute_metrics(all_preds, all_labels)

        print("\n" + "="*55)
        print("  RefLens Final Validation Metrics")
        print("="*55)
        print(f"  Accuracy   : {metrics['accuracy']:.4f}")
        print(f"  Macro F1   : {metrics['macro_f1']:.4f}")
        print(f"  Macro P    : {metrics['macro_precision']:.4f}")
        print(f"  Macro R    : {metrics['macro_recall']:.4f}")
        print("\n  Per-class F1:")
        for ev, m in metrics["per_class"].items():
            print(f"    {ev:<12} F1={m['f1']:.3f}  P={m['precision']:.3f}"
                  f"  R={m['recall']:.3f}  n={m['support']}")
        print("\n  Confusion matrix (rows=true, cols=pred):")
        print(cm_str)
        print("="*55 + "\n")

        summary = {"final_metrics": metrics, "history": self.history}
        out_path = Path(self.ckpt.dir) / "training_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"[Trainer] Summary saved → {out_path}")
        return summary


# ─────────────────────────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train RefLens DANN model")
    p.add_argument("--source_dir",    default="dataset/synthetic_sequences")
    p.add_argument("--target_dir",    default="dataset/synthetic_sequences")
    p.add_argument("--checkpoint_dir",default="checkpoints")
    p.add_argument("--epochs",   type=int,   default=30)
    p.add_argument("--batch_size",type=int,  default=32)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--patience", type=int,   default=8)
    p.add_argument("--device",   default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--demo",     action="store_true",
                   help="Generate synthetic data then train (no real footage needed)")
    p.add_argument("--workers",  type=int, default=0,
                   help="DataLoader workers (0 = single-threaded, safe on all OS)")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    args = parse_args()

    if not TORCH_OK:
        print("[Train] PyTorch not installed. Install: pip install torch")
        return

    # ── Generate synthetic data if requested ──────────────────────
    if args.demo:
        log.info("[Train] Generating synthetic dataset...")
        gen = SyntheticDataGenerator(
            args.source_dir, n_per_class=200, n_domains=4
        )
        gen.generate_all()
        report_dataset_stats(args.source_dir)
        if args.target_dir == args.source_dir:
            args.target_dir = args.source_dir   # self-domain adapt in demo

    # ── DataModule setup ──────────────────────────────────────────
    dm = NexusDataModule(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        val_split=0.15,
    )
    dm.setup()

    # ── Model ─────────────────────────────────────────────────────
    model = RefLensModel()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"[Train] RefLensModel parameters: {n_params:,}")

    # Per-class weights from source dataset
    try:
        source_full = NexusDataset(args.source_dir, augment=False)
        cw = source_full.class_weights()
        log.info(f"[Train] Class weights: {[round(w,3) for w in cw.tolist()]}")
    except Exception:
        cw = None

    # ── Trainer ───────────────────────────────────────────────────
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA unavailable — falling back to CPU.")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS unavailable — falling back to CPU.")
        device = "cpu"

    trainer = RefLensTrainer(
        model          = model,
        source_loader  = dm.source_loader(),
        target_loader  = dm.target_loader(),
        val_loader     = dm.val_loader(),
        device         = device,
        lr             = args.lr,
        max_epochs     = args.epochs,
        patience       = args.patience,
        checkpoint_dir = args.checkpoint_dir,
        class_weights  = cw,
    )

    summary = trainer.fit()

    best_f1 = summary.get("final_metrics", {}).get("macro_f1", 0.0)
    log.info(f"[Train] Complete. Best val macro-F1: {best_f1:.4f}")
    log.info(f"[Train] Best checkpoint: {trainer.ckpt.best_path}")


if __name__ == "__main__":
    main()
