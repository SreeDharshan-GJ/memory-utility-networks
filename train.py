"""
Training Script — Memory Utility Network
=========================================
End-to-end training pipeline for the Memory Utility Network (MUN).

Features:
  - Config-driven via YAML (train.yaml / default.yaml)
  - Curriculum learning (easy → medium → hard episodes)
  - Mixed-precision training (torch.cuda.amp)
  - Gradient clipping
  - Cosine annealing with linear warmup
  - Early stopping
  - Checkpoint management (keep last K)
  - Weights & Biases + TensorBoard logging
  - Reproducible seeding
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

# ── Local imports ──────────────────────────────────────────────────────────────
# Resolve package root so script is runnable from project root
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import MemoryUtilityNetwork, build_model
from training import (
    CombinedUtilityLoss,
    MemoryDataLoaderFactory,
    ReplayBuffer,
    SyntheticEpisodeGenerator,
)
from training.dataloader import CurriculumStage
from evaluation import build_policy


# ── Config Loading ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Merge with base config if _base_ key present
    if "_base_" in cfg:
        base_path = Path(config_path).parent / cfg.pop("_base_")
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base = _deep_merge(base, cfg)
        return base
    return cfg


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ── Scheduler ─────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    schedule_type: str = "cosine",
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(warmup_steps, 1)
        if schedule_type == "cosine":
            progress = float(current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif schedule_type == "linear":
            progress = float(current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 1.0 - progress)
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


# ── Checkpoint Manager ─────────────────────────────────────────────────────────

class CheckpointManager:
    """Saves checkpoints and enforces keep-last-K policy."""

    def __init__(self, checkpoint_dir: str, keep_last_k: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self._saved: List[Path] = []

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: AdamW,
        scheduler: LambdaLR,
        scaler: Optional[GradScaler],
        metrics: Dict[str, float],
        is_best: bool = False,
        global_step: int = 0,
    ) -> Path:
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        }
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()

        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(state, ckpt_path)
        self._saved.append(ckpt_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy2(ckpt_path, best_path)

        # Enforce keep-last-K
        while len(self._saved) > self.keep_last_k:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()

        return ckpt_path

    def load_best(self, device: torch.device) -> Dict[str, Any]:
        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"No best model checkpoint at {best_path}")
        return torch.load(best_path, map_location=device)

    def load_checkpoint(self, path: str, device: torch.device) -> Dict[str, Any]:
        return torch.load(path, map_location=device)


# ── Early Stopping ─────────────────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Metrics ────────────────────────────────────────────────────────────────────

class RunningMetrics:
    """Accumulate scalar metrics over steps, then compute means."""

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v) * n
            self._counts[k] = self._counts.get(k, 0) + n

    def compute(self) -> Dict[str, float]:
        return {k: self._sums[k] / max(self._counts[k], 1) for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()


def compute_batch_metrics(
    predicted: torch.Tensor,   # (B, N) float
    binary_labels: torch.Tensor,  # (B, N) int
    mask: torch.Tensor,        # (B, N) bool — True=padding
) -> Dict[str, float]:
    """Compute accuracy and AUC-style ranking metrics for a batch."""
    valid = ~mask
    pred_flat = predicted[valid].detach().cpu().float()
    label_flat = binary_labels[valid].detach().cpu().float()

    # Binary accuracy at threshold 0.5
    pred_binary = (pred_flat >= 0.5).float()
    acc = (pred_binary == label_flat).float().mean().item()

    # Positive fraction
    pos_frac = label_flat.mean().item()

    return {
        "accuracy": acc,
        "pos_fraction": pos_frac,
        "num_valid": float(valid.sum().item()),
    }


# ── Train / Val Loops ──────────────────────────────────────────────────────────

def train_epoch(
    model: MemoryUtilityNetwork,
    loader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    loss_fn: CombinedUtilityLoss,
    scaler: Optional[GradScaler],
    device: torch.device,
    grad_clip: float,
    global_step: int,
    writer: Optional[SummaryWriter],
    wandb_run,
    log_every_n: int = 50,
) -> Tuple[Dict[str, float], int]:
    model.train()
    metrics_acc = RunningMetrics()
    prev_scores: Optional[torch.Tensor] = None

    for batch in loader:
        memory_embeddings = batch["memory_embeddings"].to(device)       # (B, N, D)
        context_embeddings = batch["context_embeddings"].to(device)     # (B, L, D)
        memory_ages = batch["memory_ages"].to(device)                   # (B, N)
        binary_labels = batch["binary_labels"].to(device)               # (B, N)
        soft_labels = batch["soft_labels"].to(device)                   # (B, N)
        mem_mask = batch["memory_padding_mask"].to(device)              # (B, N)
        ctx_mask = batch["context_padding_mask"].to(device)             # (B, L)

        with autocast(device_type='cuda', enabled=(scaler is not None)):
            outputs = model(
                memory_embeddings=memory_embeddings,
                context_embeddings=context_embeddings,
                memory_ages=memory_ages,
                memory_padding_mask=mem_mask,
                context_padding_mask=ctx_mask,
            )
            predicted = outputs["utility_scores"]       # (B, N)
            contrastive_proj = outputs["contrastive_proj"]  # (B, N, proj_dim)

            loss_dict = loss_fn(
                predicted_scores=predicted,
                binary_labels=binary_labels,
                soft_labels=soft_labels,
                contrastive_proj=contrastive_proj,
                prev_scores=prev_scores,
                mask=~mem_mask,
            )
            loss = loss_dict["total"]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        global_step += 1

        # Store scores for temporal consistency loss next step
        prev_scores = predicted.detach()

        batch_metrics = compute_batch_metrics(predicted, binary_labels, mem_mask)
        step_metrics = {
            "loss/total": loss_dict["total"].item(),
            "loss/bce": loss_dict["bce"].item(),
            "loss/contrastive": loss_dict["contrastive"].item(),
            "loss/ranking": loss_dict["ranking"].item(),
            "loss/temporal": loss_dict["temporal"].item(),
            "lr": scheduler.get_last_lr()[0],
            **batch_metrics,
        }
        metrics_acc.update(step_metrics, n=memory_embeddings.size(0))

        if global_step % log_every_n == 0:
            if writer is not None:
                for k, v in step_metrics.items():
                    writer.add_scalar(f"train/{k}", v, global_step)
            if wandb_run is not None:
                wandb_run.log({"train/" + k: v for k, v in step_metrics.items()}, step=global_step)

    return metrics_acc.compute(), global_step


@torch.no_grad()
def validate(
    model: MemoryUtilityNetwork,
    loader,
    loss_fn: CombinedUtilityLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metrics_acc = RunningMetrics()

    for batch in loader:
        memory_embeddings = batch["memory_embeddings"].to(device)
        context_embeddings = batch["context_embeddings"].to(device)
        memory_ages = batch["memory_ages"].to(device)
        binary_labels = batch["binary_labels"].to(device)
        soft_labels = batch["soft_labels"].to(device)
        mem_mask = batch["memory_padding_mask"].to(device)
        ctx_mask = batch["context_padding_mask"].to(device)

        outputs = model(
            memory_embeddings=memory_embeddings,
            context_embeddings=context_embeddings,
            memory_ages=memory_ages,
            memory_padding_mask=mem_mask,
            context_padding_mask=ctx_mask,
        )
        predicted = outputs["utility_scores"]
        contrastive_proj = outputs["contrastive_proj"]

        loss_dict = loss_fn(
            predicted_scores=predicted,
            binary_labels=binary_labels,
            soft_labels=soft_labels,
            contrastive_proj=contrastive_proj,
            prev_scores=None,
            mask=~mem_mask,
        )
        batch_metrics = compute_batch_metrics(predicted, binary_labels, mem_mask)
        step_metrics = {
            "loss/total": loss_dict["total"].item(),
            "loss/bce": loss_dict["bce"].item(),
            "loss/contrastive": loss_dict["contrastive"].item(),
            "loss/ranking": loss_dict["ranking"].item(),
            **batch_metrics,
        }
        metrics_acc.update(step_metrics, n=memory_embeddings.size(0))

    return metrics_acc.compute()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    cfg = load_config(args.config)
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    project_cfg = cfg.get("project", {})

    # ── Reproducibility ──
    seed = cfg.get("project", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and training_cfg.get("mixed_precision", True)
    scaler = GradScaler(device='cuda') if use_amp else None

    # ── Logging ──
    log_dir = Path(training_cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir)) if training_cfg.get("use_tensorboard", True) else None

    wandb_run = None
    if training_cfg.get("use_wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=training_cfg.get("wandb_project", "memory-utility-networks"),
                entity=training_cfg.get("wandb_entity", None),
                config=cfg,
                name=f"mun_{int(time.time())}",
            )
        except ImportError:
            print("wandb not available; skipping W&B logging.")

    # ── Model ──
    model = build_model(cfg).to(device)
    print(model)

    # ── Loss ──
    loss_fn = CombinedUtilityLoss(
        bce_weight=training_cfg.get("utility_loss_weight", 1.0),
        contrastive_weight=training_cfg.get("contrastive_loss_weight", 0.5),
        ranking_weight=training_cfg.get("ranking_loss_weight", 0.3),
        temporal_weight=training_cfg.get("temporal_loss_weight", 0.1),
    ).to(device)

    # ── Optimizer ──
    lr = training_cfg.get("learning_rate", 3e-4)
    weight_decay = training_cfg.get("weight_decay", 1e-4)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)

    # ── Data ──
    embed_dim = model_cfg.get("embedding_dim", 384)
    batch_size = training_cfg.get("batch_size", 32)
    eval_batch_size = training_cfg.get("eval_batch_size", 64)
    num_workers = min(4, os.cpu_count() or 1)

    loader_factory = MemoryDataLoaderFactory(
        embed_dim=embed_dim,
        max_memories=dataset_cfg.get("memory_capacity", 128),
        context_window=8,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    curriculum_cfg = training_cfg.get("curriculum", {})
    use_curriculum = curriculum_cfg.get("enabled", False) and not args.no_curriculum

    if use_curriculum:
        stage_cfgs = curriculum_cfg.get("stages", [
            {"name": "easy", "epochs": 20, "episode_length": 10, "capacity": 100},
            {"name": "medium", "epochs": 40, "episode_length": 25, "capacity": 500},
            {"name": "hard", "epochs": 40, "episode_length": 50, "capacity": 1000},
        ])
        stages = [CurriculumStage.from_config(s, num_episodes=1000) for s in stage_cfgs]
        stage_loaders, val_loader = loader_factory.curriculum_loaders(
            stages=stages,
            val_episodes=200,
            generator_base_kwargs={"embed_dim": embed_dim, "num_topics": 64},
        )
    else:
        loaders = loader_factory.from_generator(
            num_train_episodes=dataset_cfg.get("num_train_samples", 2000) // 20,
            num_val_episodes=dataset_cfg.get("num_val_samples", 200) // 10,
            generator_base_kwargs={"embed_dim": embed_dim, "num_topics": 64},
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]

    # ── Scheduler ──
    num_epochs = training_cfg.get("num_epochs", 50)
    if use_curriculum:
        total_epochs = sum(s.epochs for s in stages)
    else:
        total_epochs = num_epochs

    steps_per_epoch = len(train_loader) if not use_curriculum else len(stage_loaders[0])
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = training_cfg.get("warmup_steps", min(500, total_steps // 10))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps, training_cfg.get("scheduler", "cosine"))

    # ── Checkpoint / Early stopping ──
    ckpt_mgr = CheckpointManager(
        training_cfg.get("checkpoint_dir", "checkpoints"),
        training_cfg.get("keep_last_k", 3),
    )
    early_stopper = EarlyStopper(
        patience=training_cfg.get("patience", 10),
        min_delta=training_cfg.get("min_delta", 1e-4),
        mode="min",
    ) if training_cfg.get("early_stopping", True) else None

    save_every = training_cfg.get("save_every_n_epochs", 5)
    eval_every = training_cfg.get("eval_every_n_epochs", 2)
    log_every = training_cfg.get("log_every_n_steps", 50)
    best_val_loss = float("inf")
    global_step = 0
    epoch_offset = 0

    # ── Resume from checkpoint ──
    if args.resume:
        ckpt = ckpt_mgr.load_checkpoint(args.resume, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        epoch_offset = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ── Training loop ──
    if use_curriculum:
        # Curriculum: iterate over stages
        for stage_idx, (stage, stage_loader) in enumerate(zip(stages, stage_loaders)):
            print(f"\n=== Curriculum Stage: {stage.name} ({stage.epochs} epochs) ===")
            for local_epoch in range(stage.epochs):
                epoch = epoch_offset + local_epoch
                train_metrics, global_step = train_epoch(
                    model, stage_loader, optimizer, scheduler, loss_fn,
                    scaler, device, training_cfg.get("gradient_clip", 1.0),
                    global_step, writer, wandb_run, log_every,
                )
                print(
                    f"[Stage {stage.name}] Epoch {epoch} | "
                    f"loss={train_metrics['loss/total']:.4f} | "
                    f"acc={train_metrics['accuracy']:.4f}"
                )

                if (local_epoch + 1) % eval_every == 0:
                    val_metrics = validate(model, val_loader, loss_fn, device)
                    val_loss = val_metrics["loss/total"]
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss

                    print(
                        f"  [Val] loss={val_loss:.4f} | acc={val_metrics['accuracy']:.4f}"
                        + (" <- best" if is_best else "")
                    )

                    if writer:
                        for k, v in val_metrics.items():
                            writer.add_scalar(f"val/{k}", v, global_step)
                    if wandb_run:
                        wandb_run.log({"val/" + k: v for k, v in val_metrics.items()}, step=global_step)

                    if (local_epoch + 1) % save_every == 0 or is_best:
                        ckpt_mgr.save(epoch, model, optimizer, scheduler, scaler, val_metrics, is_best, global_step)

                    if early_stopper and early_stopper(val_loss):
                        print("Early stopping triggered.")
                        break

            epoch_offset += stage.epochs
    else:
        for epoch in range(epoch_offset, epoch_offset + num_epochs):
            train_metrics, global_step = train_epoch(
                model, train_loader, optimizer, scheduler, loss_fn,
                scaler, device, training_cfg.get("gradient_clip", 1.0),
                global_step, writer, wandb_run, log_every,
            )
            print(
                f"Epoch {epoch}/{epoch_offset + num_epochs - 1} | "
                f"loss={train_metrics['loss/total']:.4f} | "
                f"acc={train_metrics['accuracy']:.4f} | "
                f"lr={train_metrics['lr']:.2e}"
            )

            if (epoch + 1) % eval_every == 0:
                val_metrics = validate(model, val_loader, loss_fn, device)
                val_loss = val_metrics["loss/total"]
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                print(
                    f"  [Val] loss={val_loss:.4f} | acc={val_metrics['accuracy']:.4f}"
                    + (" ← best" if is_best else "")
                )

                if writer:
                    for k, v in val_metrics.items():
                        writer.add_scalar(f"val/{k}", v, global_step)
                if wandb_run:
                    wandb_run.log({"val/" + k: v for k, v in val_metrics.items()}, step=global_step)

                if (epoch + 1) % save_every == 0 or is_best:
                    ckpt_mgr.save(epoch, model, optimizer, scheduler, scaler, val_metrics, is_best, global_step)

                if early_stopper and early_stopper(val_loss):
                    print("Early stopping triggered.")
                    break

    if writer:
        writer.close()
    if wandb_run:
        wandb_run.finish()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {ckpt_mgr.checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Memory Utility Network")
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--no-curriculum", action="store_true",
        help="Disable curriculum learning even if enabled in config"
    )
    args = parser.parse_args()
    main(args)
