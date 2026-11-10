"""cGAN Training Script for BB84 Eavesdropping Attack-Policy Generation.

Trains a conditional Generative Adversarial Network (cGAN) on validated BB84 attack
policies conditioned on observable channel parameters c = [Q_budget, gamma_noise_hat].

Hardware: NVIDIA RTX 3050 Laptop GPU (CUDA:0) / PyTorch 2.5.1
Dataset: results/full_training_dataset.json (Condition-partitioned)

Training specification:
- Epochs: 200
- Batch size: 64
- Generator: Multi-head (Gumbel-Softmax family, Sigmoid PIR, Sigmoid Probe)
- Discriminator: Spectral-normalized MLP (7 -> 128 -> 64 -> 1)
- Temperature annealing: tau_0 = 1.0 -> tau_min = 0.20
- Importance sampling: exp(I_AE / tau_I) on training partition
- Validation: Evaluated every 10 epochs on held-out validation conditions (C17-C20)
- Checkpoints: checkpoints/cgan_best.pt, checkpoints/cgan_latest.pt
- Training History: results/cgan_training_history.json
"""

import datetime
import json
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import psutil

from src.cgan.dataset import create_dataloaders
from src.cgan.models import ConditionalDiscriminator, ConditionalGenerator
from src.cgan.trainer import CGANTrainer


def train_cgan_model(
    dataset_path: str = "results/full_training_dataset.json",
    checkpoint_dir: str = "checkpoints",
    history_path: str = "results/cgan_training_history.json",
    epochs: int = 200,
    batch_size: int = 64,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    latent_dim: int = 16,
    seed: int = 42,
    tau_start: float = 1.0,
    tau_min: float = 0.20,
    tau_I: float = 0.10,
) -> dict:
    """Execute complete cGAN training run with full instrumentation."""
    print("=" * 80)
    print("M3 STAGE C: CONDITIONAL GAN TRAINING (BB84 ATTACK-POLICY GENERATOR)")
    print("=" * 80)

    # Hardware detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"PyTorch Device: {device} ({cuda_name})")
    print(f"Random Seed: {seed} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"Gumbel-Softmax Temperature: {tau_start:.2f} -> {tau_min:.2f} (annealed)")
    print(f"Mutual Information Temperature tau_I: {tau_I:.2f}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    # Build dataloaders with condition-level partitioning
    print(f"\nLoading dataset from: {dataset_path}...")
    loaders = create_dataloaders(
        dataset_source=dataset_path,
        batch_size=batch_size,
        tau_I=tau_I,
        use_weighted_sampling=True,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    print(f"Dataset Partitions (Feasible Records Only):")
    print(f"  Train: {len(loaders['train_dataset'])} records (16 conditions C01-C16)")
    print(f"  Val:   {len(loaders['val_dataset'])} records (4 conditions C17-C20)")
    print(f"  Test:  {len(loaders['test_dataset'])} records (4 conditions C21-C24)")

    # Instantiate generator and discriminator
    generator = ConditionalGenerator(
        latent_dim=latent_dim,
        condition_dim=2,
        hidden_dims=[128, 128],
    )
    discriminator = ConditionalDiscriminator(
        policy_dim=5,
        condition_dim=2,
        hidden_dims=[128, 64],
        use_spectral_norm=True,
    )

    total_params_g = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    total_params_d = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"\nModel Architectures:")
    print(f"  Generator:     {total_params_g:,} trainable parameters")
    print(f"  Discriminator: {total_params_d:,} trainable parameters")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    # Initialize trainer
    trainer = CGANTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_g=lr_g,
        lr_d=lr_d,
        latent_dim=latent_dim,
        device=str(device),
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        tau_start=tau_start,
        tau_min=tau_min,
    )

    t_train_start = time.perf_counter()
    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024.0 * 1024.0)

    print("\nCommencing GAN training...")
    history = trainer.train(
        epochs=epochs,
        eval_interval=10,
        save_best=True,
    )
    t_train_total = time.perf_counter() - t_train_start
    ram_end_mb = process.memory_info().rss / (1024.0 * 1024.0)
    vram_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        if torch.cuda.is_available()
        else 0.0
    )

    best_ckpt = os.path.join(checkpoint_dir, "cgan_best.pt")
    latest_ckpt = os.path.join(checkpoint_dir, "cgan_latest.pt")

    # Serialize training summary
    summary_report = {
        "metadata": {
            "training_run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "learning_rate_g": lr_g,
            "learning_rate_d": lr_d,
            "latent_dim": latent_dim,
            "tau_start": tau_start,
            "tau_min": tau_min,
            "tau_I": tau_I,
            "best_epoch": trainer.best_epoch,
            "best_val_loss": round(float(trainer.best_val_loss), 6),
            "checkpoints": {
                "best": best_ckpt,
                "latest": latest_ckpt,
            },
        },
        "performance": {
            "total_training_time_seconds": round(t_train_total, 2),
            "average_time_per_epoch_seconds": round(t_train_total / epochs, 4),
            "device": str(device),
            "cuda_name": cuda_name,
            "initial_ram_mb": round(ram_start_mb, 2),
            "final_ram_mb": round(ram_end_mb, 2),
            "peak_vram_mb": round(vram_peak_mb, 2),
        },
        "final_metrics": {
            "loss_d": round(history["loss_d"][-1], 6),
            "loss_g": round(history["loss_g"][-1], 6),
            "d_acc_real": round(history["d_acc_real"][-1], 4),
            "d_acc_fake": round(history["d_acc_fake"][-1], 4),
            "final_temperature": round(history["temperature"][-1], 4),
        },
        "history": history,
    }

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING RUN COMPLETE")
    print(f"Total Time: {t_train_total:.2f}s ({t_train_total/60.0:.2f} min)")
    print(f"Best Validation Epoch: {trainer.best_epoch} (Val Loss: {trainer.best_val_loss:.6f})")
    print(f"Final Losses: L_D={history['loss_d'][-1]:.4f}, L_G={history['loss_g'][-1]:.4f}")
    print(f"Final Accuracies: D(real)={history['d_acc_real'][-1]*100:.1f}%, D(fake)={history['d_acc_fake'][-1]*100:.1f}%")
    print(f"Checkpoints Saved: {best_ckpt}, {latest_ckpt}")
    print(f"Training History Saved: {history_path}")
    print("=" * 80)

    return summary_report


if __name__ == "__main__":
    train_cgan_model()

