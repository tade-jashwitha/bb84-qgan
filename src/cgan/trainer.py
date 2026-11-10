"""End-to-end training and checkpointing engine for classical Conditional GAN (cGAN).

Implements:
1. Non-saturating minimax GAN training with BCEWithLogitsLoss.
2. Gumbel-Softmax temperature annealing for differentiable categorical family selection.
3. Alternating Adam optimization with spectral-normalized discriminator updates.
4. Validation monitoring and best-model checkpoint serialization.
5. Reproducible seed configuration and execution logging.
"""

import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.cgan.models import ConditionalDiscriminator, ConditionalGenerator


class CGANTrainer:
    """Trainer coordinating conditional generator and discriminator minimax optimization."""

    def __init__(
        self,
        generator: ConditionalGenerator,
        discriminator: ConditionalDiscriminator,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        lr_g: float = 1e-4,
        lr_d: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        tau_start: float = 1.0,
        tau_min: float = 0.20,
        tau_decay: float = 0.98,
        checkpoint_dir: str = "checkpoints",
        seed: int = 42,
        latent_dim: Optional[int] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.lr_g = lr_g
        self.lr_d = lr_d
        self.tau_start = tau_start
        self.tau_min = tau_min
        self.tau_decay = tau_decay
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed

        # Set reproducible seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr_g,
            betas=(beta1, beta2),
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=(beta1, beta2),
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Training state history
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "loss_d": [],
            "loss_d_real": [],
            "loss_d_fake": [],
            "loss_g": [],
            "d_acc_real": [],
            "d_acc_fake": [],
            "tau": [],
            "val_loss_d": [],
            "val_acc": [],
            "epoch_duration_sec": [],
            "vram_allocated_mb": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def compute_tau(self, epoch: int) -> float:
        """Compute annealed Gumbel-Softmax temperature for current epoch."""
        return max(self.tau_min, self.tau_start * (self.tau_decay ** epoch))

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one complete training epoch over the training DataLoader."""
        self.generator.train()
        self.discriminator.train()

        tau = self.compute_tau(epoch)

        total_loss_d = 0.0
        total_loss_d_real = 0.0
        total_loss_d_fake = 0.0
        total_loss_g = 0.0
        total_acc_real = 0.0
        total_acc_fake = 0.0
        num_batches = 0

        for batch in self.train_loader:
            c_real = batch["condition"].to(self.device)
            a_real = batch["policy"].to(self.device)
            b_size = c_real.size(0)

            real_labels = torch.ones((b_size, 1), device=self.device, dtype=torch.float32)
            fake_labels = torch.zeros((b_size, 1), device=self.device, dtype=torch.float32)

            # -------------------------------------------------------------
            # 1. Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # -------------------------------------------------------------
            self.optimizer_D.zero_grad()

            # Real loss
            out_real = self.discriminator(a_real, c_real)
            loss_d_real = self.criterion(out_real, real_labels)

            # Fake loss
            z = torch.randn(b_size, self.generator.latent_dim, device=self.device, dtype=torch.float32)
            g_out = self.generator(z, c_real, tau=tau, use_gumbel=True)
            a_fake = g_out["masked_policy"]

            out_fake = self.discriminator(a_fake.detach(), c_real)
            loss_d_fake = self.criterion(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            self.optimizer_D.step()

            # Discriminator accuracies
            acc_real = (torch.sigmoid(out_real) >= 0.5).float().mean().item()
            acc_fake = (torch.sigmoid(out_fake) < 0.5).float().mean().item()

            # -------------------------------------------------------------
            # 2. Train Generator: maximize log(D(G(z))) (non-saturating)
            # -------------------------------------------------------------
            self.optimizer_G.zero_grad()

            # Sample fresh noise for generator update
            z2 = torch.randn(b_size, self.generator.latent_dim, device=self.device, dtype=torch.float32)
            g_out2 = self.generator(z2, c_real, tau=tau, use_gumbel=True)
            a_fake2 = g_out2["masked_policy"]

            out_fake_for_g = self.discriminator(a_fake2, c_real)
            loss_g = self.criterion(out_fake_for_g, real_labels)

            loss_g.backward()
            self.optimizer_G.step()

            total_loss_d += loss_d.item()
            total_loss_d_real += loss_d_real.item()
            total_loss_d_fake += loss_d_fake.item()
            total_loss_g += loss_g.item()
            total_acc_real += acc_real
            total_acc_fake += acc_fake
            num_batches += 1

        n_b = max(1, num_batches)
        metrics = {
            "epoch": epoch,
            "loss_d": total_loss_d / n_b,
            "loss_d_real": total_loss_d_real / n_b,
            "loss_d_fake": total_loss_d_fake / n_b,
            "loss_g": total_loss_g / n_b,
            "d_acc_real": total_acc_real / n_b,
            "d_acc_fake": total_acc_fake / n_b,
            "tau": tau,
        }
        return metrics

    def validate(self) -> Dict[str, float]:
        """Evaluate discriminator loss and accuracy on the validation partition."""
        if self.val_loader is None or len(self.val_loader) == 0:
            return {"val_loss_d": 0.0, "val_acc": 0.0}

        self.generator.eval()
        self.discriminator.eval()

        total_val_loss = 0.0
        total_val_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                c_real = batch["condition"].to(self.device)
                a_real = batch["policy"].to(self.device)
                b_size = c_real.size(0)

                real_labels = torch.ones((b_size, 1), device=self.device, dtype=torch.float32)
                fake_labels = torch.zeros((b_size, 1), device=self.device, dtype=torch.float32)

                out_real = self.discriminator(a_real, c_real)
                loss_real = self.criterion(out_real, real_labels)

                z = torch.randn(b_size, self.generator.latent_dim, device=self.device, dtype=torch.float32)
                g_out = self.generator(z, c_real, use_gumbel=False)
                a_fake = g_out["masked_policy"]

                out_fake = self.discriminator(a_fake, c_real)
                loss_fake = self.criterion(out_fake, fake_labels)

                val_loss = (loss_real + loss_fake).item()
                acc_r = (torch.sigmoid(out_real) >= 0.5).float().mean().item()
                acc_f = (torch.sigmoid(out_fake) < 0.5).float().mean().item()
                val_acc = 0.5 * (acc_r + acc_f)

                total_val_loss += val_loss
                total_val_acc += val_acc
                num_batches += 1

        n_b = max(1, num_batches)
        return {
            "val_loss_d": total_val_loss / n_b,
            "val_acc": total_val_acc / n_b,
        }

    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """Serialize generator, discriminator, and optimizer weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "config": {
                "latent_dim": self.generator.latent_dim,
                "condition_dim": self.generator.condition_dim,
                "policy_dim": self.discriminator.policy_dim,
                "lr_g": self.lr_g,
                "lr_d": self.lr_d,
                "tau_start": self.tau_start,
                "tau_min": self.tau_min,
                "tau_decay": self.tau_decay,
                "seed": self.seed,
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Restore model and optimizer state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", -1)

    def fit(
        self,
        epochs: int = 200,
        eval_every: int = 5,
        verbose: bool = True,
        save_best: bool = True,
    ) -> Dict[str, Any]:
        """Execute full training loop across specified number of epochs."""
        t_fit_start = time.perf_counter()

        best_checkpoint_path = os.path.join(self.checkpoint_dir, "cgan_best.pt")
        latest_checkpoint_path = os.path.join(self.checkpoint_dir, "cgan_latest.pt")

        if verbose:
            print(f"Commencing cGAN training on device: {self.device}")
            print(f"Epochs: {epochs} | Batch Size: {self.train_loader.batch_size} | Latent Dim: {self.generator.latent_dim}")
            print("-" * 80)

        for epoch in range(1, epochs + 1):
            t_ep_start = time.perf_counter()
            train_metrics = self.train_epoch(epoch)

            val_metrics = {"val_loss_d": 0.0, "val_acc": 0.0}
            if self.val_loader is not None and (epoch % eval_every == 0 or epoch == epochs):
                val_metrics = self.validate()

            t_ep = time.perf_counter() - t_ep_start

            # Record history
            self.history["epoch"].append(epoch)
            self.history["loss_d"].append(train_metrics["loss_d"])
            self.history["loss_d_real"].append(train_metrics["loss_d_real"])
            self.history["loss_d_fake"].append(train_metrics["loss_d_fake"])
            self.history["loss_g"].append(train_metrics["loss_g"])
            self.history["d_acc_real"].append(train_metrics["d_acc_real"])
            self.history["d_acc_fake"].append(train_metrics["d_acc_fake"])
            self.history["tau"].append(train_metrics["tau"])
            self.history["val_loss_d"].append(val_metrics["val_loss_d"])
            self.history["val_acc"].append(val_metrics["val_acc"])
            vram_mb = (
                torch.cuda.memory_allocated() / (1024.0 * 1024.0)
                if torch.cuda.is_available()
                else 0.0
            )
            self.history["epoch_duration_sec"].append(round(t_ep, 4))
            self.history["vram_allocated_mb"].append(round(vram_mb, 2))

            # Checkpoint best model based on validation stability
            val_loss = val_metrics["val_loss_d"]
            if save_best and val_loss > 0.0 and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(best_checkpoint_path, epoch=epoch, is_best=True)

            if verbose and (epoch % eval_every == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"Epoch [{epoch:3d}/{epochs:3d}] ({t_ep:.2f}s) | "
                    f"Loss D: {train_metrics['loss_d']:.4f} | Loss G: {train_metrics['loss_g']:.4f} | "
                    f"D Acc Real: {train_metrics['d_acc_real']*100:.1f}% | Fake: {train_metrics['d_acc_fake']*100:.1f}% | "
                    f"Val Loss: {val_metrics['val_loss_d']:.4f} | Tau: {train_metrics['tau']:.3f}"
                )

        t_total = time.perf_counter() - t_fit_start
        self.save_checkpoint(latest_checkpoint_path, epoch=epochs, is_best=False)

        if verbose:
            print("-" * 80)
            print(f"cGAN Training Complete in {t_total:.2f}s ({t_total/60.0:.2f} min).")
            print(f"Best Validation Epoch: {self.best_epoch} (Loss: {self.best_val_loss:.4f})")
            print(f"Checkpoints saved to: {self.checkpoint_dir}")

        return {
            "total_time_seconds": t_total,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "final_epoch": epochs,
            "history": self.history,
        }

    train = fit

