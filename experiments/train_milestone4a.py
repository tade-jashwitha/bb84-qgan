"""Milestone 4A — Full Classical Conditional GAN Training (200 Epochs).

Project: Adaptive QGAN-Based Eavesdropping in BB84 QKD
Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.1 / PyTorch 2.5.1)
Dataset: results/full_training_dataset.json (1,152 records)

Key Scientific Architecture:
1. Exact 200 epochs of training on Training Partition (C01-C16, 368 feasible exemplars).
2. Lexicographic & Pareto Checkpoint Selection Procedure:
   - Priority 1 (Hard Gate): 100% parameter validity (R_val = 1.0: theta_E in [0, pi/4], alpha, beta in [0, 1]).
   - Priority 2 (Hard Gate): Zero protocol aborts (abort_count == 0).
   - Priority 3 (Screening Gate): Minimum feasibility yield (Y_feas >= 0.80 under Wilson 95% UCB).
   - Priority 4 (Primary Physical Objective): Maximize Eve mutual information I(A; E) among feasible policies.
   - Priority 5 (Anti-Collapse Guard): Non-zero continuous parameter dispersion (std_theta >= 0.003 rad when probe >= 4).
   - Multi-candidate preservation:
     * cgan_m4a_lexicographic_best.pt (Qualified primary model maximizing feasible I_AE)
     * cgan_m4a_max_feasibility.pt (Pareto anchor maximizing feasibility yield)
     * cgan_m4a_max_information.pt (Pareto anchor maximizing mutual information)
     * cgan_m4a_balanced_pareto.pt (Scalar compromise on frontier maximizing Y_feas * I_AE)
     * cgan_m4a_latest.pt (Epoch 200 state)
3. Controlled Counterfactual Evaluation:
   - Holds gamma_hat_noise strictly fixed at C17 (gamma_hat = 0.008) and C19 (gamma_hat = 0.012).
   - Varies Q_budget in {0.050, 0.110} across fixed paired latents Z_cf to isolate causal sensitivity.
4. Validation Reproducibility via Common Random Numbers (CRN):
   - Fixed validation latent matrix Z_val (16 x 16) generated deterministically with seed 42.
   - Fixed CRN simulation seeds per (condition, candidate) across all checkpoints.
   - Preserves Alice bits, Alice bases, Bob bases, and sifted indices 100% byte-for-byte identical.
5. Training Initialization:
   - Strategy A: Fresh model and optimizer initialization from scratch with seed 42.
   - Historical pilot checkpoints (cgan_pilot_best.pt, cgan_pilot_latest.pt) strictly preserved.
6. Quarantine: Test partition (C21-C24) strictly isolated and unobserved.
"""

import datetime
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.cgan.dataset import create_dataloaders
from src.cgan.models import ConditionalDiscriminator, ConditionalGenerator
from src.cgan.policy_engine import cGANPolicyEngine, OperationalDecision
from src.dataset.generator import (
    execute_single_simulation,
    get_full_environmental_conditions,
)
from src.dataset.schema import (
    AttackConfiguration,
    AttackFamily,
    AttackParameters,
    StatisticalFeasibility,
)


def compute_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    if not os.path.exists(path):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class Milestone4ATrainer:
    """Specialized trainer for Milestone 4A Classical cGAN 200-Epoch Experiment."""

    def __init__(
        self,
        generator: ConditionalGenerator,
        discriminator: ConditionalDiscriminator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        tau_0: float = 1.0,
        tau_min: float = 0.20,
        tau_decay: float = 0.985,
        checkpoint_dir: str = "checkpoints/milestone4a",
        results_dir: str = "results/milestone4a",
        seed: int = 42,
        eval_physical_interval: int = 10,
        num_candidates_eval: int = 16,
        n_pulses_eval: int = 2000,
        feasibility_gate_threshold: float = 0.80,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.lr_g = lr_g
        self.lr_d = lr_d
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.tau_decay = tau_decay
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.seed = seed
        self.eval_physical_interval = eval_physical_interval
        self.num_candidates_eval = num_candidates_eval
        self.n_pulses_eval = n_pulses_eval
        self.feasibility_gate_threshold = feasibility_gate_threshold

        # Set seeds for deterministic initialization
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Fixed validation latents for exact reproducibility across epochs
        latent_gen = torch.Generator().manual_seed(seed + 999)
        self.fixed_val_latents = torch.randn(
            num_candidates_eval,
            self.generator.latent_dim,
            generator=latent_gen,
            device=self.device,
            dtype=torch.float32,
        )

        # Fixed counterfactual latents for controlled sensitivity evaluation
        cf_gen = torch.Generator().manual_seed(seed + 1234)
        self.fixed_cf_latents = torch.randn(
            num_candidates_eval,
            self.generator.latent_dim,
            generator=cf_gen,
            device=self.device,
            dtype=torch.float32,
        )

        # Optimizers (Strategy A: Fresh from scratch)
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
        self.criterion = nn.BCEWithLogitsLoss()

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Pareto and Lexicographic Checkpoint Tracking
        self.lexicographic_best: Optional[Dict[str, Any]] = None
        self.max_feasibility_best: Optional[Dict[str, Any]] = None
        self.max_information_best: Optional[Dict[str, Any]] = None
        self.balanced_pareto_best: Optional[Dict[str, Any]] = None
        self.evaluated_checkpoints: List[Dict[str, Any]] = []
        self.pareto_frontier: List[Dict[str, Any]] = []

        # History telemetry
        self.history: Dict[str, List[Any]] = {
            "epoch": [],
            "loss_d": [],
            "loss_d_real": [],
            "loss_d_fake": [],
            "loss_g": [],
            "d_acc_real": [],
            "d_acc_fake": [],
            "grad_norm_d": [],
            "grad_norm_g": [],
            "lr_g": [],
            "lr_d": [],
            "tau": [],
            "val_loss_d": [],
            "val_loss_g": [],
            "val_acc": [],
            "feasibility_yield": [],
            "abort_rate": [],
            "validity_rate": [],
            "mean_info_gain_feas": [],
            "median_info_gain_feas": [],
            "p75_info_gain_feas": [],
            "p90_info_gain_feas": [],
            "max_info_gain_feas": [],
            "continuous_dispersion": [],
            "std_theta_rad": [],
            "std_alpha": [],
            "family_entropy": [],
            "family_probs": [],
            "equilibrium_proximity": [],
            "counterfactual_sensitivity": [],
            "vram_allocated_mb": [],
            "vram_reserved_mb": [],
            "epoch_duration_sec": [],
        }

        # Detailed physical evaluations log
        self.physical_evaluations: List[Dict[str, Any]] = []

    def compute_tau(self, epoch: int) -> float:
        """Compute annealed Gumbel temperature: tau_t = max(0.20, tau_0 * (tau_decay^t))."""
        return max(self.tau_min, self.tau_0 * (self.tau_decay ** epoch))

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one complete training epoch over training partition."""
        self.generator.train()
        self.discriminator.train()
        tau = self.compute_tau(epoch)

        total_loss_d = 0.0
        total_loss_d_real = 0.0
        total_loss_d_fake = 0.0
        total_loss_g = 0.0
        total_acc_real = 0.0
        total_acc_fake = 0.0
        total_grad_d = 0.0
        total_grad_g = 0.0
        num_batches = 0

        for batch in self.train_loader:
            c_real = batch["condition"].to(self.device)
            a_real = batch["policy"].to(self.device)
            b_size = c_real.size(0)

            # Device verification
            if c_real.device.type != "cuda" and torch.cuda.is_available():
                raise RuntimeError("Unexpected CPU fallback for input tensor!")

            real_labels = torch.ones((b_size, 1), device=self.device, dtype=torch.float32)
            fake_labels = torch.zeros((b_size, 1), device=self.device, dtype=torch.float32)

            # ---------------------------------------------------------
            # 1. Discriminator Step: max log(D(real)) + log(1 - D(G(z)))
            # ---------------------------------------------------------
            self.optimizer_D.zero_grad()

            out_real = self.discriminator(a_real, c_real)
            loss_d_real = self.criterion(out_real, real_labels)

            z = torch.randn(b_size, self.generator.latent_dim, device=self.device, dtype=torch.float32)
            g_out = self.generator(z, c_real, tau=tau, use_gumbel=True)
            a_fake = g_out["masked_policy"]

            out_fake = self.discriminator(a_fake.detach(), c_real)
            loss_d_fake = self.criterion(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            if torch.isnan(loss_d) or torch.isinf(loss_d):
                raise FloatingPointError(f"NaN/Inf encountered in Discriminator loss at epoch {epoch}")

            loss_d.backward()
            grad_norm_d = nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10.0).item()
            self.optimizer_D.step()

            acc_real = (torch.sigmoid(out_real) >= 0.5).float().mean().item()
            acc_fake = (torch.sigmoid(out_fake) < 0.5).float().mean().item()

            # ---------------------------------------------------------
            # 2. Generator Step: max log(D(G(z)))
            # ---------------------------------------------------------
            self.optimizer_G.zero_grad()

            z2 = torch.randn(b_size, self.generator.latent_dim, device=self.device, dtype=torch.float32)
            g_out2 = self.generator(z2, c_real, tau=tau, use_gumbel=True)
            a_fake2 = g_out2["masked_policy"]

            out_fake_g = self.discriminator(a_fake2, c_real)
            loss_g = self.criterion(out_fake_g, real_labels)
            if torch.isnan(loss_g) or torch.isinf(loss_g):
                raise FloatingPointError(f"NaN/Inf encountered in Generator loss at epoch {epoch}")

            loss_g.backward()
            grad_norm_g = nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=10.0).item()
            self.optimizer_G.step()

            total_loss_d += loss_d.item()
            total_loss_d_real += loss_d_real.item()
            total_loss_d_fake += loss_d_fake.item()
            total_loss_g += loss_g.item()
            total_acc_real += acc_real
            total_acc_fake += acc_fake
            total_grad_d += grad_norm_d
            total_grad_g += grad_norm_g
            num_batches += 1

        nb = max(1, num_batches)
        return {
            "loss_d": total_loss_d / nb,
            "loss_d_real": total_loss_d_real / nb,
            "loss_d_fake": total_loss_d_fake / nb,
            "loss_g": total_loss_g / nb,
            "d_acc_real": total_acc_real / nb,
            "d_acc_fake": total_acc_fake / nb,
            "grad_norm_d": total_grad_d / nb,
            "grad_norm_g": total_grad_g / nb,
            "lr_g": self.optimizer_G.param_groups[0]["lr"],
            "lr_d": self.optimizer_D.param_groups[0]["lr"],
            "tau": tau,
        }

    def evaluate_neural_validation(self) -> Dict[str, float]:
        """Compute neural validation metrics across validation partition batches."""
        self.generator.eval()
        self.discriminator.eval()

        total_loss_d = 0.0
        total_loss_g = 0.0
        total_acc = 0.0
        nb = 0

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

                loss_d = (loss_real + loss_fake).item()
                loss_g = self.criterion(out_fake, real_labels).item()

                acc_r = (torch.sigmoid(out_real) >= 0.5).float().mean().item()
                acc_f = (torch.sigmoid(out_fake) < 0.5).float().mean().item()
                acc = 0.5 * (acc_r + acc_f)

                total_loss_d += loss_d
                total_loss_g += loss_g
                total_acc += acc
                nb += 1

        nb = max(1, nb)
        return {
            "val_loss_d": total_loss_d / nb,
            "val_loss_g": total_loss_g / nb,
            "val_acc": total_acc / nb,
        }

    def evaluate_physical_validation(
        self,
        epoch: int,
    ) -> Dict[str, Any]:
        """Run forward BB84 quantum simulations on validation conditions C17-C20 using Common Random Numbers."""
        self.generator.eval()

        all_conds = get_full_environmental_conditions()
        val_conds = [c for c in all_conds if c.partition == "val"]
        assert len(val_conds) == 4, f"Expected 4 validation conditions, got {len(val_conds)}"

        policy_engine = cGANPolicyEngine(generator=self.generator, device=self.device)

        sim_records = []
        valid_count = 0
        feasible_count = 0
        indeterminate_count = 0
        infeasible_count = 0
        abort_count = 0

        info_gains_all = []
        info_gains_feas = []
        thetas_probe = []
        alphas_pir = []
        fam_counts = {"Probe": 0, "PIR": 0, "IR": 0, "Null": 0}

        with torch.no_grad():
            for c_idx, cond in enumerate(val_conds):
                c_tensor = torch.tensor(
                    [[cond.q_budget, cond.gamma_noise_hat]] * self.num_candidates_eval,
                    dtype=torch.float32,
                    device=self.device,
                )

                # Generate candidates using fixed validation latents (CRN)
                g_out = self.generator(self.fixed_val_latents, c_tensor, use_gumbel=False)
                fam_weights = g_out["family_weights"].cpu().numpy()
                alphas = g_out["alpha"].cpu().numpy().flatten()
                betas = g_out["beta"].cpu().numpy().flatten()
                thetas_rad = g_out["theta_E_rad"].cpu().numpy().flatten()

                for k_idx in range(self.num_candidates_eval):
                    fw = fam_weights[k_idx]
                    top_fam_idx = int(np.argmax(fw))

                    # Parse candidate
                    if top_fam_idx == 0:
                        fam = AttackFamily.PROBABILISTIC_INTERCEPT_RESEND
                        al = float(np.clip(alphas[k_idx], 0.0, 1.0))
                        be = float(np.clip(betas[k_idx], 0.0, 1.0))
                        params = AttackParameters(alpha=al, beta=be)
                        cand = AttackConfiguration(
                            attack_id=f"GEN-VAL-E{epoch}-C{c_idx}-K{k_idx}",
                            family=fam,
                            parameters=params,
                            theoretical_disturbance=0.25 * al,
                            description=f"Generated PIR alpha={al:.4f}, beta={be:.4f}",
                        )
                        fam_counts["PIR"] += 1
                        alphas_pir.append(al)
                    elif top_fam_idx == 1:
                        fam = AttackFamily.UNITARY_PROBE
                        th = float(np.clip(thetas_rad[k_idx], 0.0, math.pi / 4.0))
                        params = AttackParameters(theta_E_rad=th, theta_E_deg=math.degrees(th))
                        cand = AttackConfiguration(
                            attack_id=f"GEN-VAL-E{epoch}-C{c_idx}-K{k_idx}",
                            family=fam,
                            parameters=params,
                            theoretical_disturbance=math.sin(th) ** 2,
                            description=f"Generated Probe theta={math.degrees(th):.2f} deg",
                        )
                        fam_counts["Probe"] += 1
                        thetas_probe.append(th)
                    else:
                        fam = AttackFamily.NULL
                        params = AttackParameters()
                        cand = AttackConfiguration(
                            attack_id=f"GEN-VAL-E{epoch}-C{c_idx}-K{k_idx}",
                            family=fam,
                            parameters=params,
                            theoretical_disturbance=0.0,
                            description="Generated Null",
                        )
                        fam_counts["Null"] += 1

                    # 1. Hard Gate 1: Parameter validity check
                    is_valid = True
                    if fam == AttackFamily.UNITARY_PROBE:
                        if not (0.0 <= params.theta_E_rad <= (math.pi / 4.0 + 1e-9)):
                            is_valid = False
                    elif fam == AttackFamily.PROBABILISTIC_INTERCEPT_RESEND:
                        if not (0.0 <= params.alpha <= 1.0 and 0.0 <= params.beta <= 1.0):
                            is_valid = False

                    if is_valid:
                        valid_count += 1
                    else:
                        raise ValueError(f"Epoch {epoch}: Invalid policy parameters proposed: {cand}")

                    # 2. Forward Simulation with Common Random Numbers (CRN)
                    # Fixed seed per (condition, candidate) across all epochs
                    seed_sim = 50000 + c_idx * 100 + k_idx
                    rec = execute_single_simulation(
                        condition=cond,
                        attack_config=cand,
                        seed_sim=seed_sim,
                        n_pulses=self.n_pulses_eval,
                    )

                    feas_cat = rec.derived_evaluation.feasibility_category
                    aborted = rec.derived_evaluation.protocol_aborted
                    eve_info = float(rec.raw_observation.eve_info_gain)

                    if aborted:
                        abort_count += 1

                    # Statistical Feasibility (Indeterminate is explicitly NOT counted as feasible)
                    if feas_cat == StatisticalFeasibility.FEASIBLE:
                        feasible_count += 1
                        # Information gain reward applies strictly to feasible, non-aborted, non-null policies
                        if not aborted and fam != AttackFamily.NULL:
                            info_gains_feas.append(eve_info)
                    elif feas_cat == StatisticalFeasibility.INDETERMINATE:
                        indeterminate_count += 1
                    else:
                        infeasible_count += 1

                    info_gains_all.append(eve_info)
                    sim_records.append({
                        "condition_id": cond.condition_id,
                        "attack_family": fam.value,
                        "theta_E_rad": params.theta_E_rad,
                        "alpha": params.alpha,
                        "beta": params.beta,
                        "qber_test": rec.raw_observation.qber_test,
                        "wilson_ucb": rec.derived_evaluation.qber_wilson_ucb,
                        "q_budget": cond.q_budget,
                        "feasibility_category": feas_cat.value,
                        "aborted": aborted,
                        "eve_info_gain": eve_info,
                    })

        total_proposals = len(val_conds) * self.num_candidates_eval
        validity_rate = valid_count / total_proposals
        feasibility_yield = feasible_count / total_proposals
        abort_rate = abort_count / total_proposals

        # Information metrics on feasible policies
        if len(info_gains_feas) > 0:
            arr_feas = np.array(info_gains_feas)
            mean_info_feas = float(np.mean(arr_feas))
            median_info_feas = float(np.median(arr_feas))
            p75_info_feas = float(np.percentile(arr_feas, 75))
            p90_info_feas = float(np.percentile(arr_feas, 90))
            max_info_feas = float(np.max(arr_feas))
        else:
            mean_info_feas = 0.0
            median_info_feas = 0.0
            p75_info_feas = 0.0
            p90_info_feas = 0.0
            max_info_feas = 0.0

        # Continuous dispersion
        std_theta = float(np.std(thetas_probe)) if len(thetas_probe) > 1 else 0.0
        std_alpha = float(np.std(alphas_pir)) if len(alphas_pir) > 1 else 0.0
        dispersion_score = min(1.0, (std_theta / 0.01) + (std_alpha / 0.01))

        # Mode entropy
        total_fams = sum(fam_counts.values())
        entropy_bits = 0.0
        for cnt in fam_counts.values():
            if cnt > 0:
                p = cnt / total_fams
                entropy_bits -= p * math.log2(p)

        eval_summary = {
            "epoch": epoch,
            "total_evaluated": total_proposals,
            "validity_rate": validity_rate,
            "feasibility_yield": feasibility_yield,
            "feasible_count": feasible_count,
            "indeterminate_count": indeterminate_count,
            "infeasible_count": infeasible_count,
            "abort_rate": abort_rate,
            "abort_count": abort_count,
            "mean_info_gain_feas": mean_info_feas,
            "median_info_gain_feas": median_info_feas,
            "p75_info_gain_feas": p75_info_feas,
            "p90_info_gain_feas": p90_info_feas,
            "max_info_gain_feas": max_info_feas,
            "continuous_dispersion": dispersion_score,
            "std_theta_rad": std_theta,
            "std_alpha": std_alpha,
            "family_entropy_bits": entropy_bits,
            "family_distribution": {k: v / total_fams for k, v in fam_counts.items()},
            "sim_records": sim_records,
        }
        return eval_summary

    def evaluate_controlled_counterfactual(self, epoch: int) -> Dict[str, Any]:
        """Controlled Counterfactual Evaluation: vary Q_budget while holding gamma_hat_noise strictly fixed.

        Exact Calibration Values (from get_full_environmental_conditions):
        - Low-Noise Slice: C17 (p_depol=0.01, k=8, n=1000) -> gamma_hat = 0.008
        - Medium-Noise Slice: C19 (p_depol=0.03, k=12, n=1000) -> gamma_hat = 0.012
        """
        self.generator.eval()

        gamma_low = 0.008  # C17 exact calibration value
        gamma_med = 0.012  # C19 exact calibration value

        results = {}
        with torch.no_grad():
            for g_name, g_val in [("gamma_low_0.008", gamma_low), ("gamma_med_0.012", gamma_med)]:
                c_tight = torch.tensor([[0.050, g_val]] * self.num_candidates_eval, dtype=torch.float32, device=self.device)
                c_loose = torch.tensor([[0.110, g_val]] * self.num_candidates_eval, dtype=torch.float32, device=self.device)

                out_tight = self.generator(self.fixed_cf_latents, c_tight, use_gumbel=False)
                out_loose = self.generator(self.fixed_cf_latents, c_loose, use_gumbel=False)

                th_tight = out_tight["theta_E_rad"].cpu().numpy().flatten()
                th_loose = out_loose["theta_E_rad"].cpu().numpy().flatten()
                al_tight = out_tight["alpha"].cpu().numpy().flatten()
                al_loose = out_loose["alpha"].cpu().numpy().flatten()

                delta_th = th_loose - th_tight
                delta_al = al_loose - al_tight

                # Proportion of paired latents where theta increases or stays equal with looser budget
                p_th_increase = float(np.mean(delta_th >= -1e-5))
                p_al_increase = float(np.mean(delta_al >= -1e-5))

                results[g_name] = {
                    "mean_theta_q05_rad": float(np.mean(th_tight)),
                    "mean_theta_q11_rad": float(np.mean(th_loose)),
                    "mean_delta_theta_rad": float(np.mean(delta_th)),
                    "p_theta_increase": p_th_increase,
                    "mean_alpha_q05": float(np.mean(al_tight)),
                    "mean_alpha_q11": float(np.mean(al_loose)),
                    "mean_delta_alpha": float(np.mean(delta_al)),
                    "p_alpha_increase": p_al_increase,
                }

        return results

    def update_pareto_frontier(self, candidate: Dict[str, Any]):
        """Maintain the non-dominated Pareto frontier across (feasibility_yield, mean_info_gain_feas)."""
        y = candidate["feasibility_yield"]
        info = candidate["mean_info_gain_feas"]

        # Check if dominated by existing
        is_dominated = False
        for p in self.pareto_frontier:
            if p["feasibility_yield"] >= y and p["mean_info_gain_feas"] >= info:
                if p["feasibility_yield"] > y or p["mean_info_gain_feas"] > info:
                    is_dominated = True
                    break

        if not is_dominated:
            # Remove any candidates dominated by the new one
            self.pareto_frontier = [
                p for p in self.pareto_frontier
                if not (y >= p["feasibility_yield"] and info >= p["mean_info_gain_feas"] and (y > p["feasibility_yield"] or info > p["mean_info_gain_feas"]))
            ]
            self.pareto_frontier.append(candidate)

    def update_lexicographic_and_pareto_checkpoints(
        self,
        epoch: int,
        phys_eval: Dict[str, Any],
        neural_val: Dict[str, float],
    ):
        """Update and persist Lexicographic Best, Max Feasibility, Max Information, and Balanced Pareto checkpoints."""
        y_feas = phys_eval["feasibility_yield"]
        r_val = phys_eval["validity_rate"]
        abort_count = phys_eval["abort_count"]
        abort_rate = phys_eval["abort_rate"]
        mean_i_feas = phys_eval["mean_info_gain_feas"]
        std_th = phys_eval["std_theta_rad"]
        probe_count = phys_eval["family_distribution"].get("Probe", 0.0) * phys_eval["total_evaluated"]

        candidate = {
            "epoch": epoch,
            "validity_rate": r_val,
            "feasibility_yield": y_feas,
            "abort_count": abort_count,
            "abort_rate": abort_rate,
            "mean_info_gain_feas": mean_i_feas,
            "median_info_gain_feas": phys_eval["median_info_gain_feas"],
            "p75_info_gain_feas": phys_eval["p75_info_gain_feas"],
            "p90_info_gain_feas": phys_eval["p90_info_gain_feas"],
            "max_info_gain_feas": phys_eval["max_info_gain_feas"],
            "std_theta_rad": std_th,
            "val_loss_d": neural_val["val_loss_d"],
            "val_loss_g": neural_val["val_loss_g"],
        }
        self.evaluated_checkpoints.append(candidate)

        # Hard Gate 1: Physical Validity must be 100%
        if r_val < 1.0:
            return

        # Hard Gate 2: Zero Observed Aborts for stealth
        has_zero_aborts = (abort_count == 0)

        # Anti-Collapse Guard: If probe >= 4, require std_theta >= 0.003
        passes_diversity = (std_th >= 0.003) if probe_count >= 4 else True

        # Update Pareto Frontier
        if has_zero_aborts and passes_diversity:
            self.update_pareto_frontier(candidate)

        # 1. Lexicographic Best:
        # Priority 1: r_val == 1.0 (Passed above)
        # Priority 2: abort_count == 0
        # Priority 3: y_feas >= feasibility_gate_threshold (0.80)
        # Priority 4: Maximize mean_info_gain_feas
        # Tie-breaker: Higher y_feas
        is_qualified = has_zero_aborts and (y_feas >= self.feasibility_gate_threshold) and passes_diversity
        if is_qualified:
            is_better = False
            if self.lexicographic_best is None:
                is_better = True
            elif mean_i_feas > self.lexicographic_best["mean_info_gain_feas"]:
                is_better = True
            elif math.isclose(mean_i_feas, self.lexicographic_best["mean_info_gain_feas"], abs_tol=1e-5):
                # Tie-breaker: higher feasibility yield
                if y_feas > self.lexicographic_best["feasibility_yield"]:
                    is_better = True

            if is_better:
                self.lexicographic_best = candidate
                path = os.path.join(self.checkpoint_dir, "cgan_m4a_lexicographic_best.pt")
                self.save_checkpoint(path, epoch=epoch, tag="lexicographic_best", meta=candidate)

        # 2. Max Feasibility Best (Anchor on Pareto Frontier)
        if has_zero_aborts:
            is_max_feas = False
            if self.max_feasibility_best is None:
                is_max_feas = True
            elif y_feas > self.max_feasibility_best["feasibility_yield"]:
                is_max_feas = True
            elif math.isclose(y_feas, self.max_feasibility_best["feasibility_yield"], abs_tol=1e-5):
                if mean_i_feas > self.max_feasibility_best["mean_info_gain_feas"]:
                    is_max_feas = True

            if is_max_feas:
                self.max_feasibility_best = candidate
                path = os.path.join(self.checkpoint_dir, "cgan_m4a_max_feasibility.pt")
                self.save_checkpoint(path, epoch=epoch, tag="max_feasibility", meta=candidate)

        # 3. Max Information Best (Anchor on Pareto Frontier)
        if has_zero_aborts:
            is_max_info = False
            if self.max_information_best is None:
                is_max_info = True
            elif mean_i_feas > self.max_information_best["mean_info_gain_feas"]:
                is_max_info = True
            elif math.isclose(mean_i_feas, self.max_information_best["mean_info_gain_feas"], abs_tol=1e-5):
                if y_feas > self.max_information_best["feasibility_yield"]:
                    is_max_info = True

            if is_max_info:
                self.max_information_best = candidate
                path = os.path.join(self.checkpoint_dir, "cgan_m4a_max_information.pt")
                self.save_checkpoint(path, epoch=epoch, tag="max_information", meta=candidate)

        # 4. Balanced Pareto Compromise: Maximize scalar product Y_feas * mean_I_feas
        if has_zero_aborts:
            score_pareto = y_feas * mean_i_feas
            prev_best = (
                self.balanced_pareto_best["feasibility_yield"] * self.balanced_pareto_best["mean_info_gain_feas"]
                if self.balanced_pareto_best is not None
                else -1.0
            )
            if score_pareto > prev_best:
                self.balanced_pareto_best = candidate
                path = os.path.join(self.checkpoint_dir, "cgan_m4a_balanced_pareto.pt")
                self.save_checkpoint(path, epoch=epoch, tag="balanced_pareto", meta=candidate)

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        tag: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ):
        """Serialize model weights, optimizer states, and provenance metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "tag": tag,
            "metadata": meta or {},
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "history": self.history,
            "config": {
                "initialization_strategy": "Strategy A (Fresh Initialization from Scratch)",
                "latent_dim": self.generator.latent_dim,
                "condition_dim": self.generator.condition_dim,
                "policy_dim": self.discriminator.policy_dim,
                "lr_g": self.lr_g,
                "lr_d": self.lr_d,
                "tau_0": self.tau_0,
                "tau_min": self.tau_min,
                "tau_decay": self.tau_decay,
                "seed": self.seed,
            },
        }
        torch.save(checkpoint, path)

    def train(self, epochs: int = 200, verbose: bool = True) -> Dict[str, Any]:
        """Execute the full 200-epoch training and validation procedure."""
        t_start = time.perf_counter()
        process = psutil.Process(os.getpid())
        ram_start_mb = process.memory_info().rss / (1024.0 * 1024.0)

        if verbose:
            print("=" * 80)
            print("MILESTONE 4A: CLASSICAL cGAN TRAINING (200 EPOCHS)")
            print(f"Device: {self.device} | PyTorch: {torch.__version__} | Seed: {self.seed}")
            print(f"Checkpoints: {self.checkpoint_dir} | Results: {self.results_dir}")
            print(f"Initialization Strategy: Strategy A (Fresh Weights from Scratch)")
            print(f"Gumbel Schedule: tau_0={self.tau_0} -> tau_min={self.tau_min} (decay={self.tau_decay})")
            print(f"Lexicographic Feasibility Gate: Y_feas >= {self.feasibility_gate_threshold*100:.1f}%, Zero Aborts")
            print("=" * 80)

        last_phys_eval = {
            "feasibility_yield": 0.0,
            "abort_rate": 0.0,
            "validity_rate": 1.0,
            "mean_info_gain_feas": 0.0,
            "median_info_gain_feas": 0.0,
            "p75_info_gain_feas": 0.0,
            "p90_info_gain_feas": 0.0,
            "max_info_gain_feas": 0.0,
            "continuous_dispersion": 1.0,
            "std_theta_rad": 0.0,
            "std_alpha": 0.0,
            "family_entropy_bits": 0.50,
            "family_distribution": {"Probe": 1.0, "PIR": 0.0, "IR": 0.0, "Null": 0.0},
        }
        last_cf_eval: Dict[str, Any] = {}

        for epoch in range(1, epochs + 1):
            t_ep_start = time.perf_counter()

            # 1. Train epoch
            train_metrics = self.train_epoch(epoch)

            # 2. Neural validation evaluation (fast, every epoch)
            neural_val = self.evaluate_neural_validation()

            # 3. Physical quantum validation simulation (periodic)
            run_physical = (
                epoch % self.eval_physical_interval == 0
                or epoch == 1
                or epoch == epochs
            )
            if run_physical:
                phys_eval = self.evaluate_physical_validation(epoch)
                cf_eval = self.evaluate_controlled_counterfactual(epoch)
                self.physical_evaluations.append({
                    "physical_eval": phys_eval,
                    "counterfactual_eval": cf_eval,
                })
                last_phys_eval = phys_eval
                last_cf_eval = cf_eval

                # Update Lexicographic and Pareto Checkpoints
                self.update_lexicographic_and_pareto_checkpoints(epoch, phys_eval, neural_val)

            # Compute equilibrium diagnostic proximity P_eq
            nash_loss = 2.0 * math.log(2.0)
            delta_eq = abs(neural_val["val_loss_d"] - nash_loss) / nash_loss
            p_eq = max(0.0, 1.0 - delta_eq)

            t_ep = time.perf_counter() - t_ep_start
            vram_alloc = torch.cuda.memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
            vram_res = torch.cuda.memory_reserved() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0

            # Record history
            self.history["epoch"].append(epoch)
            self.history["loss_d"].append(round(train_metrics["loss_d"], 6))
            self.history["loss_d_real"].append(round(train_metrics["loss_d_real"], 6))
            self.history["loss_d_fake"].append(round(train_metrics["loss_d_fake"], 6))
            self.history["loss_g"].append(round(train_metrics["loss_g"], 6))
            self.history["d_acc_real"].append(round(train_metrics["d_acc_real"], 4))
            self.history["d_acc_fake"].append(round(train_metrics["d_acc_fake"], 4))
            self.history["grad_norm_d"].append(round(train_metrics["grad_norm_d"], 4))
            self.history["grad_norm_g"].append(round(train_metrics["grad_norm_g"], 4))
            self.history["lr_g"].append(train_metrics["lr_g"])
            self.history["lr_d"].append(train_metrics["lr_d"])
            self.history["tau"].append(round(train_metrics["tau"], 4))
            self.history["val_loss_d"].append(round(neural_val["val_loss_d"], 6))
            self.history["val_loss_g"].append(round(neural_val["val_loss_g"], 6))
            self.history["val_acc"].append(round(neural_val["val_acc"], 4))
            self.history["feasibility_yield"].append(round(last_phys_eval["feasibility_yield"], 4))
            self.history["abort_rate"].append(round(last_phys_eval["abort_rate"], 4))
            self.history["validity_rate"].append(round(last_phys_eval["validity_rate"], 4))
            self.history["mean_info_gain_feas"].append(round(last_phys_eval["mean_info_gain_feas"], 6))
            self.history["median_info_gain_feas"].append(round(last_phys_eval["median_info_gain_feas"], 6))
            self.history["p75_info_gain_feas"].append(round(last_phys_eval["p75_info_gain_feas"], 6))
            self.history["p90_info_gain_feas"].append(round(last_phys_eval["p90_info_gain_feas"], 6))
            self.history["max_info_gain_feas"].append(round(last_phys_eval["max_info_gain_feas"], 6))
            self.history["continuous_dispersion"].append(round(last_phys_eval["continuous_dispersion"], 4))
            self.history["std_theta_rad"].append(round(last_phys_eval["std_theta_rad"], 6))
            self.history["std_alpha"].append(round(last_phys_eval["std_alpha"], 6))
            self.history["family_entropy"].append(round(last_phys_eval["family_entropy_bits"], 4))
            self.history["family_probs"].append(last_phys_eval["family_distribution"])
            self.history["equilibrium_proximity"].append(round(p_eq, 4))
            self.history["counterfactual_sensitivity"].append(last_cf_eval)
            self.history["vram_allocated_mb"].append(round(vram_alloc, 2))
            self.history["vram_reserved_mb"].append(round(vram_res, 2))
            self.history["epoch_duration_sec"].append(round(t_ep, 4))

            if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
                phys_str = (
                    f"Feas: {last_phys_eval['feasibility_yield']*100:.1f}% | Aborts: {last_phys_eval['abort_count']} | "
                    f"I(A;E): {last_phys_eval['mean_info_gain_feas']:.4f}b (med:{last_phys_eval['median_info_gain_feas']:.4f}b)"
                    if run_physical
                    else ""
                )
                print(
                    f"Epoch [{epoch:3d}/{epochs:3d}] ({t_ep:.1f}s) | "
                    f"Loss D: {train_metrics['loss_d']:.4f} | G: {train_metrics['loss_g']:.4f} | "
                    f"Acc D: {train_metrics['d_acc_real']*100:.0f}%/{train_metrics['d_acc_fake']*100:.0f}% | "
                    f"Tau: {train_metrics['tau']:.3f} | {phys_str}"
                )

        t_total = time.perf_counter() - t_start
        ram_end_mb = process.memory_info().rss / (1024.0 * 1024.0)

        # Save latest checkpoint
        latest_ckpt_path = os.path.join(self.checkpoint_dir, "cgan_m4a_latest.pt")
        self.save_checkpoint(latest_ckpt_path, epoch=epochs, tag="latest", meta={"epoch": epochs})

        # Save history
        history_path = os.path.join(self.results_dir, "cgan_m4a_training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "initialization_strategy": "Strategy A (Fresh from Scratch, Seed 42)",
                    "epochs": epochs,
                    "seed": self.seed,
                    "device": str(self.device),
                    "total_duration_sec": round(t_total, 2),
                    "initial_ram_mb": round(ram_start_mb, 2),
                    "final_ram_mb": round(ram_end_mb, 2),
                    "qualified_lexicographic_found": self.lexicographic_best is not None,
                    "lexicographic_best": self.lexicographic_best,
                    "max_feasibility_best": self.max_feasibility_best,
                    "max_information_best": self.max_information_best,
                    "balanced_pareto_best": self.balanced_pareto_best,
                    "pareto_frontier_count": len(self.pareto_frontier),
                    "pareto_frontier": self.pareto_frontier,
                },
                "history": self.history,
            }, f, indent=2)

        evals_path = os.path.join(self.results_dir, "cgan_m4a_validation_evaluations.json")
        with open(evals_path, "w", encoding="utf-8") as f:
            json.dump({
                "evaluations": self.physical_evaluations,
            }, f, indent=2)

        if verbose:
            print("=" * 80)
            print(f"MILESTONE 4A TRAINING COMPLETE in {t_total:.1f}s ({t_total/60.0:.2f} min)")
            print(f"Qualified Lexicographic Best Checkpoint: {self.lexicographic_best}")
            print(f"Max Feasibility Best Checkpoint: {self.max_feasibility_best}")
            print(f"Max Information Best Checkpoint: {self.max_information_best}")
            print(f"Balanced Pareto Checkpoint: {self.balanced_pareto_best}")
            print(f"Non-Dominated Pareto Frontier Checkpoints: {len(self.pareto_frontier)}")
            print(f"History Saved: {history_path}")
            print(f"Evaluations Saved: {evals_path}")
            print("=" * 80)

        return {
            "total_time_seconds": t_total,
            "lexicographic_best": self.lexicographic_best,
            "max_feasibility_best": self.max_feasibility_best,
            "max_information_best": self.max_information_best,
            "balanced_pareto_best": self.balanced_pareto_best,
            "pareto_frontier": self.pareto_frontier,
            "history": self.history,
        }


def run_milestone4a_training(
    dataset_path: str = "results/full_training_dataset.json",
    checkpoint_dir: str = "checkpoints/milestone4a",
    results_dir: str = "results/milestone4a",
    epochs: int = 200,
    batch_size: int = 64,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    beta1: float = 0.5,
    beta2: float = 0.999,
    latent_dim: int = 16,
    tau_0: float = 1.0,
    tau_min: float = 0.20,
    tau_decay: float = 0.985,
    tau_I: float = 0.10,
    seed: int = 42,
    eval_physical_interval: int = 10,
    feasibility_gate_threshold: float = 0.80,
) -> Dict[str, Any]:
    """Execute complete Milestone 4A Classical cGAN experiment."""
    # Ensure test quarantine
    loaders = create_dataloaders(
        dataset_source=dataset_path,
        batch_size=batch_size,
        tau_I=tau_I,
        use_weighted_sampling=True,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Verify partition separation
    train_conds = {r.condition.condition_id for r in loaders["train_dataset"].filtered_records}
    val_conds = {r.condition.condition_id for r in loaders["val_dataset"].filtered_records}
    test_conds = {r.condition.condition_id for r in loaders["test_dataset"].filtered_records}

    assert train_conds.isdisjoint(test_conds), "CRITICAL: Train partition overlaps with Test partition!"
    assert val_conds.isdisjoint(test_conds), "CRITICAL: Validation partition overlaps with Test partition!"

    generator = ConditionalGenerator(latent_dim=latent_dim, condition_dim=2, hidden_dims=[128, 128])
    discriminator = ConditionalDiscriminator(policy_dim=5, condition_dim=2, hidden_dims=[128, 64], use_spectral_norm=True)

    trainer = Milestone4ATrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_g=lr_g,
        lr_d=lr_d,
        beta1=beta1,
        beta2=beta2,
        tau_0=tau_0,
        tau_min=tau_min,
        tau_decay=tau_decay,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        seed=seed,
        eval_physical_interval=eval_physical_interval,
        feasibility_gate_threshold=feasibility_gate_threshold,
    )

    return trainer.train(epochs=epochs, verbose=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Milestone 4A Full Classical cGAN Training Experiment")
    parser.add_argument("--execute", action="store_true", help="Authorize and execute the full 200-epoch training experiment.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default 200).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    args = parser.parse_args()

    if args.execute:
        print("Explicit authorization detected. Starting Milestone 4A Full Training...")
        run_milestone4a_training(epochs=args.epochs, seed=args.seed)
    else:
        print("Milestone 4A Runner initialized in SAFE mode. Awaiting explicit authorization (--execute) before training.")
