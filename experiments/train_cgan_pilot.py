"""Pilot 20-Epoch Training and Comprehensive Validation for BB84 Conditional GAN.

Executes:
1. Exactly 20 epochs of cGAN training on the training partition (C01-C16).
2. Validation evaluation at every epoch on held-out validation conditions (C17-C20).
3. Test conditions C21-C24 strictly quarantined and untouched.
4. Comprehensive post-pilot physical evaluation:
   - Evaluates K=16 proposed policies per validation condition (C17-C20) on the physical BB84 simulator.
   - Measures QBER, Wilson 95% confidence intervals, statistical feasibility yield, Eve mutual info.
   - Checks mode-collapse metrics, policy diversity, and parameter bounds.
   - Saves complete results to results/cgan_pilot_training_history.json and results/cgan_pilot_validation_evaluation.json.
"""

import datetime
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

from src.cgan.dataset import create_dataloaders
from src.cgan.models import ConditionalDiscriminator, ConditionalGenerator
from src.cgan.policy_engine import cGANPolicyEngine, OperationalDecision
from src.cgan.trainer import CGANTrainer
from src.dataset.generator import (
    execute_single_simulation,
    get_full_environmental_conditions,
)
from src.dataset.schema import AttackFamily, StatisticalFeasibility


def run_pilot_training_and_validation(
    dataset_path: str = "results/full_training_dataset.json",
    checkpoint_dir: str = "checkpoints",
    history_path: str = "results/cgan_pilot_training_history.json",
    eval_output_path: str = "results/cgan_pilot_validation_evaluation.json",
    epochs: int = 20,
    batch_size: int = 64,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    latent_dim: int = 16,
    seed: int = 42,
    tau_start: float = 1.0,
    tau_min: float = 0.20,
    tau_I: float = 0.10,
    num_candidates_eval: int = 16,
    n_pulses_eval: int = 2000,
) -> Dict[str, Any]:
    """Execute 20-epoch pilot training and detailed post-pilot physical validation."""
    print("=" * 80)
    print("M3 STAGE C: AUTHORIZED PILOT cGAN TRAINING (20 EPOCHS MAXIMUM)")
    print("=" * 80)

    # 1. Hardware & Environment Telemetry
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_mb = (
        torch.cuda.get_device_properties(0).total_memory / (1024.0 * 1024.0)
        if torch.cuda.is_available()
        else 0.0
    )
    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024.0 * 1024.0)

    print(f"Device: {device} ({cuda_name})")
    print(f"VRAM Capacity: {total_vram_mb:.1f} MB | Initial RAM: {ram_start_mb:.1f} MB")
    print(f"Random Seed: {seed} | Epochs: {epochs} (Strict Maximum) | Batch Size: {batch_size}")
    print(f"Annealing Tau: {tau_start:.2f} -> {tau_min:.2f} | Info-Gain Temp tau_I: {tau_I:.2f}")

    # 2. Build DataLoaders (Training partition C01-C16, Validation C17-C20)
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

    print(f"Partition Records (Feasible Only):")
    print(f"  Train: {len(loaders['train_dataset'])} records (Conditions C01-C16)")
    print(f"  Val:   {len(loaders['val_dataset'])} records (Conditions C17-C20)")
    print(f"  Test:  {len(loaders['test_dataset'])} records (Conditions C21-C24) - STRICTLY QUARANTINED")

    # 3. Model Architecture Instantiation
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
    print(f"\nModel Parameters:")
    print(f"  Generator G_phi:     {total_params_g:,} parameters")
    print(f"  Discriminator D_psi: {total_params_d:,} parameters")
    print(f"  Total Parameters:    {total_params_g + total_params_d:,} parameters")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    # 4. Initialize Trainer
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

    # 5. Execute 20-Epoch Pilot Training Run
    t_train_start = time.perf_counter()
    print("\n--- Commencing 20-Epoch Pilot Training Loop ---")

    best_pilot_ckpt = os.path.join(checkpoint_dir, "cgan_pilot_best.pt")
    latest_pilot_ckpt = os.path.join(checkpoint_dir, "cgan_pilot_latest.pt")
    best_standard_ckpt = os.path.join(checkpoint_dir, "cgan_best.pt")

    # Train epoch by epoch with validation at every epoch
    fit_summary = trainer.fit(
        epochs=epochs,
        eval_every=1,  # Evaluate validation at every epoch for detailed pilot diagnostics
        verbose=True,
        save_best=True,
    )
    t_train_total = time.perf_counter() - t_train_start

    # Save dedicated pilot checkpoints
    trainer.save_checkpoint(latest_pilot_ckpt, epoch=epochs, is_best=False)
    # Also save as cgan_pilot_best.pt
    if os.path.exists(best_standard_ckpt):
        import shutil
        shutil.copy2(best_standard_ckpt, best_pilot_ckpt)

    ram_end_mb = process.memory_info().rss / (1024.0 * 1024.0)
    vram_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        if torch.cuda.is_available()
        else 0.0
    )

    # 6. Save Complete Training History
    training_summary_payload = {
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
                "best": best_pilot_ckpt,
                "latest": latest_pilot_ckpt,
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
        "history": trainer.history,
    }

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_summary_payload, f, indent=2)

    print(f"\nTraining History Saved to: {history_path}")

    # 7. Post-Pilot Physical Simulation on Validation Conditions (Section 4)
    print("\n" + "=" * 80)
    print("M3 STAGE C: POST-PILOT VALIDATION EVALUATION (PHYSICAL SIMULATION)")
    print("=" * 80)

    # Load best checkpoint into policy engine
    policy_engine = cGANPolicyEngine(generator=generator, device=device)
    if os.path.exists(best_pilot_ckpt):
        ckpt_data = torch.load(best_pilot_ckpt, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt_data["generator_state_dict"])
        print(f"Loaded best generator weights from epoch {ckpt_data.get('epoch', 'N/A')}")

    # Get validation conditions (C17 to C20)
    all_conds = get_full_environmental_conditions()
    val_conditions = [c for c in all_conds if c.partition == "val"]
    print(f"Evaluating {len(val_conditions)} validation conditions with K={num_candidates_eval} physical queries each:")

    val_evaluation_records: Dict[str, Any] = {}
    total_valid_policies = 0
    total_feasible_candidates = 0
    total_candidates = len(val_conditions) * num_candidates_eval

    all_alphas: List[float] = []
    all_thetas: List[float] = []
    family_counter = {"PIR": 0, "Unitary Probe": 0}

    t_val_start = time.perf_counter()
    for cond in val_conditions:
        print(f"\nCondition {cond.condition_id}: p_depol={cond.p_depol:.2f}, Q_budget={cond.q_budget:.2f}")

        # Propose K candidates from generator
        candidates = policy_engine.propose_candidate_configurations(
            condition=cond,
            num_candidates=num_candidates_eval,
            seed=seed * 10 + int(cond.condition_id[1:]),
        )

        cond_records = []
        cond_feasible = 0
        cond_valid = 0

        for i, cand in enumerate(candidates):
            # Check validity
            is_valid = True
            if cand.family == AttackFamily.PROBABILISTIC_INTERCEPT_RESEND:
                family_counter["PIR"] += 1
                a = cand.parameters.alpha
                b = cand.parameters.beta
                if a is not None:
                    all_alphas.append(a)
                if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
                    is_valid = False
            elif cand.family == AttackFamily.UNITARY_PROBE:
                family_counter["Unitary Probe"] += 1
                th = cand.parameters.theta_E_rad
                if th is not None:
                    all_thetas.append(th)
                if not (0.0 <= th <= math.pi / 4.0 + 1e-6):
                    is_valid = False

            if is_valid:
                cond_valid += 1
                total_valid_policies += 1

            # Execute forward physical simulation
            sim_seed = 8000 + int(cond.condition_id[1:]) * 100 + i
            rec = execute_single_simulation(
                condition=cond,
                attack_config=cand,
                seed_sim=sim_seed,
                n_pulses=n_pulses_eval,
                rho_test=0.50,
            )

            is_feas = rec.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE
            if is_feas:
                cond_feasible += 1
                total_feasible_candidates += 1

            cond_records.append({
                "candidate_index": i,
                "attack_id": cand.attack_id,
                "family": cand.family.value,
                "parameters": cand.parameters.to_dict(),
                "sample_qber_test": round(rec.raw_observation.qber_test, 5),
                "wilson_center": round(rec.derived_evaluation.wilson_q_center, 5),
                "wilson_lcb": round(rec.derived_evaluation.wilson_q_lcb, 5),
                "wilson_ucb": round(rec.derived_evaluation.wilson_q_ucb, 5),
                "wilson_margin": round(rec.derived_evaluation.wilson_margin, 5),
                "feasibility": rec.derived_evaluation.feasibility_category.value,
                "eve_info_gain": round(rec.raw_observation.eve_info_gain, 5),
                "protocol_abort": rec.raw_observation.protocol_abort,
            })

        val_evaluation_records[cond.condition_id] = {
            "p_depol": cond.p_depol,
            "q_budget": cond.q_budget,
            "gamma_noise_hat": cond.gamma_noise_hat,
            "candidates_evaluated": num_candidates_eval,
            "valid_parameter_count": cond_valid,
            "feasible_count": cond_feasible,
            "feasible_rate": round(cond_feasible / num_candidates_eval, 4),
            "candidates": cond_records,
        }

        print(f"  Valid Proposals: {cond_valid}/{num_candidates_eval} | Feasible Candidates: {cond_feasible}/{num_candidates_eval}")

    t_val_total = time.perf_counter() - t_val_start

    # 8. Mode Collapse Diagnostics
    alpha_std = float(np.std(all_alphas)) if len(all_alphas) > 1 else 0.0
    theta_std = float(np.std(all_thetas)) if len(all_thetas) > 1 else 0.0
    pir_ratio = family_counter["PIR"] / max(1, total_candidates)
    probe_ratio = family_counter["Unitary Probe"] / max(1, total_candidates)

    mode_collapse_risk = "LOW"
    if alpha_std < 0.01 and theta_std < 0.01:
        mode_collapse_risk = "HIGH (Severe continuous parameter collapse)"
    elif pir_ratio == 0.0 or probe_ratio == 0.0:
        mode_collapse_risk = "MODERATE (Single family dominance)"
    elif alpha_std < 0.03 or theta_std < 0.03:
        mode_collapse_risk = "MODERATE (Narrow continuous variance)"

    val_diagnostics = {
        "summary": {
            "total_candidates_generated": total_candidates,
            "total_valid_policies": total_valid_policies,
            "valid_policy_rate": round(total_valid_policies / total_candidates, 4),
            "total_feasible_candidates": total_feasible_candidates,
            "overall_feasibility_yield": round(total_feasible_candidates / total_candidates, 4),
            "family_diversity": {
                "PIR_count": family_counter["PIR"],
                "PIR_ratio": round(pir_ratio, 4),
                "Probe_count": family_counter["Unitary Probe"],
                "Probe_ratio": round(probe_ratio, 4),
            },
            "continuous_diversity": {
                "alpha_count": len(all_alphas),
                "alpha_mean": round(float(np.mean(all_alphas)), 4) if all_alphas else 0.0,
                "alpha_std": round(alpha_std, 4),
                "theta_count": len(all_thetas),
                "theta_mean_rad": round(float(np.mean(all_thetas)), 4) if all_thetas else 0.0,
                "theta_std_rad": round(theta_std, 4),
            },
            "mode_collapse_assessment": mode_collapse_risk,
            "physical_evaluation_runtime_seconds": round(t_val_total, 2),
        },
        "condition_evaluations": val_evaluation_records,
    }

    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(val_diagnostics, f, indent=2)

    print("\n" + "=" * 80)
    print("PILOT EVALUATION SUMMARY")
    print(f"Valid Parameter Rate: {total_valid_policies}/{total_candidates} ({100.0*total_valid_policies/total_candidates:.1f}%)")
    print(f"Physical Feasibility Rate: {total_feasible_candidates}/{total_candidates} ({100.0*total_feasible_candidates/total_candidates:.1f}%)")
    print(f"Family Distribution: PIR={family_counter['PIR']} ({100.0*pir_ratio:.1f}%), Probe={family_counter['Unitary Probe']} ({100.0*probe_ratio:.1f}%)")
    print(f"Parameter Stds: alpha_std={alpha_std:.4f}, theta_std={theta_std:.4f} rad")
    print(f"Mode Collapse Assessment: {mode_collapse_risk}")
    print(f"Validation Evaluation Saved to: {eval_output_path}")
    print("=" * 80)

    return {
        "training_summary": training_summary_payload,
        "validation_diagnostics": val_diagnostics,
    }


if __name__ == "__main__":
    run_pilot_training_and_validation()

