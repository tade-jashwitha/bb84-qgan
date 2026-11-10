"""Experimental Validation Harness for Milestone 1B.

Executes and documents the 5 required research experiments:
1. Experiment 1: Ideal BB84 (N=10,000, noiseless, no Eve, expected QBER = 0.0%).
2. Experiment 2: Full Intercept-Resend (N=20,000, alpha=1.0, beta=0.5, expected QBER ~ 25.0%).
3. Experiment 3: Partial Intercept-Resend (alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], alpha vs QBER and I_AE).
4. Experiment 4: Basis Bias Analysis (beta in [0.0, 0.25, 0.5, 0.75, 1.0], QBER, QBER_Z, QBER_X, basis dist, I_AE).
5. Experiment 5: Unitary Probe Evaluation (theta_E in [0, pi/12, pi/8, pi/6, pi/5, pi/4], disturbance, info gain, trade-off).

Saves plots and structured JSON metrics to `results/`.
"""

import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["MPLCONFIGDIR"] = os.path.abspath("results")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from src.bb84.alice import Alice, BASIS_Z, BASIS_X
from src.bb84.bob import Bob
from src.bb84.channel import QuantumChannel
from src.bb84.attacks import (
    InterceptResendAttack,
    FuchsPeresBrandtProbe,
    binary_entropy,
)
from src.bb84.bb84_engine import BB84Engine


def run_experiment_1(engine: BB84Engine) -> dict:
    print("=" * 70)
    print("EXPERIMENT 1: Ideal BB84 Protocol (N=10,000, No Eve, No Channel Noise)")
    print("=" * 70)
    t0 = time.time()
    res = engine.run(n_signals=10000, attack=None, seed_simulator=42)
    elapsed = time.time() - t0

    metrics = {
        "num_signals": res.num_signals,
        "sifted_key_length": res.sifted_key_length,
        "sifting_efficiency": float(res.sifting_efficiency),
        "error_count": res.error_count,
        "qber": float(res.qber),
        "qber_z": float(res.qber_z),
        "qber_x": float(res.qber_x),
        "elapsed_sec": float(elapsed),
    }

    print(f"Signals transmitted : {res.num_signals}")
    print(f"Sifted key bits     : {res.sifted_key_length} (Efficiency: {res.sifting_efficiency * 100:.2f}%)")
    print(f"Errors observed     : {res.error_count}")
    print(f"Total QBER          : {res.qber * 100:.4f}% (Theoretical: 0.0000%)")
    print(f"Z-basis QBER        : {res.qber_z * 100:.4f}% (Z sifted bits: {res.sifted_count_z})")
    print(f"X-basis QBER        : {res.qber_x * 100:.4f}% (X sifted bits: {res.sifted_count_x})")
    print(f"Runtime             : {elapsed:.2f} s\n")
    return metrics


def run_experiment_2(engine: BB84Engine) -> dict:
    print("=" * 70)
    print("EXPERIMENT 2: Full Intercept-Resend Attack (N=20,000, alpha=1.0, beta=0.5)")
    print("=" * 70)
    t0 = time.time()
    attack = InterceptResendAttack(alpha=1.0, beta=0.5, seed=12345)
    res = engine.run(n_signals=20000, attack=attack, seed_simulator=54321)
    elapsed = time.time() - t0

    theoretical_qber = attack.theoretical_qber()
    theoretical_info = attack.theoretical_information_gain()
    empirical_info = res.attack_metadata.get("empirical_information_gain", 0.0)

    metrics = {
        "num_signals": res.num_signals,
        "sifted_key_length": res.sifted_key_length,
        "sifting_efficiency": float(res.sifting_efficiency),
        "error_count": res.error_count,
        "qber": float(res.qber),
        "theoretical_qber": float(theoretical_qber),
        "qber_z": float(res.qber_z),
        "qber_x": float(res.qber_x),
        "eve_theoretical_info_bits": float(theoretical_info),
        "eve_empirical_info_bits": float(empirical_info),
        "intercept_count": res.attack_metadata.get("intercept_count"),
        "intercept_ratio": res.attack_metadata.get("intercept_ratio"),
        "elapsed_sec": float(elapsed),
    }

    print(f"Signals transmitted : {res.num_signals}")
    print(f"Intercepted pulses  : {res.attack_metadata.get('intercept_count')} (100.0%)")
    print(f"Sifted key bits     : {res.sifted_key_length} (Efficiency: {res.sifting_efficiency * 100:.2f}%)")
    print(f"Errors observed     : {res.error_count}")
    print(f"Total QBER          : {res.qber * 100:.2f}% (Theoretical: {theoretical_qber * 100:.2f}%)")
    print(f"Z-basis QBER        : {res.qber_z * 100:.2f}% (Z sifted bits: {res.sifted_count_z})")
    print(f"X-basis QBER        : {res.qber_x * 100:.2f}% (X sifted bits: {res.sifted_count_x})")
    print(f"Eve Mutual Info I_AE: {empirical_info:.4f} bits (Theoretical: {theoretical_info:.4f} bits)")
    print(f"Runtime             : {elapsed:.2f} s\n")
    return metrics


def run_experiment_3(engine: BB84Engine, output_dir: str) -> dict:
    print("=" * 70)
    print("EXPERIMENT 3: Partial Intercept-Resend (alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])")
    print("=" * 70)

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_per_alpha = 10000

    results_table = []
    for a in alphas:
        t0 = time.time()
        attack = InterceptResendAttack(alpha=a, beta=0.5, seed=int(a * 1000 + 42))
        res = engine.run(n_signals=n_per_alpha, attack=attack, seed_simulator=101)
        el = time.time() - t0

        theo_q = attack.theoretical_qber()
        theo_i = attack.theoretical_information_gain()
        emp_i = res.attack_metadata.get("empirical_information_gain", 0.0)

        results_table.append({
            "alpha": float(a),
            "observed_qber": float(res.qber),
            "theoretical_qber": float(theo_q),
            "observed_info_gain": float(emp_i),
            "theoretical_info_gain": float(theo_i),
            "intercept_count": res.attack_metadata.get("intercept_count", 0),
            "sifted_bits": res.sifted_key_length,
            "runtime_sec": float(el),
        })
        print(f"alpha={a:.1f} | Obs QBER={res.qber * 100:5.2f}% (Theo {theo_q * 100:5.2f}%) | "
              f"Obs I_AE={emp_i:.4f} bits (Theo {theo_i:.4f} bits) | Time={el:.2f}s")

    # Generate Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    a_vals = [r["alpha"] for r in results_table]
    obs_q = [r["observed_qber"] * 100 for r in results_table]
    theo_q = [r["theoretical_qber"] * 100 for r in results_table]
    obs_i = [r["observed_info_gain"] for r in results_table]
    theo_i = [r["theoretical_info_gain"] for r in results_table]

    # Plot 1: Attack Probability vs QBER
    ax1.plot(a_vals, theo_q, "r--", linewidth=2, label=r"Theory: $\text{QBER} = \alpha / 4$")
    ax1.plot(a_vals, obs_q, "bo-", linewidth=1.5, markersize=7, label="Observed QBER (Simulated)")
    ax1.axhline(11.0, color="darkred", linestyle=":", label=r"Protocol Abort Threshold ($11\%$)")
    ax1.set_xlabel(r"Interception Probability ($\alpha$)", fontsize=11)
    ax1.set_ylabel("QBER (%)", fontsize=11)
    ax1.set_title("Attack Probability vs. QBER", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    # Plot 2: Attack Probability vs Information Gain
    ax2.plot(a_vals, theo_i, "g--", linewidth=2, label=r"Theory: $1 - h_2(0.5 - 0.25\alpha)$")
    ax2.plot(a_vals, obs_i, "ms-", linewidth=1.5, markersize=7, label="Observed $I_{AE}$ (Bits)")
    ax2.set_xlabel(r"Interception Probability ($\alpha$)", fontsize=11)
    ax2.set_ylabel(r"Eve Mutual Information $I_{AE}$ (Bits / Sifted Pulse)", fontsize=11)
    ax2.set_title("Attack Probability vs. Information Gain", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "exp3_partial_intercept_resend.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {plot_path}\n")

    return {"data": results_table, "plot_path": plot_path}


def run_experiment_4(engine: BB84Engine, output_dir: str) -> dict:
    print("=" * 70)
    print("EXPERIMENT 4: Basis Bias Analysis (beta in [0.0, 0.25, 0.5, 0.75, 1.0], alpha=1.0)")
    print("=" * 70)

    betas = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_signals = 10000

    results_table = []
    for b in betas:
        t0 = time.time()
        attack = InterceptResendAttack(alpha=1.0, beta=b, seed=int(b * 1000 + 77))
        res = engine.run(n_signals=n_signals, attack=attack, seed_simulator=202)
        el = time.time() - t0

        theo_qz, theo_qx = attack.theoretical_qber_by_basis()
        theo_i = attack.theoretical_information_gain()
        emp_i = res.attack_metadata.get("empirical_information_gain", 0.0)

        results_table.append({
            "beta": float(b),
            "qber_total": float(res.qber),
            "qber_z": float(res.qber_z),
            "theo_qber_z": float(theo_qz),
            "qber_x": float(res.qber_x),
            "theo_qber_x": float(theo_qx),
            "eve_z_fraction": float(np.mean(res.attack_metadata["eve_bases"] == BASIS_Z)),
            "eve_x_fraction": float(np.mean(res.attack_metadata["eve_bases"] == BASIS_X)),
            "observed_info_gain": float(emp_i),
            "theoretical_info_gain": float(theo_i),
            "runtime_sec": float(el),
        })

        print(f"beta={b:4.2f} | Total QBER={res.qber * 100:5.2f}% | "
              f"Z-QBER={res.qber_z * 100:5.2f}% (Theo {theo_qz * 100:5.2f}%) | "
              f"X-QBER={res.qber_x * 100:5.2f}% (Theo {theo_qx * 100:5.2f}%) | "
              f"Eve Z-frac={np.mean(res.attack_metadata['eve_bases'] == BASIS_Z):.3f} | "
              f"I_AE={emp_i:.4f} bits")

    # Generate Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    b_vals = [r["beta"] for r in results_table]
    q_tot = [r["qber_total"] * 100 for r in results_table]
    q_z = [r["qber_z"] * 100 for r in results_table]
    q_x = [r["qber_x"] * 100 for r in results_table]
    theo_qz = [r["theo_qber_z"] * 100 for r in results_table]
    theo_qx = [r["theo_qber_x"] * 100 for r in results_table]
    z_fracs = [r["eve_z_fraction"] * 100 for r in results_table]

    # Plot 1: Basis Bias vs QBER Splits
    ax1.plot(b_vals, [25.0] * len(b_vals), "k--", linewidth=1.5, label=r"Total QBER (Theory: $25\%$)")
    ax1.plot(b_vals, q_tot, "ko", markersize=6, label="Total QBER (Simulated)")
    ax1.plot(b_vals, theo_qz, "b--", linewidth=1.5, label=r"Z-QBER Theory: $0.5(1-\beta)$")
    ax1.plot(b_vals, q_z, "bs-", markersize=6, label="Z-QBER (Simulated)")
    ax1.plot(b_vals, theo_qx, "r--", linewidth=1.5, label=r"X-QBER Theory: $0.5\beta$")
    ax1.plot(b_vals, q_x, "r^-", markersize=6, label="X-QBER (Simulated)")
    ax1.set_xlabel(r"Basis Selection Bias $\beta = P(\text{Eve chooses } Z)$", fontsize=11)
    ax1.set_ylabel("QBER (%)", fontsize=11)
    ax1.set_title("Basis Bias vs. Sub-Basis Error Distribution", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=8, loc="center right")

    # Plot 2: Basis Distribution and Information Gain
    ax2.plot(b_vals, z_fracs, "c-o", label=r"Eve Z-Basis Selection Rate (%)")
    ax2.plot(b_vals, [r["theoretical_info_gain"] for r in results_table], "g--s", label=r"Theoretical $I_{AE}$ (Bits)")
    ax2.plot(b_vals, [r["observed_info_gain"] for r in results_table], "m:d", label=r"Empirical $I_{AE}$ (Bits)")
    ax2.set_xlabel(r"Basis Selection Bias $\beta$", fontsize=11)
    ax2.set_ylabel("Percentage (%) / Information (Bits)", fontsize=11)
    ax2.set_title("Eve Interception Profile & Information Gain", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "exp4_basis_bias_analysis.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {plot_path}\n")

    return {"data": results_table, "plot_path": plot_path}


def run_experiment_5(engine: BB84Engine, output_dir: str) -> dict:
    print("=" * 70)
    print("EXPERIMENT 5: Fuchs-Peres-Brandt Unitary Probe Evaluation")
    print("=" * 70)

    thetas = [0.0, np.pi / 24, np.pi / 12, np.pi / 8, np.pi / 6, np.pi / 4]
    n_signals = 8000

    results_table = []
    for theta in thetas:
        t0 = time.time()
        probe = FuchsPeresBrandtProbe(theta_E=theta)
        res = engine.run(n_signals=n_signals, attack=probe, seed_simulator=303)
        el = time.time() - t0

        theo_d = probe.theoretical_disturbance()
        theo_i = probe.theoretical_information_gain()

        # State fidelity calculation on carrier states
        fidelities = []
        for state in [[1.0, 0.0], [0.0, 1.0], [1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]]:
            rho_A, _, _ = probe.compute_reduced_states(state)
            pure_dm = np.outer(state, np.conj(state))
            # Fidelity for pure target state |psi>: F = <psi|rho|psi>
            F = float(np.real(np.trace(pure_dm @ rho_A.data)))
            fidelities.append(F)
        avg_fidelity = float(np.mean(fidelities))

        results_table.append({
            "theta_rad": float(theta),
            "theta_deg": float(np.degrees(theta)),
            "observed_qber": float(res.qber),
            "theoretical_disturbance": float(theo_d),
            "qber_z": float(res.qber_z),
            "qber_x": float(res.qber_x),
            "theoretical_info_gain": float(theo_i),
            "average_carrier_fidelity": float(avg_fidelity),
            "runtime_sec": float(el),
        })

        print(f"theta={theta:6.4f} rad ({np.degrees(theta):4.1f} deg) | "
              f"Obs QBER={res.qber * 100:5.2f}% (Theo D={theo_d * 100:5.2f}%) | "
              f"I_AE={theo_i:.4f} bits | Fidelity={avg_fidelity * 100:5.2f}% | Time={el:.2f}s")

    # Generate Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    th_degs = [r["theta_deg"] for r in results_table]
    obs_q = [r["observed_qber"] * 100 for r in results_table]
    theo_d = [r["theoretical_disturbance"] * 100 for r in results_table]
    theo_i = [r["theoretical_info_gain"] for r in results_table]
    fidelities = [r["average_carrier_fidelity"] * 100 for r in results_table]

    # Plot 1: Interaction Angle vs QBER and Fidelity
    ax1.plot(th_degs, theo_d, "r--", linewidth=1.5, label=r"Theory: $D(\theta_E) = 0.5\sin^2\theta_E$")
    ax1.plot(th_degs, obs_q, "ro-", markersize=6, label="Observed QBER (%)")
    ax1.plot(th_degs, fidelities, "b^-", markersize=6, label="State Fidelity (%)")
    ax1.axhline(11.0, color="darkred", linestyle=":", label=r"Abort Threshold ($11\%$)")
    ax1.set_xlabel(r"Coupling Angle $\theta_E$ (Degrees)", fontsize=11)
    ax1.set_ylabel("Percentage (%)", fontsize=11)
    ax1.set_title("Unitary Probe Coupling vs. Disturbance & Fidelity", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    # Plot 2: Information-Disturbance Trade-off
    ax2.plot(obs_q, theo_i, "gs-", markersize=7, linewidth=1.8, label="Simulated (QBER, $I_{AE}$)")
    ax2.set_xlabel(r"Induced QBER / Disturbance (%)", fontsize=11)
    ax2.set_ylabel(r"Eve Accessible Information $I_{AE}$ (Bits)", fontsize=11)
    ax2.set_title("Information-Disturbance Trade-Off Curve", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "exp5_unitary_probe_tradeoff.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {plot_path}\n")

    return {"data": results_table, "plot_path": plot_path}


def main():
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("STARTING COMPLETE MILESTONE 1B EXPERIMENTAL SUITE")
    print(f"Target Results Directory: {os.path.abspath(output_dir)}")
    print("=" * 70)

    engine = BB84Engine(alice_seed=42, bob_seed=999)

    exp1_res = run_experiment_1(engine)
    exp2_res = run_experiment_2(engine)
    exp3_res = run_experiment_3(engine, output_dir)
    exp4_res = run_experiment_4(engine, output_dir)
    exp5_res = run_experiment_5(engine, output_dir)

    all_results = {
        "experiment_1_ideal": exp1_res,
        "experiment_2_full_intercept": exp2_res,
        "experiment_3_partial_intercept": exp3_res,
        "experiment_4_basis_bias": exp4_res,
        "experiment_5_unitary_probe": exp5_res,
    }

    json_path = os.path.join(output_dir, "milestone1b_experimental_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 70)
    print(f"ALL 5 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results JSON: {os.path.abspath(json_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
