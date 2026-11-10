"""Experimental Validation Harness for Milestone 2: NISQ Noise Modelling.

Executes and documents the 5 required research experiments for Milestone 2:
1. Experiment 1: Depolarizing noise sweep (p in [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]).
   Measures: Overall QBER, Z-basis QBER, X-basis QBER, State fidelity.
2. Experiment 2: Amplitude damping sweep (gamma in [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]).
   Measures: Z-basis QBER, X-basis QBER, Error asymmetry, Bit distribution dependence.
3. Experiment 3: Phase damping sweep (lambda in [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]).
   Measures: Z-basis QBER, X-basis QBER, Coherence loss, Average fidelity.
4. Experiment 4: Thermal relaxation (T1/T2 physical decay across physical regimes).
   Measures: Total QBER, Basis-dependent disturbance, State fidelity.
5. Experiment 5: Combined noise and eavesdropping (Ideal, Noise only, Attack only, Noise+Attack).
   Measures: QBER decomposition, Information gain, Detection behavior relative to 11% abort threshold.

Saves high-resolution figures and structured JSON metrics to `results/`.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["MPLCONFIGDIR"] = os.path.abspath("results")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator

from src.bb84.alice import Alice, BASIS_X, BASIS_Z, prepare_bb84_state
from src.bb84.bob import Bob
from src.channel.quantum_channel import QuantumChannel
from src.bb84.attacks import (
    FuchsPeresBrandtProbe,
    InterceptResendAttack,
    binary_entropy,
)
from src.bb84.bb84_engine import BB84Engine
from src.bb84.bob import Bob
from src.channel.noise_models import (
    NoiseChannelProfile,
    NoiseType,
    apply_kraus_channel,
    build_amplitude_damping_noise_model,
    build_combined_noise_model,
    build_depolarizing_noise_model,
    build_phase_damping_noise_model,
    build_thermal_relaxation_noise_model,
    compute_channel_fidelity,
    compute_coherence,
    compute_state_fidelity,
    get_amplitude_damping_kraus,
    get_depolarizing_kraus,
    get_phase_damping_kraus,
    get_thermal_relaxation_kraus,
    predict_amplitude_damping_qber,
    predict_depolarizing_qber,
    predict_phase_damping_qber,
    predict_thermal_relaxation_qber,
)


def run_experiment_1(engine: BB84Engine, output_dir: str) -> Dict[str, Any]:
    """Experiment 1: Depolarizing Noise Sweep.

    p in [0.00, 0.01, 0.03, 0.05, 0.10, 0.20].
    """
    print("=" * 70)
    print("EXPERIMENT 1: Depolarizing Noise Sweep (p in [0.00 .. 0.20])")
    print("=" * 70)

    p_values = [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]
    n_signals = 10000

    results_table = []
    for p in p_values:
        t0 = time.time()
        profile = build_depolarizing_noise_model(p=p)
        res = engine.run(n_signals=n_signals, noise=profile, seed_simulator=101 + int(p * 1000))
        el = time.time() - t0

        kraus = get_depolarizing_kraus(p)
        avg_fidelity = compute_channel_fidelity(kraus)
        theo_q, theo_qz, theo_qx = predict_depolarizing_qber(p)

        row = {
            "p": float(p),
            "observed_qber": float(res.qber),
            "theoretical_qber": float(theo_q),
            "observed_qber_z": float(res.qber_z),
            "theoretical_qber_z": float(theo_qz),
            "observed_qber_x": float(res.qber_x),
            "theoretical_qber_x": float(theo_qx),
            "average_state_fidelity": float(avg_fidelity),
            "theoretical_fidelity": float(1.0 - p / 2.0),
            "sifted_bits": res.sifted_key_length,
            "error_count": res.error_count,
            "runtime_sec": float(el),
        }
        results_table.append(row)

        print(
            f"p={p:4.2f} | Obs QBER={res.qber * 100:5.2f}% (Theo {theo_q * 100:5.2f}%) | "
            f"Z={res.qber_z * 100:5.2f}% | X={res.qber_x * 100:5.2f}% | "
            f"Fidelity={avg_fidelity * 100:5.2f}% | Time={el:.2f}s"
        )

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    p_arr = [r["p"] for r in results_table]
    obs_q = [r["observed_qber"] * 100 for r in results_table]
    theo_q = [r["theoretical_qber"] * 100 for r in results_table]
    obs_qz = [r["observed_qber_z"] * 100 for r in results_table]
    obs_qx = [r["observed_qber_x"] * 100 for r in results_table]
    fid = [r["average_state_fidelity"] * 100 for r in results_table]
    theo_fid = [r["theoretical_fidelity"] * 100 for r in results_table]

    # Plot 1: Depolarizing Rate vs QBER
    ax1.plot(p_arr, theo_q, "r--", linewidth=2, label=r"Theory: $\text{QBER} = p / 2$")
    ax1.plot(p_arr, obs_q, "bo-", markersize=7, label="Observed Total QBER")
    ax1.plot(p_arr, obs_qz, "g^-", markersize=5, alpha=0.8, label="Z-basis QBER")
    ax1.plot(p_arr, obs_qx, "ms-", markersize=5, alpha=0.8, label="X-basis QBER")
    ax1.axhline(11.0, color="darkred", linestyle=":", label=r"Abort Threshold ($11\%$)")
    ax1.set_xlabel("Depolarizing Parameter $p$", fontsize=11)
    ax1.set_ylabel("QBER (%)", fontsize=11)
    ax1.set_title("Depolarizing Noise Rate vs. Error Rates", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    # Plot 2: Depolarizing Rate vs State Fidelity
    ax2.plot(p_arr, theo_fid, "k--", linewidth=2, label=r"Theory: $F = 1 - p / 2$")
    ax2.plot(p_arr, fid, "cd-", markersize=7, label="Average State Fidelity")
    ax2.set_xlabel("Depolarizing Parameter $p$", fontsize=11)
    ax2.set_ylabel("State Fidelity (%)", fontsize=11)
    ax2.set_title("Carrier State Fidelity Degradation", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "exp1_depolarizing_sweep.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {plot_path}\n")

    return {"data": results_table, "plot_path": plot_path}


def run_experiment_2(engine: BB84Engine, output_dir: str) -> Dict[str, Any]:
    """Experiment 2: Amplitude Damping Sweep and Bit Asymmetry.

    gamma in [0.00, 0.01, 0.03, 0.05, 0.10, 0.20].
    """
    print("=" * 70)
    print("EXPERIMENT 2: Amplitude Damping Sweep & Bit Asymmetry Analysis")
    print("=" * 70)

    gamma_values = [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]
    n_signals = 10000

    results_table = []
    bit_bias_results = {}

    for g in gamma_values:
        t0 = time.time()
        profile = build_amplitude_damping_noise_model(gamma=g)
        res = engine.run(n_signals=n_signals, noise=profile, seed_simulator=202 + int(g * 1000))
        el = time.time() - t0

        theo_tot, theo_qz, theo_qx = predict_amplitude_damping_qber(gamma=g, f1=0.5)
        asymmetry = float(res.qber_z - res.qber_x)

        row = {
            "gamma": float(g),
            "observed_qber": float(res.qber),
            "theoretical_qber": float(theo_tot),
            "observed_qber_z": float(res.qber_z),
            "theoretical_qber_z": float(theo_qz),
            "observed_qber_x": float(res.qber_x),
            "theoretical_qber_x": float(theo_qx),
            "error_asymmetry_delta": asymmetry,
            "theoretical_asymmetry": float(theo_qz - theo_qx),
            "sifted_bits": res.sifted_key_length,
            "runtime_sec": float(el),
        }
        results_table.append(row)

        print(
            f"gamma={g:4.2f} | Total={res.qber * 100:5.2f}% | "
            f"Z-QBER={res.qber_z * 100:5.2f}% (Theo {theo_qz * 100:5.2f}%) | "
            f"X-QBER={res.qber_x * 100:5.2f}% (Theo {theo_qx * 100:5.2f}%) | "
            f"Asymmetry={asymmetry * 100:+5.2f}% | Time={el:.2f}s"
        )

    # Investigate dependence on input bit distribution at gamma = 0.10
    test_gamma = 0.10
    bias_conditions = [0.25, 0.50, 0.75]  # fraction of |1> states
    bias_records = []
    print("\n--- Investigating Input Bit Distribution at gamma = 0.10 ---")
    for f1 in bias_conditions:
        # Custom run with biased Alice bits
        n_custom = 8000
        alice_biased = Alice(seed=999)
        # Generate custom biased bits
        raw_bits = (np.random.default_rng(1234).random(n_custom) < f1).astype(np.int8)
        raw_bases = alice_biased.generate_bases(n_custom)
        circuits = [prepare_bb84_state(bit=int(raw_bits[i]), basis=raw_bases[i]) for i in range(n_custom)]

        profile = build_amplitude_damping_noise_model(gamma=test_gamma)
        from qiskit_aer import AerSimulator
        noisy_backend = AerSimulator(noise_model=profile.noise_model)
        bob = Bob(seed=888, backend=noisy_backend)
        bob_bases = bob.generate_bases(n_custom)

        # Transmit through noisy channel
        channel = QuantumChannel(noise_model=profile)
        trans_res = channel.transmit_batch(circuits)
        meas_bits = bob.measure_pulses(
            trans_res.received_circuits,
            bases=bob_bases,
            seed_simulator=555,
            backend=noisy_backend,
        )

        # Sift
        sift_mask = (raw_bases == bob_bases)
        sift_bits_a = raw_bits[sift_mask]
        sift_bits_b = meas_bits[sift_mask]
        sift_bases = raw_bases[sift_mask]

        z_mask = (sift_bases == BASIS_Z)
        obs_qz = float(np.mean(sift_bits_a[z_mask] != sift_bits_b[z_mask]))
        theo_qz = float(f1 * test_gamma)

        bias_rec = {
            "f1_target": float(f1),
            "f1_empirical": float(np.mean(raw_bits)),
            "gamma": float(test_gamma),
            "observed_qber_z": obs_qz,
            "theoretical_qber_z": theo_qz,
        }
        bias_records.append(bias_rec)
        print(f"Fraction |1> (f1) = {f1:4.2f} -> Observed Z-QBER = {obs_qz * 100:5.2f}% (Theoretical: {theo_qz * 100:5.2f}%)")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    g_arr = [r["gamma"] for r in results_table]
    obs_qz = [r["observed_qber_z"] * 100 for r in results_table]
    theo_qz = [r["theoretical_qber_z"] * 100 for r in results_table]
    obs_qx = [r["observed_qber_x"] * 100 for r in results_table]
    theo_qx = [r["theoretical_qber_x"] * 100 for r in results_table]
    asym = [r["error_asymmetry_delta"] * 100 for r in results_table]

    # Plot 1: Basis Asymmetry vs Gamma
    ax1.plot(g_arr, theo_qz, "r--", linewidth=2, label=r"Theory $Q_Z = 0.5 \gamma$")
    ax1.plot(g_arr, obs_qz, "ro-", markersize=6, label=r"Observed $Q_Z$ (Z-basis)")
    ax1.plot(g_arr, theo_qx, "b--", linewidth=2, label=r"Theory $Q_X = (1 - \sqrt{1 - \gamma}) / 2$")
    ax1.plot(g_arr, obs_qx, "bs-", markersize=6, label=r"Observed $Q_X$ (X-basis)")
    ax1.plot(g_arr, asym, "g^:", markersize=6, label=r"Asymmetry $\Delta Q = Q_Z - Q_X$")
    ax1.set_xlabel(r"Amplitude Damping Parameter $\gamma$", fontsize=11)
    ax1.set_ylabel("QBER (%)", fontsize=11)
    ax1.set_title("Amplitude Damping: Conjugate Basis Asymmetry", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    # Plot 2: Bit Bias Dependence
    f_vals = [r["f1_target"] for r in bias_records]
    obs_qz_bias = [r["observed_qber_z"] * 100 for r in bias_records]
    theo_qz_bias = [r["theoretical_qber_z"] * 100 for r in bias_records]
    ax2.plot(f_vals, theo_qz_bias, "k--", linewidth=2, label=r"Theory: $Q_Z = f_1 \cdot \gamma$")
    ax2.plot(f_vals, obs_qz_bias, "md-", markersize=8, label=r"Observed $Q_Z$ ($\gamma = 0.10$)")
    ax2.set_xlabel(r"Fraction of Transmitted $|1\rangle$ States ($f_1$)", fontsize=11)
    ax2.set_ylabel("Z-Basis QBER (%)", fontsize=11)
    ax2.set_title(r"Z-Basis QBER Dependence on State Distribution $f_1$", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "exp2_amplitude_damping_sweep.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {plot_path}\n")

    return {
        "data": results_table,
        "bit_bias_investigation": bias_records,
        "plot_path": plot_path,
    }


def run_experiment_3(engine: BB84Engine, output_dir: str) -> Dict[str, Any]:
    """Experiment 3: Phase Damping Sweep, Coherence Loss, and Basis Selectivity.

    lambda in [0.00, 0.01, 0.03, 0.05, 0.10, 0.20].
    """
    print("=" * 70)
    print("EXPERIMENT 3: Phase Damping Sweep (lambda in [0.00 .. 0.20])")
    print("=" * 70)

    lam_values = [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]
    n_signals = 10000

    results_table = []
    rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)

    for lam in lam_values:
        t0 = time.time()
        profile = build_phase_damping_noise_model(lam=lam)
        res = engine.run(n_signals=n_signals, noise=profile, seed_simulator=303 + int(lam * 1000))
        el = time.time() - t0

        kraus = get_phase_damping_kraus(lam)
        rho_out = apply_kraus_channel(rho_plus, kraus)
        coh_l1 = compute_coherence(rho_out)
        avg_fidelity = compute_channel_fidelity(kraus)
        theo_tot, theo_qz, theo_qx = predict_phase_damping_qber(lam=lam)

        row = {
            "lambda": float(lam),
            "observed_qber": float(res.qber),
            "theoretical_qber": float(theo_tot),
            "observed_qber_z": float(res.qber_z),
            "theoretical_qber_z": 0.0,
            "observed_qber_x": float(res.qber_x),
            "theoretical_qber_x": float(theo_qx),
            "coherence_l1": float(coh_l1),
            "coherence_loss": float(1.0 - coh_l1),
            "average_state_fidelity": float(avg_fidelity),
            "sifted_bits": res.sifted_key_length,
            "runtime_sec": float(el),
        }
        results_table.append(row)

        print(
            f"lambda={lam:4.2f} | Total={res.qber * 100:5.2f}% | "
            f"Z-QBER={res.qber_z * 100:5.2f}% (Theo 0.00%) | "
            f"X-QBER={res.qber_x * 100:5.2f}% (Theo {theo_qx * 100:5.2f}%) | "
            f"Coherence={coh_l1:6.4f} | Fidelity={avg_fidelity * 100:5.2f}% | Time={el:.2f}s"
        )

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    lam_arr = [r["lambda"] for r in results_table]
    obs_qz = [r["observed_qber_z"] * 100 for r in results_table]
    obs_qx = [r["observed_qber_x"] * 100 for r in results_table]
    theo_qx = [r["theoretical_qber_x"] * 100 for r in results_table]
    coh = [r["coherence_l1"] for r in results_table]
    fid = [r["average_state_fidelity"] * 100 for r in results_table]

    # Plot 1: Basis Selectivity
    ax1.plot(lam_arr, [0.0] * len(lam_arr), "k--", linewidth=1.5, label=r"Theory $Q_Z = 0.0\%$ (Strict Conservation)")
    ax1.plot(lam_arr, obs_qz, "ko", markersize=6, label="Observed $Q_Z$")
    ax1.plot(lam_arr, theo_qx, "r--", linewidth=2, label=r"Theory $Q_X = (1 - \sqrt{1 - \lambda}) / 2$")
    ax1.plot(lam_arr, obs_qx, "rs-", markersize=6, label="Observed $Q_X$")
    ax1.set_xlabel(r"Phase Damping Parameter $\lambda$", fontsize=11)
    ax1.set_ylabel("QBER (%)", fontsize=11)
    ax1.set_title("Phase Damping: Complete Computational Basis Immunity", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    # Plot 2: Coherence Loss & State Fidelity
    ax2.plot(lam_arr, coh, "b-o", linewidth=1.8, label=r"Off-diagonal Coherence $2|\rho_{01}|$")
    ax2.plot(lam_arr, [f / 100.0 for f in fid], "g--s", linewidth=1.8, label=r"Average State Fidelity $\bar{F}$")
    ax2.set_xlabel(r"Phase Damping Parameter $\lambda$", fontsize=11)
    ax2.set_ylabel("Normalized Measure in [0, 1]", fontsize=11)
    ax2.set_title("Carrier Coherence Decay and State Fidelity", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "exp3_phase_damping_sweep.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {plot_path}\n")

    return {"data": results_table, "plot_path": plot_path}


def run_experiment_4(engine: BB84Engine, output_dir: str) -> Dict[str, Any]:
    """Experiment 4: Physical Thermal Relaxation (T1/T2 Decay Models).

    Evaluates physically valid regimes:
    Regime A: Pure energy relaxation (T2 = 2*T1).
    Regime B: Balanced relaxation (T2 = T1).
    Regime C: Dephasing dominated (T2 < T1).
    """
    print("=" * 70)
    print("EXPERIMENT 4: Thermal Relaxation Physical Decay Across Regimes")
    print("=" * 70)

    # Transit times in microseconds
    t_microseconds = [0.5, 1.0, 2.0, 5.0, 10.0]
    t1_fixed = 50.0e-6  # 50 microseconds

    regimes = [
        {"name": "Regime A (T2 = 2*T1, Limit of pure relaxation)", "t2": 100.0e-6},
        {"name": "Regime B (T2 = T1, Balanced relaxation)", "t2": 50.0e-6},
        {"name": "Regime C (T2 = 0.5*T1, Dephasing dominated)", "t2": 25.0e-6},
    ]

    all_regime_data = []
    n_signals = 8000

    for reg in regimes:
        reg_name = reg["name"]
        t2_val = reg["t2"]
        print(f"\n--- {reg_name} (T1 = 50 us, T2 = {t2_val * 1e6:.1f} us) ---")

        curve_points = []
        for t_us in t_microseconds:
            transit_time = t_us * 1e-6
            t0 = time.time()
            profile = build_thermal_relaxation_noise_model(
                t1=t1_fixed,
                t2=t2_val,
                time=transit_time,
                p_excited=0.0,
            )
            res = engine.run(n_signals=n_signals, noise=profile, seed_simulator=404 + int(t_us * 10))
            el = time.time() - t0

            theo_tot, theo_qz, theo_qx = predict_thermal_relaxation_qber(
                t1=t1_fixed,
                t2=t2_val,
                time=transit_time,
            )
            kraus = get_thermal_relaxation_kraus(t1=t1_fixed, t2=t2_val, time=transit_time)
            avg_fid = compute_channel_fidelity(kraus)

            point = {
                "transit_time_us": float(t_us),
                "transit_time_s": float(transit_time),
                "t1_s": float(t1_fixed),
                "t2_s": float(t2_val),
                "observed_qber": float(res.qber),
                "theoretical_qber": float(theo_tot),
                "observed_qber_z": float(res.qber_z),
                "theoretical_qber_z": float(theo_qz),
                "observed_qber_x": float(res.qber_x),
                "theoretical_qber_x": float(theo_qx),
                "average_fidelity": float(avg_fid),
                "runtime_sec": float(el),
            }
            curve_points.append(point)

            print(
                f"t={t_us:4.1f} us | QBER={res.qber * 100:5.2f}% (Theo {theo_tot * 100:5.2f}%) | "
                f"Z={res.qber_z * 100:5.2f}% | X={res.qber_x * 100:5.2f}% | "
                f"Fid={avg_fid * 100:5.2f}%"
            )

        all_regime_data.append({"regime": reg_name, "points": curve_points})

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    styles = [("b", "o"), ("g", "s"), ("r", "^")]
    for idx, reg_entry in enumerate(all_regime_data):
        color, marker = styles[idx]
        reg_lbl = reg_entry["regime"].split("(")[0].strip()
        t_vals = [p["transit_time_us"] for p in reg_entry["points"]]
        q_tot = [p["observed_qber"] * 100 for p in reg_entry["points"]]
        q_theo = [p["theoretical_qber"] * 100 for p in reg_entry["points"]]
        fid = [p["average_fidelity"] * 100 for p in reg_entry["points"]]

        ax1.plot(t_vals, q_theo, color=color, linestyle="--", alpha=0.7)
        ax1.plot(t_vals, q_tot, color=color, marker=marker, linewidth=1.8, label=f"{reg_lbl} (Observed)")

        ax2.plot(t_vals, fid, color=color, marker=marker, linewidth=1.8, label=f"{reg_lbl}")

    ax1.axhline(11.0, color="darkred", linestyle=":", label=r"Abort Threshold ($11\%$)")
    ax1.set_xlabel(r"Transit Time $t$ ($\mu$s)", fontsize=11)
    ax1.set_ylabel("Total QBER (%)", fontsize=11)
    ax1.set_title("Thermal Relaxation: QBER across T1/T2 Regimes", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=8)

    ax2.set_xlabel(r"Transit Time $t$ ($\mu$s)", fontsize=11)
    ax2.set_ylabel("Average Carrier State Fidelity (%)", fontsize=11)
    ax2.set_title("Thermal Relaxation: State Fidelity Decay", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "exp4_thermal_relaxation.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {plot_path}\n")

    return {"regimes": all_regime_data, "plot_path": plot_path}


def run_experiment_5(engine: BB84Engine, output_dir: str) -> Dict[str, Any]:
    """Experiment 5: Combined Noise and Eavesdropping Interaction.

    Compares:
    - Condition A: Ideal channel (no noise, no attack).
    - Condition B: Noise only (depolarizing p=0.04 -> QBER ~ 2.0%).
    - Condition C1: Intercept-resend only (alpha=0.40 -> QBER ~ 10.0%).
    - Condition C2: Unitary probe only (theta=pi/10 -> QBER = sin^2(pi/10) ~ 9.55%).
    - Condition D1: Noise (p=0.04) + Intercept-resend (alpha=0.40) -> compounds to 11.60%.
    - Condition D2: Noise (p=0.04) + Unitary probe (theta=pi/10) -> compounds to 11.17%.

    Demonstrates detection threshold crossing (11.0% abort limit).
    """
    print("=" * 70)
    print("EXPERIMENT 5: Noise and Eavesdropping Interaction & Compounding")
    print("=" * 70)

    n_signals = 12000
    p_noise = 0.04
    noise_profile = build_depolarizing_noise_model(p=p_noise)
    ir_attack = InterceptResendAttack(alpha=0.40, beta=0.5, seed=123)
    probe_angle = float(np.pi / 10.0)
    probe_attack = FuchsPeresBrandtProbe(theta_E=probe_angle)

    conditions = [
        {"name": "Condition A: Ideal Channel", "noise": None, "attack": None},
        {"name": "Condition B: Noise Only (p=0.04)", "noise": noise_profile, "attack": None},
        {"name": "Condition C1: Intercept-Resend Only (alpha=0.40)", "noise": None, "attack": ir_attack},
        {"name": "Condition C2: Unitary Probe Only (theta=pi/10)", "noise": None, "attack": probe_attack},
        {"name": "Condition D1: Noise + Intercept-Resend", "noise": noise_profile, "attack": ir_attack},
        {"name": "Condition D2: Noise + Unitary Probe", "noise": noise_profile, "attack": probe_attack},
    ]

    records = []
    for cond in conditions:
        t0 = time.time()
        res = engine.run(
            n_signals=n_signals,
            attack=cond["attack"],
            noise=cond["noise"],
            seed_simulator=505,
        )
        el = time.time() - t0

        aborted = bool(res.qber >= 0.11)
        rec = {
            "condition": cond["name"],
            "observed_qber": float(res.qber),
            "observed_qber_z": float(res.qber_z),
            "observed_qber_x": float(res.qber_x),
            "eve_info_gain": float(res.eve_information_gain),
            "aborted": aborted,
            "runtime_sec": float(el),
        }
        records.append(rec)

        status_str = "ABORT PROTOCOL (> 11%)" if aborted else "SECURE KEY EXTRACTED (< 11%)"
        print(
            f"{cond['name']:<48} | QBER={res.qber * 100:5.2f}% | "
            f"I_AE={res.eve_information_gain:.4f} bits | {status_str}"
        )

    # Theoretical compounding analysis
    q_env = p_noise / 2.0  # 0.02
    q_ir = 0.40 / 4.0      # 0.10
    q_probe = float(np.sin(probe_angle) ** 2)  # 0.09549

    theo_comp_ir = q_env + q_ir - 2.0 * q_env * q_ir
    theo_comp_probe = q_env + q_probe - 2.0 * q_env * q_probe

    print("\n--- Theoretical Compounding Validation ---")
    print(f"Theory Noise + Intercept-Resend : {theo_comp_ir * 100:.2f}% | Observed: {records[4]['observed_qber'] * 100:.2f}%")
    print(f"Theory Noise + Unitary Probe    : {theo_comp_probe * 100:.2f}% | Observed: {records[5]['observed_qber'] * 100:.2f}%")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = ["Ideal", "Noise Only", "PIR Only", "Probe Only", "Noise+PIR", "Noise+Probe"]
    qbers = [r["observed_qber"] * 100 for r in records]
    infos = [r["eve_info_gain"] for r in records]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c", "#c0392b"]

    bars1 = ax1.bar(names, qbers, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.axhline(11.0, color="darkred", linestyle="--", linewidth=2, label=r"QKD Abort Threshold ($11\%$)")
    ax1.set_ylabel("Observed QBER (%)", fontsize=11)
    ax1.set_title("Compounded QBER across Noise & Eavesdropping", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 16)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # Annotate bar values
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Plot 2: Eve Information Gain vs Abort State
    bars2 = ax2.bar(names, infos, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.set_ylabel(r"Eve Information Gain $I_{AE}$ (Bits / Pulse)", fontsize=11)
    ax2.set_title("Eavesdropper Intercepted Information", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 0.4)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "exp5_noise_vs_eavesdropping.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved to: {plot_path}\n")

    return {
        "conditions": records,
        "theoretical_compounding": {
            "noise_plus_intercept_resend": float(theo_comp_ir),
            "noise_plus_unitary_probe": float(theo_comp_probe),
        },
        "plot_path": plot_path,
    }


def main():
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("STARTING COMPLETE MILESTONE 2 EXPERIMENTAL SUITE")
    print(f"Target Results Directory: {os.path.abspath(output_dir)}")
    print("=" * 70)

    engine = BB84Engine(alice_seed=42, bob_seed=999)

    exp1_res = run_experiment_1(engine, output_dir)
    exp2_res = run_experiment_2(engine, output_dir)
    exp3_res = run_experiment_3(engine, output_dir)
    exp4_res = run_experiment_4(engine, output_dir)
    exp5_res = run_experiment_5(engine, output_dir)

    all_results = {
        "experiment_1_depolarizing_sweep": exp1_res,
        "experiment_2_amplitude_damping_sweep": exp2_res,
        "experiment_3_phase_damping_sweep": exp3_res,
        "experiment_4_thermal_relaxation": exp4_res,
        "experiment_5_noise_and_eavesdropping": exp5_res,
    }

    json_path = os.path.join(output_dir, "milestone2_experimental_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 70)
    print("ALL 5 MILESTONE 2 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results JSON: {os.path.abspath(json_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
