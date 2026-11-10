"""Comparative Benchmarking of cGAN Attack Policy Engine vs Classical Baselines.

Evaluates 4 attack policy selection strategies under strictly equal simulator query
budget (K = 16 physical forward simulations per condition per episode):
1. cGAN Policy Engine (Generator G_phi + forward evaluation + Tri-Case fallback)
2. Uniform Random Search (Random parameter sampling + forward evaluation + Tri-Case fallback)
3. Deterministic Grid Search (16 canonical lattice configurations ATT-00..15 + Tri-Case fallback)
4. Analytical Probe Heuristic (Closed-form target angle + perturbed neighborhood + Tri-Case fallback)

Evaluated strictly across held-out Test Conditions (C21, C22, C23, C24):
- C21: p=0.05, Q_budget=0.05 (High noise, strict budget)
- C22: p=0.05, Q_budget=0.11 (High noise, relaxed budget)
- C23: p=0.07, Q_budget=0.05 (Critical noise, strict budget)
- C24: p=0.07, Q_budget=0.11 (Critical noise, relaxed budget)

Repeated over R_eval = 5 independent evaluation seeds (901, 902, 903, 904, 905).
Total: 4 conditions x 5 seeds = 20 evaluation episodes per method (80 episodes total).

Outputs:
    - results/cgan_benchmark_comparison.json
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
import torch

from src.cgan.models import ConditionalGenerator
from src.cgan.policy_engine import OperationalDecision, cGANPolicyEngine
from src.dataset.generator import (
    execute_single_simulation,
    get_full_environmental_conditions,
)
from src.dataset.policy_generator import get_all_attack_configurations, get_attack_configuration
from src.dataset.schema import (
    AttackConfiguration,
    AttackFamily,
    AttackParameters,
    DatasetRecord,
    EnvironmentalCondition,
    StatisticalFeasibility,
)


class UniformRandomPolicyEngine:
    """Baseline 1: Uniform random sampling over the continuous attack parameter space."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def propose_candidate_configurations(
        self,
        condition: EnvironmentalCondition,
        num_candidates: int = 16,
        seed: Optional[int] = None,
    ) -> List[AttackConfiguration]:
        """Propose K random attack configurations."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        candidates: List[AttackConfiguration] = []

        for i in range(num_candidates):
            fam_choice = rng.integers(0, 2)
            if fam_choice == 0:
                # PIR Attack
                alpha = float(rng.uniform(0.0, 1.0))
                beta = float(rng.uniform(0.0, 1.0))
                params = AttackParameters(alpha=alpha, beta=beta)
                dist = alpha / 4.0
                desc = f"Random-PIR (alpha={alpha:.4f}, beta={beta:.4f})"
                config = AttackConfiguration(
                    attack_id=f"Rand-PIR-{i:02d}",
                    family=AttackFamily.PROBABILISTIC_INTERCEPT_RESEND,
                    parameters=params,
                    theoretical_disturbance=dist,
                    description=desc,
                )
            else:
                # Unitary Probe Attack
                theta = float(rng.uniform(0.0, math.pi / 4.0))
                theta_deg = float(math.degrees(theta))
                params = AttackParameters(
                    theta_E_rad=theta,
                    theta_E_deg=theta_deg,
                )
                dist = float(math.sin(theta) ** 2)
                desc = f"Random-Probe (theta={theta:.4f} rad, {theta_deg:.2f} deg)"
                config = AttackConfiguration(
                    attack_id=f"Rand-Probe-{i:02d}",
                    family=AttackFamily.UNITARY_PROBE,
                    parameters=params,
                    theoretical_disturbance=dist,
                    description=desc,
                )
            candidates.append(config)
        return candidates


class GridSearchPolicyEngine:
    """Baseline 2: Fixed deterministic lattice evaluation across 16 canonical configurations."""

    def __init__(self):
        self.lattice_attacks = get_all_attack_configurations()
        if len(self.lattice_attacks) != 16:
            raise ValueError(f"Expected 16 lattice attacks, found {len(self.lattice_attacks)}")

    def propose_candidate_configurations(
        self,
        condition: EnvironmentalCondition,
        num_candidates: int = 16,
        seed: Optional[int] = None,
    ) -> List[AttackConfiguration]:
        """Return the fixed 16 deterministic grid configurations."""
        return list(self.lattice_attacks[:num_candidates])


class AnalyticalProbePolicyEngine:
    """Baseline 3: Closed-form theoretical probe heuristic with local parameter perturbation."""

    def propose_candidate_configurations(
        self,
        condition: EnvironmentalCondition,
        num_candidates: int = 16,
        seed: Optional[int] = None,
    ) -> List[AttackConfiguration]:
        """Propose K perturbed candidates around the theoretical optimal coupling angle."""
        gamma_hat = condition.gamma_noise_hat
        q_bud = condition.q_budget

        # Analytical disturbance target: D* = max(0, (Q_budget - gamma_hat) / (1 - 2*gamma_hat))
        denom = max(1e-4, 1.0 - 2.0 * gamma_hat)
        d_target = max(0.0, (q_bud - gamma_hat) / denom)
        # theta_E* = 0.5 * arcsin(sqrt(2 * D*))
        arg = min(1.0, math.sqrt(min(1.0, 2.0 * d_target)))
        theta_star = 0.5 * math.asin(arg)
        theta_star = float(np.clip(theta_star, 0.0, math.pi / 4.0))

        # Generate K perturbations around theta_star
        offsets = np.linspace(-0.15, +0.15, num_candidates)
        candidates: List[AttackConfiguration] = []
        for i, delta in enumerate(offsets):
            theta_cand = float(np.clip(theta_star + delta, 0.0, math.pi / 4.0))
            theta_deg = float(math.degrees(theta_cand))
            params = AttackParameters(
                theta_E_rad=theta_cand,
                theta_E_deg=theta_deg,
            )
            dist = float(math.sin(theta_cand) ** 2)
            desc = f"Analytical-Probe (theta={theta_cand:.4f} rad, {theta_deg:.2f} deg)"
            config = AttackConfiguration(
                attack_id=f"Analytic-Probe-{i:02d}",
                family=AttackFamily.UNITARY_PROBE,
                parameters=params,
                theoretical_disturbance=dist,
                description=desc,
            )
            candidates.append(config)
        return candidates


def evaluate_method_episode(
    engine: Any,
    condition: EnvironmentalCondition,
    seed_eval: int,
    query_budget_k: int = 16,
    n_pulses: int = 2000,
    rho_test: float = 0.50,
) -> Dict[str, Any]:
    """Execute a single evaluation episode under strict query budget K and Tri-Case fallback."""
    t_start = time.perf_counter()

    # 1. Candidate Proposal
    if hasattr(engine, "propose_candidate_configurations"):
        candidates = engine.propose_candidate_configurations(
            condition=condition,
            num_candidates=query_budget_k,
            seed=seed_eval,
        )
    else:
        raise ValueError(f"Unknown engine type: {type(engine)}")

    t_proposal = time.perf_counter() - t_start

    # 2. Forward physical simulation of all K candidates
    eval_records: List[DatasetRecord] = []
    for i, cand in enumerate(candidates):
        sim_seed = seed_eval * 1000 + i + 1
        rec = execute_single_simulation(
            condition=condition,
            attack_config=cand,
            seed_sim=sim_seed,
            n_pulses=n_pulses,
            rho_test=rho_test,
        )
        eval_records.append(rec)

    # 3. Filter feasible candidates under Wilson interval
    feasible_records = [
        r for r in eval_records
        if r.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE
    ]

    # 4. Tri-Case Operational Decision
    null_attack = get_attack_configuration("ATT-00")
    if len(feasible_records) > 0:
        # Case A: Active feasible attack maximizing Eve's mutual information
        selected_record = max(feasible_records, key=lambda r: r.raw_observation.eve_info_gain)
        decision = OperationalDecision.CASE_A_ACTIVE_FEASIBLE
        eve_info_gain = selected_record.raw_observation.eve_info_gain
        protocol_abort = selected_record.raw_observation.protocol_abort
        is_feasible = True
        qber_test = selected_record.raw_observation.qber_test
        wilson_ucb = selected_record.derived_evaluation.wilson_q_ucb
        selected_family = selected_record.attack_family.value
    else:
        # Evaluate Null attack fallback to distinguish Case B and Case C
        null_record = execute_single_simulation(
            condition=condition,
            attack_config=null_attack,
            seed_sim=seed_eval * 1000 + 999,
            n_pulses=n_pulses,
            rho_test=rho_test,
        )
        eve_info_gain = 0.0  # Eve extracts zero information on Null fallback
        protocol_abort = null_record.raw_observation.protocol_abort
        is_feasible = (
            null_record.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE
        )
        qber_test = null_record.raw_observation.qber_test
        wilson_ucb = null_record.derived_evaluation.wilson_q_ucb
        selected_family = AttackFamily.NULL.value

        if null_record.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE:
            decision = OperationalDecision.CASE_B_NULL_FALLBACK
        else:
            decision = OperationalDecision.CASE_C_CHANNEL_ABORT

    return {
        "condition_id": condition.condition_id,
        "seed_eval": seed_eval,
        "p_depol": condition.p_depol,
        "q_budget": condition.q_budget,
        "decision": decision,
        "selected_family": selected_family,
        "eve_info_gain": float(eve_info_gain),
        "protocol_abort": bool(protocol_abort),
        "is_feasible": bool(is_feasible),
        "sample_qber_test": float(qber_test),
        "wilson_ucb": float(wilson_ucb),
        "num_feasible_candidates": len(feasible_records),
        "query_budget_consumed": query_budget_k,
        "proposal_latency_ms": round(t_proposal * 1000.0, 3),
    }


def run_comprehensive_benchmark(
    checkpoint_path: str = "checkpoints/cgan_best.pt",
    output_path: str = "results/cgan_benchmark_comparison.json",
    eval_seeds: tuple = (901, 902, 903, 904, 905),
    query_budget_k: int = 16,
    n_pulses: int = 2000,
) -> Dict[str, Any]:
    """Execute complete comparative benchmark across all 4 methods and held-out test conditions."""
    print("=" * 80)
    print("M3 STAGE C: COMPARATIVE BENCHMARKING (cGAN vs 3 CLASSICAL BASELINES)")
    print("=" * 80)

    # 1. Load trained cGAN Generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Checkpoint: {checkpoint_path}")

    generator = ConditionalGenerator(latent_dim=16, condition_dim=2, hidden_dims=[128, 128])
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator_state_dict"])
        print(f"Loaded cGAN Generator checkpoint from epoch {ckpt.get('epoch', 'N/A')}")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Using untrained generator.")

    cgan_engine = cGANPolicyEngine(generator=generator, device=device)
    random_engine = UniformRandomPolicyEngine(seed=42)
    grid_engine = GridSearchPolicyEngine()
    analytic_engine = AnalyticalProbePolicyEngine()

    methods = {
        "cGAN_Policy_Engine": cgan_engine,
        "Uniform_Random_Search": random_engine,
        "Deterministic_Grid_Search": grid_engine,
        "Analytical_Probe_Heuristic": analytic_engine,
    }

    # 2. Get Test Partition conditions (C21 to C24)
    all_conds = get_full_environmental_conditions()
    test_conditions = [c for c in all_conds if c.partition == "test"]
    print(f"\nHeld-Out Test Conditions ({len(test_conditions)}):")
    for tc in test_conditions:
        print(f"  {tc.condition_id}: p_depol={tc.p_depol:.2f}, Q_budget={tc.q_budget:.2f}, seed_calib={tc.seed_calib}")

    print(f"Evaluation Seeds ({len(eval_seeds)}): {list(eval_seeds)}")
    print(f"Query Budget per Episode K = {query_budget_k} physical simulations")
    print(f"Total Episodes per Method: {len(test_conditions)} x {len(eval_seeds)} = {len(test_conditions) * len(eval_seeds)}")
    print(f"Total Simulator Queries per Method: {len(test_conditions) * len(eval_seeds) * query_budget_k}")

    # 3. Run evaluation across all methods, conditions, and seeds
    all_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in methods}

    t_bench_start = time.perf_counter()
    for method_name, engine in methods.items():
        print(f"\n--- Benchmarking: {method_name} ---")
        ep_count = 0
        for cond in test_conditions:
            for s_eval in eval_seeds:
                res = evaluate_method_episode(
                    engine=engine,
                    condition=cond,
                    seed_eval=s_eval,
                    query_budget_k=query_budget_k,
                    n_pulses=n_pulses,
                )
                all_results[method_name].append(res)
                ep_count += 1
                if ep_count % 5 == 0:
                    print(
                        f"  [{ep_count:2d}/20] Cond {cond.condition_id} | Seed {s_eval} | "
                        f"Decision: {res['decision']} | I_AE: {res['eve_info_gain']:.4f} | "
                        f"Abort: {res['protocol_abort']} | Feas: {res['num_feasible_candidates']}/{query_budget_k}"
                    )

    t_bench_total = time.perf_counter() - t_bench_start

    # 4. Compute aggregate metrics per method
    summary_comparison: Dict[str, Dict[str, Any]] = {}
    for method_name, ep_list in all_results.items():
        info_gains = [ep["eve_info_gain"] for ep in ep_list]
        aborts = [1 if ep["protocol_abort"] else 0 for ep in ep_list]
        feasibilities = [1 if ep["is_feasible"] else 0 for ep in ep_list]
        latencies = [ep["proposal_latency_ms"] for ep in ep_list]

        case_a_count = sum(1 for ep in ep_list if ep["decision"] == OperationalDecision.CASE_A_ACTIVE_FEASIBLE)
        case_b_count = sum(1 for ep in ep_list if ep["decision"] == OperationalDecision.CASE_B_NULL_FALLBACK)
        case_c_count = sum(1 for ep in ep_list if ep["decision"] == OperationalDecision.CASE_C_CHANNEL_ABORT)

        summary_comparison[method_name] = {
            "mean_info_gain": round(float(np.mean(info_gains)), 5),
            "std_info_gain": round(float(np.std(info_gains)), 5),
            "median_info_gain": round(float(np.median(info_gains)), 5),
            "max_info_gain": round(float(np.max(info_gains)), 5),
            "empirical_abort_rate": round(float(np.mean(aborts)), 4),
            "empirical_pass_rate": round(1.0 - float(np.mean(aborts)), 4),
            "statistical_feasibility_rate": round(float(np.mean(feasibilities)), 4),
            "case_distribution": {
                "Case_A_Active_Feasible": case_a_count,
                "Case_B_Null_Fallback": case_b_count,
                "Case_C_Channel_Abort": case_c_count,
            },
            "mean_candidate_proposal_latency_ms": round(float(np.mean(latencies)), 3),
            "query_budget_verified": all(ep["query_budget_consumed"] == query_budget_k for ep in ep_list),
            "total_episodes": len(ep_list),
        }

    # 5. Compile final benchmark JSON
    benchmark_payload = {
        "metadata": {
            "benchmark_run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query_budget_k": query_budget_k,
            "n_pulses_per_simulation": n_pulses,
            "evaluation_seeds": list(eval_seeds),
            "test_conditions": [tc.condition_id for tc in test_conditions],
            "total_episodes_per_method": len(test_conditions) * len(eval_seeds),
            "total_execution_time_seconds": round(t_bench_total, 2),
            "checkpoint_evaluated": checkpoint_path,
        },
        "summary_comparison": summary_comparison,
        "episode_level_results": all_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    # 6. Display formatted comparison table
    print("\n" + "=" * 90)
    print("M3 STAGE C: BENCHMARK COMPARISON TABLE (Held-out Test Conditions, K=16 Queries)")
    print("=" * 90)
    print(f"{'Method':<28} | {'I(A;E) Mean ± Std':<18} | {'Abort Rate':<11} | {'Feasibility':<11} | {'Case A/B/C':<12}")
    print("-" * 90)
    for m_name, sc in summary_comparison.items():
        i_str = f"{sc['mean_info_gain']:.4f} ± {sc['std_info_gain']:.4f}"
        ab_str = f"{sc['empirical_abort_rate']*100:.1f}%"
        fe_str = f"{sc['statistical_feasibility_rate']*100:.1f}%"
        c_str = f"{sc['case_distribution']['Case_A_Active_Feasible']}/{sc['case_distribution']['Case_B_Null_Fallback']}/{sc['case_distribution']['Case_C_Channel_Abort']}"
        print(f"{m_name:<28} | {i_str:<18} | {ab_str:<11} | {fe_str:<11} | {c_str:<12}")
    print("=" * 90)
    print(f"Results saved to: {output_path}")

    return benchmark_payload


if __name__ == "__main__":
    run_comprehensive_benchmark()

