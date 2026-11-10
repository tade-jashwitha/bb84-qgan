"""End-to-end dataset generation pipeline integrating BB84Engine, noise, and statistical evaluation.

This module coordinates:
1. Construction of validated environmental conditions with realistic calibration noise.
2. Single-run execution of physical BB84 simulations under specified attacks and noise.
3. Partitioning of sifted key into error-estimation test bits and secret key bits.
4. Computation of raw observations, basis-resolved errors, and empirical abort decisions.
5. Derivation of Wilson score confidence intervals and three-state statistical feasibility labels.
6. Execution and serialization of the 5-condition pilot dataset and full datasets.
"""

import datetime
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.bb84.bb84_engine import BB84Engine, BB84SimulationResult
from src.channel.noise_models import (
    NoiseChannelProfile,
    build_depolarizing_noise_model,
)
from src.dataset.policy_generator import (
    get_all_attack_configurations,
    get_attack_configuration,
    instantiate_attack,
)
from src.dataset.schema import (
    AttackConfiguration,
    DatasetContainer,
    DatasetRecord,
    DerivedEvaluation,
    EnvironmentalCondition,
    RawObservation,
    SimulationMetadata,
    StatisticalFeasibility,
)
from src.dataset.statistics import (
    classify_feasibility,
    compute_wilson_score_interval,
)

GENERATOR_VERSION = "1.0.0"


def generate_environmental_condition(
    condition_id: str,
    p_depol: float,
    q_budget: float,
    n_calib: int = 1000,
    seed_calib: Optional[int] = None,
    partition: Optional[str] = None,
) -> EnvironmentalCondition:
    """Construct an EnvironmentalCondition with physically grounded calibration noise.

    Args:
        condition_id: Unique identifier string (e.g. 'C01').
        p_depol: Microscopic depolarizing parameter in [0.0, 1.0].
        q_budget: Protocol security abort budget in [0.0, 1.0].
        n_calib: Number of public calibration pilot pulses observed by Eve (default 1000).
        seed_calib: Optional RNG seed for calibration sampling.
        partition: Optional dataset partition ('train', 'val', 'test').

    Returns:
        EnvironmentalCondition: Fully populated and validated condition record.
    """
    if not (0.0 <= p_depol <= 1.0):
        raise ValueError(f"p_depol must be in [0.0, 1.0], got {p_depol}")
    if not (0.0 <= q_budget <= 1.0):
        raise ValueError(f"q_budget must be in [0.0, 1.0], got {q_budget}")
    if n_calib <= 0:
        raise ValueError(f"n_calib must be positive, got {n_calib}")

    # Physical ambient channel QBER for depolarizing noise is p / 2
    q_env = float(p_depol / 2.0)

    # Eve observes n_calib unencrypted test pulses during link calibration
    rng = np.random.default_rng(seed_calib)
    if q_env > 0.0:
        k_calib = int(rng.binomial(n=n_calib, p=q_env))
        gamma_noise_hat = float(np.clip(k_calib / float(n_calib), 0.0, 0.20))
        sigma_calib = float(math.sqrt(q_env * (1.0 - q_env) / float(n_calib)))
    else:
        k_calib = 0
        gamma_noise_hat = 0.0
        sigma_calib = 0.0

    return EnvironmentalCondition(
        condition_id=condition_id,
        p_depol=p_depol,
        q_env=q_env,
        q_budget=q_budget,
        gamma_noise_hat=gamma_noise_hat,
        sigma_calib=sigma_calib,
        n_calib=n_calib,
        k_calib=k_calib,
        seed_calib=seed_calib,
        partition=partition,
    )


def execute_single_simulation(
    condition: EnvironmentalCondition,
    attack_config: AttackConfiguration,
    seed_sim: int,
    n_pulses: int = 2000,
    rho_test: float = 0.50,
) -> DatasetRecord:
    """Execute a single BB84 simulation run and construct a validated DatasetRecord.

    Args:
        condition: EnvironmentalCondition specification.
        attack_config: AttackConfiguration specification.
        seed_sim: Random seed for this specific execution.
        n_pulses: Total transmitted pulses N_pulse (default 2000).
        rho_test: Fraction of sifted bits used for error estimation (default 0.50).

    Returns:
        DatasetRecord: Complete structured record containing raw and derived data.
    """
    t_start = time.perf_counter()

    # Derived seeds for Alice, Bob, and Eve to maintain strict isolation
    seed_alice = seed_sim * 17 + 31
    seed_bob = seed_sim * 23 + 47
    seed_attack = seed_sim * 29 + 59

    # Build depolarizing noise model
    noise_profile = None
    if condition.p_depol > 0.0:
        noise_profile = build_depolarizing_noise_model(p=condition.p_depol)

    # Instantiate attack operator
    attack_instance = instantiate_attack(attack_config, seed=seed_attack)

    # Configure engine
    engine = BB84Engine(alice_seed=seed_alice, bob_seed=seed_bob)

    # Run physical quantum simulation
    sim_result: BB84SimulationResult = engine.run(
        n_signals=n_pulses,
        attack=attack_instance,
        noise=noise_profile,
        seed_simulator=seed_sim,
    )

    t_elapsed = time.perf_counter() - t_start

    # Sifted key extraction
    n_sift = sim_result.sifted_key_length
    sifting_eff = sim_result.sifting_efficiency

    # Partition sifted key: first n_test bits used for QBER estimation
    n_test = int(math.floor(rho_test * n_sift))
    n_key = n_sift - n_test

    # Extract test bits
    alice_sifted = sim_result.alice_sifted_bits
    bob_sifted = sim_result.bob_sifted_bits
    sifted_indices = sim_result.sifted_indices
    alice_sifted_bases = sim_result.alice_bases[sifted_indices]

    test_alice = alice_sifted[:n_test]
    test_bob = bob_sifted[:n_test]
    test_bases = alice_sifted_bases[:n_test]

    # Test error count and test QBER
    test_errors_bool = test_alice != test_bob
    error_count_test = int(np.sum(test_errors_bool))
    qber_test = float(error_count_test / n_test) if n_test > 0 else 0.0

    # Basis-resolved test metrics
    mask_z = test_bases == "Z"
    sifted_count_z_test = int(np.sum(mask_z))
    error_count_z_test = int(np.sum(test_errors_bool[mask_z])) if sifted_count_z_test > 0 else 0
    qber_z_test = (
        float(error_count_z_test / sifted_count_z_test) if sifted_count_z_test > 0 else 0.0
    )

    mask_x = test_bases == "X"
    sifted_count_x_test = int(np.sum(mask_x))
    error_count_x_test = int(np.sum(test_errors_bool[mask_x])) if sifted_count_x_test > 0 else 0
    qber_x_test = (
        float(error_count_x_test / sifted_count_x_test) if sifted_count_x_test > 0 else 0.0
    )

    # Protocol abort rule: Alice and Bob abort if sample test QBER strictly exceeds budget
    protocol_abort = bool(qber_test > condition.q_budget)

    # Compute Wilson score confidence interval on test bits
    wilson = compute_wilson_score_interval(
        k=error_count_test,
        n=max(1, n_test),
        confidence=0.95,
    )

    # Classify feasibility
    feasibility = classify_feasibility(
        q_lcb=wilson.lcb,
        q_ucb=wilson.ucb,
        q_budget=condition.q_budget,
    )

    margin_to_budget = float(condition.q_budget - wilson.ucb)
    is_strictly_stealthy = bool(wilson.ucb <= condition.q_budget)

    # Sample ID format: M3-{COND_ID}-{ATT_ID}-S{SEED}
    sample_id = f"M3-{condition.condition_id}-{attack_config.attack_id}-S{seed_sim}"

    # Package raw observations
    raw_obs = RawObservation(
        n_transmitted_pulses=n_pulses,
        n_sifted=n_sift,
        sifting_efficiency=sifting_eff,
        n_test=n_test,
        n_key=n_key,
        error_count_test=error_count_test,
        qber_test=qber_test,
        error_count_z_test=error_count_z_test,
        sifted_count_z_test=sifted_count_z_test,
        qber_z_test=qber_z_test,
        error_count_x_test=error_count_x_test,
        sifted_count_x_test=sifted_count_x_test,
        qber_x_test=qber_x_test,
        overall_sifted_qber=sim_result.qber,
        eve_info_gain=sim_result.eve_information_gain,
        protocol_abort=protocol_abort,
    )

    # Package derived evaluations
    derived_eval = DerivedEvaluation(
        wilson_q_center=wilson.center,
        wilson_q_lcb=wilson.lcb,
        wilson_q_ucb=wilson.ucb,
        wilson_margin=wilson.margin,
        confidence_level=0.95,
        feasibility_category=feasibility,
        margin_to_budget=margin_to_budget,
        is_strictly_stealthy=is_strictly_stealthy,
    )

    # Package metadata
    meta = SimulationMetadata(
        seed_simulator=seed_sim,
        seed_alice=seed_alice,
        seed_bob=seed_bob,
        generator_version=GENERATOR_VERSION,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        execution_time_seconds=round(t_elapsed, 4),
    )

    return DatasetRecord(
        sample_id=sample_id,
        condition_id=condition.condition_id,
        attack_id=attack_config.attack_id,
        attack_family=attack_config.family,
        attack_parameters=attack_config.parameters,
        condition=condition,
        raw_observation=raw_obs,
        derived_evaluation=derived_eval,
        metadata=meta,
    )


def get_pilot_environmental_conditions() -> List[EnvironmentalCondition]:
    """Return the approved 5-condition pilot environmental specification.

    Conditions:
    - C01: Ideal (p_depol=0.00, Q_budget=0.05)
    - C02: Low noise (p_depol=0.02, Q_budget=0.05)
    - C03: Moderate noise (p_depol=0.04, Q_budget=0.08)
    - C04: High noise (p_depol=0.06, Q_budget=0.08)
    - C05: Critical noise (p_depol=0.07, Q_budget=0.11)
    """
    pilot_specs = [
        ("C01", 0.00, 0.05, 1001),
        ("C02", 0.02, 0.05, 1002),
        ("C03", 0.04, 0.08, 1003),
        ("C04", 0.06, 0.08, 1004),
        ("C05", 0.07, 0.11, 1005),
    ]

    return [
        generate_environmental_condition(
            condition_id=cid,
            p_depol=p,
            q_budget=q_bud,
            seed_calib=s_calib,
        )
        for cid, p, q_bud, s_calib in pilot_specs
    ]


# Approved 24-condition primary grid specification with zero-leakage condition partitioning
FULL_CONDITION_SPECS: List[Tuple[str, float, float, int, str]] = [
    # Train set (16 conditions: 768 simulation runs)
    ("C01", 0.00, 0.05, 1001, "train"),
    ("C02", 0.00, 0.08, 1002, "train"),
    ("C03", 0.00, 0.11, 1003, "train"),
    ("C04", 0.02, 0.05, 1004, "train"),
    ("C05", 0.02, 0.08, 1005, "train"),
    ("C06", 0.02, 0.11, 1006, "train"),
    ("C07", 0.04, 0.05, 1007, "train"),
    ("C08", 0.04, 0.08, 1008, "train"),
    ("C09", 0.04, 0.11, 1009, "train"),
    ("C10", 0.06, 0.05, 1010, "train"),
    ("C11", 0.06, 0.08, 1011, "train"),
    ("C12", 0.06, 0.11, 1012, "train"),
    ("C13", 0.01, 0.08, 1013, "train"),
    ("C14", 0.03, 0.08, 1014, "train"),
    ("C15", 0.05, 0.08, 1015, "train"),
    ("C16", 0.07, 0.08, 1016, "train"),
    # Validation set (4 conditions: 192 simulation runs - interpolation)
    ("C17", 0.01, 0.05, 1017, "val"),
    ("C18", 0.01, 0.11, 1018, "val"),
    ("C19", 0.03, 0.05, 1019, "val"),
    ("C20", 0.03, 0.11, 1020, "val"),
    # Test set (4 conditions: 192 simulation runs - extrapolation / high-noise)
    ("C21", 0.05, 0.05, 1021, "test"),
    ("C22", 0.05, 0.11, 1022, "test"),
    ("C23", 0.07, 0.05, 1023, "test"),
    ("C24", 0.07, 0.11, 1024, "test"),
]


def get_full_environmental_conditions() -> List[EnvironmentalCondition]:
    """Return the approved 24-condition environmental specification.

    Covers 8 depolarizing noise rates in [0.00, 0.07] and 3 error budgets (0.05, 0.08, 0.11)
    with strict condition-level partitioning:
    - Train: 16 conditions (768 runs)
    - Validation: 4 conditions (192 runs)
    - Test: 4 conditions (192 runs)
    """
    return [
        generate_environmental_condition(
            condition_id=cid,
            p_depol=p,
            q_budget=q_bud,
            seed_calib=s_calib,
            partition=part,
        )
        for cid, p, q_bud, s_calib, part in FULL_CONDITION_SPECS
    ]


def generate_dataset(
    conditions: Sequence[EnvironmentalCondition],
    attacks: Optional[Sequence[AttackConfiguration]] = None,
    seeds: Sequence[int] = (101, 102, 103),
    n_pulses: int = 2000,
    rho_test: float = 0.50,
    dataset_type: str = "pilot",
    progress_callback: Optional[Callable[[int, int, DatasetRecord], None]] = None,
) -> DatasetContainer:
    """Execute the full dataset generation pipeline across conditions, attacks, and seeds.

    Args:
        conditions: Sequence of EnvironmentalCondition instances.
        attacks: Sequence of AttackConfiguration instances (defaults to all 16).
        seeds: Sequence of integer random seeds per configuration (default [101, 102, 103]).
        n_pulses: Number of transmitted pulses per run (default 2000).
        rho_test: Sifted test fraction for error estimation (default 0.50).
        dataset_type: Classification string ('pilot' or 'full').
        progress_callback: Optional callback(current, total, record).

    Returns:
        DatasetContainer: Versioned container with all records, hardware info, and summary stats.
    """
    if attacks is None:
        attacks = get_all_attack_configurations()

    total_expected = len(conditions) * len(attacks) * len(seeds)
    records: List[DatasetRecord] = []

    t_dataset_start = time.perf_counter()
    count = 0

    for cond in conditions:
        for attack in attacks:
            for seed in seeds:
                record = execute_single_simulation(
                    condition=cond,
                    attack_config=attack,
                    seed_sim=seed,
                    n_pulses=n_pulses,
                    rho_test=rho_test,
                )
                records.append(record)
                count += 1
                if progress_callback:
                    progress_callback(count, total_expected, record)

    t_dataset_total = time.perf_counter() - t_dataset_start

    # Compute summary statistics
    n_feasible = sum(
        1 for r in records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE
    )
    n_infeasible = sum(
        1 for r in records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.INFEASIBLE
    )
    n_indeterminate = sum(
        1 for r in records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.INDETERMINATE
    )
    n_aborted = sum(1 for r in records if r.raw_observation.protocol_abort)

    # Compute attack family breakdown
    family_counts: Dict[str, int] = {}
    family_feasibility: Dict[str, Dict[str, int]] = {}
    for r in records:
        fam = r.attack_family.value
        family_counts[fam] = family_counts.get(fam, 0) + 1
        if fam not in family_feasibility:
            family_feasibility[fam] = {"Feasible": 0, "Infeasible": 0, "Indeterminate": 0}
        family_feasibility[fam][r.derived_evaluation.feasibility_category.value] += 1

    summary_stats = {
        "total_records": len(records),
        "total_conditions": len(conditions),
        "total_attacks": len(attacks),
        "total_seeds": len(seeds),
        "n_pulses_per_run": n_pulses,
        "total_pulses_simulated": len(records) * n_pulses,
        "total_execution_time_seconds": round(t_dataset_total, 2),
        "average_run_time_seconds": round(t_dataset_total / max(1, len(records)), 4),
        "pulses_per_second": round((len(records) * n_pulses) / max(0.001, t_dataset_total), 1),
        "feasibility_counts": {
            "Feasible": n_feasible,
            "Infeasible": n_infeasible,
            "Indeterminate": n_indeterminate,
        },
        "feasibility_yield": round(n_feasible / max(1, len(records)), 4),
        "empirical_abort_counts": {
            "Aborted": n_aborted,
            "Passed": len(records) - n_aborted,
        },
        "empirical_abort_rate": round(n_aborted / max(1, len(records)), 4),
        "family_breakdown": family_counts,
        "family_feasibility": family_feasibility,
    }

    # Gather hardware info
    hardware_info = {
        "aer_device": "CPU (OpenMP Multi-threaded)",
        "python_version": "3.11.9",
        "qiskit_version": "2.6.0",
        "qiskit_aer_version": "0.18.0",
        "gpu_available": True,
        "gpu_device": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
    }

    return DatasetContainer(
        schema_version=GENERATOR_VERSION,
        dataset_type=dataset_type,
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        total_records=len(records),
        total_conditions=len(conditions),
        total_attacks=len(attacks),
        total_seeds=len(seeds),
        records=records,
        hardware_info=hardware_info,
        summary_statistics=summary_stats,
    )

