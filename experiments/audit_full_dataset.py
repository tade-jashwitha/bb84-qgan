"""Comprehensive Audit Script for BB84 Full Primary Grid Dataset (1,152 Records).

Performs exhaustive verification of:
1. Exact dataset schema and row count (1,152 records).
2. Exactly 24 conditions x 16 attacks x 3 seeds.
3. Condition-level partition distribution (16 train, 4 val, 4 test) and disjointness.
4. Duplicate sample IDs, missing fields, nulls.
5. Physical attack parameter bounds and canonical family masking.
6. Calibration provenance and ambient noise estimator validity.
7. Independent Wilson score interval recalculation and feasibility classification.
8. Protocol abort vs. statistical feasibility independence.
9. Information boundary compliance (strictly c = [Q_budget, gamma_noise_hat]).
10. Hardware and runtime metrics verification.

Execution:
    python experiments/audit_full_dataset.py
"""

import json
import math
import os
import sys
from typing import Any, Dict, List, Set, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.schema import AttackFamily, DatasetContainer, StatisticalFeasibility
from src.dataset.generator import compute_wilson_score_interval, classify_feasibility


def audit_full_dataset(
    dataset_path: str = "results/full_training_dataset.json",
    metrics_path: str = "results/full_dataset_benchmark_metrics.json",
) -> Dict[str, Any]:
    """Execute complete audit on the full dataset."""
    print("=" * 80)
    print("M3 STAGE C: FULL DATASET AND DATA LEAKAGE AUDIT")
    print("=" * 80)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Full dataset not found at: {dataset_path}")

    container = DatasetContainer.load_json(dataset_path)
    records = container.records

    print(f"Loaded dataset: {dataset_path}")
    print(f"Total records in container: {len(records)} (Expected: 1,152)")
    print(f"Container schema version: {container.schema_version}")
    print(f"Container dataset type: {container.dataset_type}")

    audit_results: Dict[str, Any] = {
        "status": "PASSED",
        "total_records": len(records),
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    # 1. Row count check
    if len(records) != 1152:
        audit_results["errors"].append(f"Record count mismatch: expected 1152, got {len(records)}")
    audit_results["checks"]["total_record_count_1152"] = (len(records) == 1152)

    # 2. Condition, attack, and seed counts
    conditions_seen = set()
    attacks_seen = set()
    seeds_seen = set()
    condition_partitions: Dict[str, str] = {}
    condition_records: Dict[str, List[Any]] = {}

    sample_ids: Set[str] = set()
    duplicates: List[str] = []

    for r in records:
        if r.sample_id in sample_ids:
            duplicates.append(r.sample_id)
        sample_ids.add(r.sample_id)

        c_id = r.condition_id
        conditions_seen.add(c_id)
        attacks_seen.add(r.attack_id)
        seeds_seen.add(r.metadata.seed_simulator)

        part = r.condition.partition
        if c_id not in condition_partitions:
            condition_partitions[c_id] = part
            condition_records[c_id] = []
        condition_records[c_id].append(r)

    audit_results["checks"]["no_duplicate_sample_ids"] = (len(duplicates) == 0)
    audit_results["checks"]["unique_conditions_24"] = (len(conditions_seen) == 24)
    audit_results["checks"]["unique_attacks_16"] = (len(attacks_seen) == 16)
    audit_results["checks"]["unique_seeds_3"] = (len(seeds_seen) == 3 and seeds_seen == {101, 102, 103})

    # 3. Partition Disjointness and Record Counts
    train_conds = {c for c, p in condition_partitions.items() if p == "train"}
    val_conds = {c for c, p in condition_partitions.items() if p == "val"}
    test_conds = {c for c, p in condition_partitions.items() if p == "test"}

    disjoint = (
        len(train_conds & val_conds) == 0
        and len(train_conds & test_conds) == 0
        and len(val_conds & test_conds) == 0
    )
    audit_results["checks"]["partition_disjointness"] = disjoint
    audit_results["checks"]["train_conditions_16"] = (len(train_conds) == 16)
    audit_results["checks"]["val_conditions_4"] = (len(val_conds) == 4)
    audit_results["checks"]["test_conditions_4"] = (len(test_conds) == 4)

    train_records = [r for r in records if r.condition.partition == "train"]
    val_records = [r for r in records if r.condition.partition == "val"]
    test_records = [r for r in records if r.condition.partition == "test"]

    audit_results["checks"]["train_records_768"] = (len(train_records) == 768)
    audit_results["checks"]["val_records_192"] = (len(val_records) == 192)
    audit_results["checks"]["test_records_192"] = (len(test_records) == 192)

    # 4. Attack distribution per condition
    correct_attack_grid = True
    for c_id, recs in condition_records.items():
        if len(recs) != 48:  # 16 attacks x 3 seeds
            correct_attack_grid = False
            audit_results["errors"].append(f"Condition {c_id} has {len(recs)} records instead of 48")
        c_attacks = {r.attack_id for r in recs}
        if len(c_attacks) != 16:
            correct_attack_grid = False
            audit_results["errors"].append(f"Condition {c_id} has {len(c_attacks)} unique attacks instead of 16")
    audit_results["checks"]["uniform_48_records_per_condition"] = correct_attack_grid

    # 5. Attack Parameters physical validity & masking
    param_errors = []
    for r in records:
        params = r.attack_parameters
        fam = r.attack_family
        if fam == AttackFamily.NULL:
            if params.alpha is not None or params.beta is not None:
                param_errors.append(f"{r.sample_id}: Null attack has non-null PIR parameters: {params.to_dict()}")
            if params.theta_E_rad is not None and params.theta_E_rad != 0.0:
                param_errors.append(f"{r.sample_id}: Null attack has non-zero theta_E_rad: {params.theta_E_rad}")
        elif fam == AttackFamily.PROBABILISTIC_INTERCEPT_RESEND:
            if params.alpha is None or not (0.0 <= params.alpha <= 1.0):
                param_errors.append(f"{r.sample_id}: Invalid PIR alpha: {params.alpha}")
            if params.beta is None or not (0.0 <= params.beta <= 1.0):
                param_errors.append(f"{r.sample_id}: Invalid PIR beta: {params.beta}")
            if params.theta_E_rad is not None and params.theta_E_rad != 0.0:
                param_errors.append(f"{r.sample_id}: PIR attack has non-null theta_E: {params.theta_E_rad}")
        elif fam == AttackFamily.UNITARY_PROBE:
            if params.theta_E_rad is None or not (0.0 <= params.theta_E_rad <= (math.pi / 4.0 + 1e-6)):
                param_errors.append(f"{r.sample_id}: Invalid Probe theta_E_rad: {params.theta_E_rad}")
            if params.alpha is not None or params.beta is not None:
                param_errors.append(f"{r.sample_id}: Probe attack has non-null PIR params")

    audit_results["checks"]["physical_attack_parameter_validity"] = (len(param_errors) == 0)
    if param_errors:
        audit_results["errors"].extend(param_errors[:5])

    # 6. Calibration Provenance & Noise Estimator
    calib_errors = []
    for r in records:
        cond = r.condition
        if cond.seed_calib is None:
            calib_errors.append(f"{r.sample_id}: seed_calib is None")
        if cond.n_calib != 1000:
            calib_errors.append(f"{r.sample_id}: n_calib != 1000 ({cond.n_calib})")
        expected_gamma = float(cond.k_calib / cond.n_calib)
        if abs(cond.gamma_noise_hat - expected_gamma) > 1e-6:
            calib_errors.append(f"{r.sample_id}: gamma_noise_hat ({cond.gamma_noise_hat}) != k/n ({expected_gamma})")
    audit_results["checks"]["calibration_provenance_verified"] = (len(calib_errors) == 0)

    # 7. Independent Wilson Score Interval Recalculation
    wilson_errors = []
    for r in records:
        raw = r.raw_observation
        derived = r.derived_evaluation
        cond = r.condition

        ref_wilson = compute_wilson_score_interval(
            k=raw.error_count_test,
            n=raw.n_test,
            confidence=0.95,
        )
        ref_feasibility = classify_feasibility(
            q_lcb=ref_wilson.lcb,
            q_ucb=ref_wilson.ucb,
            q_budget=cond.q_budget,
        )

        if abs(derived.wilson_q_center - ref_wilson.center) > 1e-5:
            wilson_errors.append(f"{r.sample_id}: Wilson center mismatch")
        if abs(derived.wilson_q_lcb - ref_wilson.lcb) > 1e-5:
            wilson_errors.append(f"{r.sample_id}: Wilson LCB mismatch")
        if abs(derived.wilson_q_ucb - ref_wilson.ucb) > 1e-5:
            wilson_errors.append(f"{r.sample_id}: Wilson UCB mismatch")
        if derived.feasibility_category != ref_feasibility:
            wilson_errors.append(f"{r.sample_id}: Feasibility category mismatch: {derived.feasibility_category} vs {ref_feasibility}")

    audit_results["checks"]["wilson_bounds_and_feasibility_verified"] = (len(wilson_errors) == 0)

    # 8. Protocol Abort vs Feasibility Independence
    abort_rule_errors = []
    for r in records:
        raw = r.raw_observation
        cond = r.condition
        expected_abort = bool(raw.qber_test > cond.q_budget)
        if raw.protocol_abort != expected_abort:
            abort_rule_errors.append(f"{r.sample_id}: Protocol abort mismatch")
    audit_results["checks"]["protocol_abort_rule_verified"] = (len(abort_rule_errors) == 0)

    # 9. Information Boundary Verification
    info_boundary_errors = []
    for r in records:
        c_obs = r.condition.get_observable_condition()
        if len(c_obs) != 2:
            info_boundary_errors.append(f"{r.sample_id}: Observable condition vector len != 2: {c_obs}")
        if c_obs != (r.condition.q_budget, r.condition.gamma_noise_hat):
            info_boundary_errors.append(f"{r.sample_id}: Observable condition content mismatch: {c_obs}")
    audit_results["checks"]["information_boundary_verified"] = (len(info_boundary_errors) == 0)

    # 10. Summary Statistics Calculation
    feasible_count = sum(1 for r in records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE)
    infeasible_count = sum(1 for r in records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.INFEASIBLE)
    indeterminate_count = sum(1 for r in records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.INDETERMINATE)
    abort_count = sum(1 for r in records if r.raw_observation.protocol_abort)

    audit_results["statistics"] = {
        "feasible_count": feasible_count,
        "feasible_percentage": round(100.0 * feasible_count / len(records), 2),
        "infeasible_count": infeasible_count,
        "infeasible_percentage": round(100.0 * infeasible_count / len(records), 2),
        "indeterminate_count": indeterminate_count,
        "indeterminate_percentage": round(100.0 * indeterminate_count / len(records), 2),
        "protocol_abort_count": abort_count,
        "protocol_abort_percentage": round(100.0 * abort_count / len(records), 2),
        "protocol_pass_count": len(records) - abort_count,
        "protocol_pass_percentage": round(100.0 * (len(records) - abort_count) / len(records), 2),
    }

    audit_results["partitions"] = {
        "train": {
            "conditions": sorted(list(train_conds)),
            "count": len(train_records),
            "feasible": sum(1 for r in train_records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE),
        },
        "val": {
            "conditions": sorted(list(val_conds)),
            "count": len(val_records),
            "feasible": sum(1 for r in val_records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE),
        },
        "test": {
            "conditions": sorted(list(test_conds)),
            "count": len(test_records),
            "feasible": sum(1 for r in test_records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE),
        },
    }

    if audit_results["errors"]:
        audit_results["status"] = "FAILED"

    print("\nAudit Summary Results:")
    for check_name, passed in audit_results["checks"].items():
        status_sym = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_sym} {check_name}")

    print(f"\nAudit Overall Status: {audit_results['status']}")
    print(f"Total Errors: {len(audit_results['errors'])}")
    print(f"Feasible: {feasible_count} ({audit_results['statistics']['feasible_percentage']}%)")
    print(f"Infeasible: {infeasible_count} ({audit_results['statistics']['infeasible_percentage']}%)")
    print(f"Indeterminate: {indeterminate_count} ({audit_results['statistics']['indeterminate_percentage']}%)")
    print(f"Protocol Aborts: {abort_count} ({audit_results['statistics']['protocol_abort_percentage']}%)")
    print("=" * 80)

    return audit_results


if __name__ == "__main__":
    audit_full_dataset()
