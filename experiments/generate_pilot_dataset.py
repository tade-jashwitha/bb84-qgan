"""Reproducible generation and benchmarking script for the 5-condition BB84 pilot dataset.

Execution:
    python experiments/generate_pilot_dataset.py

Outputs:
    - results/pilot_dataset.json (240 simulation records)
    - results/pilot_benchmark_metrics.json (Hardware, timing, and feasibility metrics)
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psutil
import torch

from src.dataset.generator import (
    generate_dataset,
    get_pilot_environmental_conditions,
)
from src.dataset.policy_generator import get_all_attack_configurations
from src.dataset.schema import AttackFamily, DatasetContainer, StatisticalFeasibility


def run_pilot_generation(
    output_path: str = "results/pilot_dataset.json",
    metrics_path: str = "results/pilot_benchmark_metrics.json",
    n_pulses: int = 2000,
    seeds: tuple = (101, 102, 103),
) -> DatasetContainer:
    """Execute the complete pilot dataset generation workflow with resource profiling."""
    print("=" * 78)
    print("M3 STAGE B: BB84 ATTACK-POLICY PILOT DATASET GENERATION")
    print("=" * 78)

    # Hardware profile
    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024.0 * 1024.0)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    vram_start_mb = (
        torch.cuda.memory_allocated() / (1024.0 * 1024.0) if cuda_available else 0.0
    )

    print(f"System: Windows | Python: {sys.version.split()[0]}")
    print(f"PyTorch CUDA Available: {cuda_available} | Device: {device_name}")
    print(f"Initial Process RAM: {ram_start_mb:.2f} MB | Initial VRAM: {vram_start_mb:.2f} MB")

    # Environmental conditions & attack configurations
    conditions = get_pilot_environmental_conditions()
    attacks = get_all_attack_configurations()
    total_expected = len(conditions) * len(attacks) * len(seeds)

    print(f"\nConfiguration:")
    print(f"  Conditions: {len(conditions)} (C01 to C05)")
    for c in conditions:
        print(
            f"    - {c.condition_id}: p_depol={c.p_depol:.2f}, Q_env={c.q_env:.4f}, "
            f"Q_budget={c.q_budget:.2f}, gamma_hat={c.gamma_noise_hat:.4f}"
        )
    print(f"  Attacks per Condition: {len(attacks)} (ATT-00 to ATT-15)")
    print(f"  Seeds per Attack: {len(seeds)} {list(seeds)}")
    print(f"  Pulses per Run: {n_pulses}")
    print(f"  Total Expected Simulation Runs: {total_expected}")
    print(f"  Total Expected Pulses: {total_expected * n_pulses:,}")
    print("\nStarting simulation executions...\n")

    # Benchmarking trackers
    t0_total = time.perf_counter()
    condition_times: Dict[str, float] = {c.condition_id: 0.0 for c in conditions}
    family_times: Dict[str, float] = {
        AttackFamily.NULL.value: 0.0,
        AttackFamily.UNITARY_PROBE.value: 0.0,
        AttackFamily.PROBABILISTIC_INTERCEPT_RESEND.value: 0.0,
    }
    family_counts: Dict[str, int] = {k: 0 for k in family_times}

    def progress_monitor(current: int, total: int, record):
        cid = record.condition_id
        fam = record.attack_family.value
        dur = record.metadata.execution_time_seconds
        condition_times[cid] = condition_times.get(cid, 0.0) + dur
        family_times[fam] = family_times.get(fam, 0.0) + dur
        family_counts[fam] = family_counts.get(fam, 0) + 1

        if current % 20 == 0 or current == total:
            pct = (current / float(total)) * 100.0
            elapsed = time.perf_counter() - t0_total
            rate = current / max(0.001, elapsed)
            eta = (total - current) / max(0.001, rate)
            print(
                f"[{current:3d}/{total:3d}] {pct:5.1f}% | "
                f"Elapsed: {elapsed:5.1f}s | Rate: {rate:4.2f} runs/s | ETA: {eta:5.1f}s | "
                f"Last: {record.sample_id} ({dur:.2f}s, {record.derived_evaluation.feasibility_category.value})"
            )

    # Generate dataset
    container = generate_dataset(
        conditions=conditions,
        attacks=attacks,
        seeds=seeds,
        n_pulses=n_pulses,
        dataset_type="pilot",
        progress_callback=progress_monitor,
    )

    t_total = time.perf_counter() - t0_total
    ram_end_mb = process.memory_info().rss / (1024.0 * 1024.0)
    ram_peak_mb = ram_end_mb  # Upper envelope
    vram_end_mb = (
        torch.cuda.memory_allocated() / (1024.0 * 1024.0) if cuda_available else 0.0
    )

    print("\n" + "=" * 78)
    print("SIMULATION EXECUTIONS COMPLETE — RUNNING INTEGRITY VERIFICATION")
    print("=" * 78)

    # Programmatic integrity checks
    records = container.records
    assert len(records) == total_expected, f"Record count mismatch: {len(records)} vs {total_expected}"

    sample_ids = [r.sample_id for r in records]
    assert len(sample_ids) == len(set(sample_ids)), "Duplicate sample_id detected in dataset!"

    for i, r in enumerate(records):
        raw = r.raw_observation
        derived = r.derived_evaluation

        # Bit count consistency
        assert raw.n_transmitted_pulses == n_pulses
        assert raw.n_test + raw.n_key == raw.n_sifted
        assert raw.sifted_count_z_test + raw.sifted_count_x_test == raw.n_test
        assert raw.error_count_z_test + raw.error_count_x_test == raw.error_count_test

        # Statistical intervals
        assert 0.0 <= derived.wilson_q_lcb <= derived.wilson_q_ucb <= 1.0
        assert math.isfinite(derived.wilson_q_center)
        assert math.isfinite(derived.wilson_margin)

        # Feasibility label consistency
        if derived.wilson_q_ucb <= r.condition.q_budget:
            assert derived.feasibility_category == StatisticalFeasibility.FEASIBLE
        elif derived.wilson_q_lcb > r.condition.q_budget:
            assert derived.feasibility_category == StatisticalFeasibility.INFEASIBLE
        else:
            assert derived.feasibility_category == StatisticalFeasibility.INDETERMINATE

        # Protocol abort consistency
        expected_abort = raw.qber_test > r.condition.q_budget
        assert raw.protocol_abort == expected_abort, f"Protocol abort inconsistency at index {i}"

    print("All 240 records passed programmatic integrity validation!")

    # Calculate family timing averages
    avg_family_times = {}
    for fam, tot_time in family_times.items():
        cnt = family_counts.get(fam, 0)
        avg_family_times[fam] = {
            "total_time_seconds": round(tot_time, 2),
            "run_count": cnt,
            "average_time_per_run_seconds": round(tot_time / max(1, cnt), 4),
            "pulses_per_second": round((cnt * n_pulses) / max(0.001, tot_time), 1),
        }

    # Calculate condition timing averages
    avg_condition_times = {}
    for cid, tot_time in condition_times.items():
        cnt = len(attacks) * len(seeds)
        avg_condition_times[cid] = {
            "total_time_seconds": round(tot_time, 2),
            "run_count": cnt,
            "average_time_per_run_seconds": round(tot_time / max(1, cnt), 4),
            "pulses_per_second": round((cnt * n_pulses) / max(0.001, tot_time), 1),
        }

    # Save dataset to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    container.save_json(output_path)
    dataset_file_size_kb = os.path.getsize(output_path) / 1024.0

    print(f"\nPilot dataset successfully saved to: {output_path} ({dataset_file_size_kb:.2f} KB)")

    # Build comprehensive benchmark metrics dictionary
    benchmark_metrics = {
        "timestamp_utc": container.created_at_utc,
        "dataset_type": "pilot",
        "total_records": len(records),
        "total_pulses_simulated": len(records) * n_pulses,
        "n_pulses_per_run": n_pulses,
        "total_runtime_seconds": round(t_total, 2),
        "total_runtime_minutes": round(t_total / 60.0, 2),
        "overall_average_time_per_run_seconds": round(t_total / len(records), 4),
        "overall_pulses_per_second": round((len(records) * n_pulses) / t_total, 1),
        "runtime_by_attack_family": avg_family_times,
        "runtime_by_condition": avg_condition_times,
        "memory_and_hardware": {
            "ram_initial_mb": round(ram_start_mb, 2),
            "ram_final_mb": round(ram_end_mb, 2),
            "ram_peak_delta_mb": round(ram_end_mb - ram_start_mb, 2),
            "vram_initial_mb": round(vram_start_mb, 2),
            "vram_final_mb": round(vram_end_mb, 2),
            "vram_delta_mb": round(vram_end_mb - vram_start_mb, 2),
            "aer_backend": "AerSimulator (CPU Multi-threaded OpenMP)",
            "pytorch_cuda_device": device_name,
        },
        "feasibility_statistics": container.summary_statistics["feasibility_counts"],
        "feasibility_yield": container.summary_statistics["feasibility_yield"],
        "empirical_abort_statistics": container.summary_statistics["empirical_abort_counts"],
        "empirical_abort_rate": container.summary_statistics["empirical_abort_rate"],
        "dataset_file_size_kb": round(dataset_file_size_kb, 2),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_metrics, f, indent=2)

    print(f"Benchmark metrics saved to: {metrics_path}")

    # Display final summary table
    print("\n" + "=" * 78)
    print("PILOT DATASET GENERATION SUMMARY")
    print("=" * 78)
    print(f"Total Runs: {len(records)} | Total Pulses: {len(records)*n_pulses:,}")
    print(f"Total Runtime: {t_total:.2f} s ({t_total/60.0:.2f} min)")
    print(f"Overall Throughput: {benchmark_metrics['overall_pulses_per_second']} pulses/s")
    print(f"Feasibility Yield: {benchmark_metrics['feasibility_yield']*100:.1f}% ({benchmark_metrics['feasibility_statistics']['Feasible']}/{len(records)})")
    print(f"Indeterminate Runs: {benchmark_metrics['feasibility_statistics']['Indeterminate']}")
    print(f"Infeasible Runs: {benchmark_metrics['feasibility_statistics']['Infeasible']}")
    print(f"Empirical Abort Rate: {benchmark_metrics['empirical_abort_rate']*100:.1f}% ({benchmark_metrics['empirical_abort_statistics']['Aborted']}/{len(records)})")
    print(f"Memory Delta: +{ram_end_mb - ram_start_mb:.2f} MB RAM | 0.0 MB VRAM (CPU simulation)")
    print("=" * 78)

    return container


if __name__ == "__main__":
    run_pilot_generation()
