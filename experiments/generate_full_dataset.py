"""Full training dataset generation and benchmarking script for BB84 attack policies.

Generates the complete approved 24-condition primary grid:
- 8 depolarizing noise rates: 0.00 to 0.07
- 3 QBER abort budgets: 0.05, 0.08, 0.11
- 24 environmental conditions (C01 to C24)
- 16 attack configurations per condition (ATT-00 to ATT-15)
- 3 independent seeds per configuration (101, 102, 103)
Total: 24 x 16 x 3 = 1,152 physical simulation records (2,000 pulses each).

Condition-level zero-leakage partitions:
- Train: 16 conditions (768 records)
- Validation: 4 conditions (192 records)
- Test: 4 conditions (192 records)

Execution:
    python experiments/generate_full_dataset.py

Outputs:
    - results/full_training_dataset.json (1,152 simulation records)
    - results/full_dataset_benchmark_metrics.json (Execution time, hardware profile, feasibility stats)
"""

import datetime
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
    get_full_environmental_conditions,
)
from src.dataset.policy_generator import get_all_attack_configurations
from src.dataset.schema import AttackFamily, DatasetContainer, StatisticalFeasibility


def run_full_dataset_generation(
    output_path: str = "results/full_training_dataset.json",
    metrics_path: str = "results/full_dataset_benchmark_metrics.json",
    n_pulses: int = 2000,
    seeds: tuple = (101, 102, 103),
) -> DatasetContainer:
    """Execute the full 24-condition dataset generation with detailed resource profiling."""
    print("=" * 80)
    print("M3 STAGE C: FULL BB84 ATTACK-POLICY DATASET GENERATION (1,152 RECORDS)")
    print("=" * 80)

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
    conditions = get_full_environmental_conditions()
    attacks = get_all_attack_configurations()
    total_expected = len(conditions) * len(attacks) * len(seeds)

    train_conds = [c for c in conditions if c.partition == "train"]
    val_conds = [c for c in conditions if c.partition == "val"]
    test_conds = [c for c in conditions if c.partition == "test"]

    print(f"\nExperimental Grid Specification:")
    print(f"  Total Conditions: {len(conditions)} (C01 to C24)")
    print(f"    - Train Partition: {len(train_conds)} conditions ({len(train_conds) * len(attacks) * len(seeds)} runs)")
    print(f"    - Val Partition:   {len(val_conds)} conditions ({len(val_conds) * len(attacks) * len(seeds)} runs)")
    print(f"    - Test Partition:  {len(test_conds)} conditions ({len(test_conds) * len(attacks) * len(seeds)} runs)")
    print(f"  Attacks per Condition: {len(attacks)} (ATT-00 to ATT-15)")
    print(f"  Seeds per Attack: {len(seeds)} {list(seeds)}")
    print(f"  Total Simulation Runs: {total_expected}")
    print(f"  Total Pulses Simulated: {total_expected * n_pulses:,} optical carrier pulses\n")

    t_start = time.perf_counter()

    # Progress reporting callback
    def progress(current: int, total: int, rec: Any):
        if current % 48 == 0 or current == total:
            elapsed = time.perf_counter() - t_start
            rate = current / elapsed if elapsed > 0 else 0
            eta_sec = (total - current) / rate if rate > 0 else 0
            print(
                f"[{current:4d}/{total:4d}] "
                f"Condition: {rec.condition_id} ({rec.condition.partition}) | "
                f"Noise p={rec.condition.p_depol:.2f}, Q_bud={rec.condition.q_budget:.2f} | "
                f"Rate: {rate:.1f} runs/s | Elapsed: {elapsed/60.0:.1f}m | ETA: {eta_sec/60.0:.1f}m",
                flush=True,
            )

    print("Commencing quantum simulations...", flush=True)
    container = generate_dataset(
        conditions=conditions,
        attacks=attacks,
        seeds=seeds,
        n_pulses=n_pulses,
        rho_test=0.50,
        dataset_type="full",
        progress_callback=progress,
    )

    t_total = time.perf_counter() - t_start

    # Final memory
    ram_end_mb = process.memory_info().rss / (1024.0 * 1024.0)
    ram_peak_mb = ram_end_mb
    vram_end_mb = (
        torch.cuda.memory_allocated() / (1024.0 * 1024.0) if cuda_available else 0.0
    )

    # Save dataset JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    container.save_json(output_path)
    file_size_kb = os.path.getsize(output_path) / 1024.0

    # Partition statistics
    partition_stats: Dict[str, Dict[str, int]] = {}
    for part in ["train", "val", "test"]:
        part_records = [r for r in container.records if r.condition.partition == part]
        partition_stats[part] = {
            "total_runs": len(part_records),
            "feasible": sum(1 for r in part_records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE),
            "infeasible": sum(1 for r in part_records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.INFEASIBLE),
            "indeterminate": sum(1 for r in part_records if r.derived_evaluation.feasibility_category == StatisticalFeasibility.INDETERMINATE),
            "protocol_abort": sum(1 for r in part_records if r.raw_observation.protocol_abort),
            "protocol_pass": sum(1 for r in part_records if not r.raw_observation.protocol_abort),
        }

    # Summary metrics dictionary
    metrics = {
        "execution_metadata": {
            "dataset_version": container.dataset_version,
            "dataset_type": "full_primary_grid",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "output_path": output_path,
            "file_size_kb": round(file_size_kb, 2),
            "total_records": len(container.records),
            "total_pulses_simulated": len(container.records) * n_pulses,
            "n_pulses_per_run": n_pulses,
            "seeds_per_attack": list(seeds),
        },
        "performance_and_hardware": {
            "total_execution_time_seconds": round(t_total, 2),
            "total_execution_time_minutes": round(t_total / 60.0, 2),
            "average_time_per_run_seconds": round(t_total / len(container.records), 4),
            "simulation_throughput_pulses_per_second": round(
                (len(container.records) * n_pulses) / t_total, 1
            ),
            "initial_ram_mb": round(ram_start_mb, 2),
            "final_ram_mb": round(ram_end_mb, 2),
            "ram_delta_mb": round(ram_end_mb - ram_start_mb, 2),
            "cuda_device_name": device_name,
            "vram_allocated_mb": round(vram_end_mb, 2),
        },
        "aggregate_feasibility_and_aborts": {
            "feasible_count": container.feasibility_summary.get("Feasible", 0),
            "feasible_percentage": round(
                100.0 * container.feasibility_summary.get("Feasible", 0) / len(container.records), 2
            ),
            "infeasible_count": container.feasibility_summary.get("Infeasible", 0),
            "infeasible_percentage": round(
                100.0 * container.feasibility_summary.get("Infeasible", 0) / len(container.records), 2
            ),
            "indeterminate_count": container.feasibility_summary.get("Indeterminate", 0),
            "indeterminate_percentage": round(
                100.0 * container.feasibility_summary.get("Indeterminate", 0) / len(container.records), 2
            ),
            "protocol_abort_count": container.protocol_aborts_count,
            "protocol_abort_percentage": round(
                100.0 * container.protocol_aborts_count / len(container.records), 2
            ),
            "protocol_pass_count": len(container.records) - container.protocol_aborts_count,
            "protocol_pass_percentage": round(
                100.0 * (len(container.records) - container.protocol_aborts_count) / len(container.records), 2
            ),
        },
        "partition_breakdown": partition_stats,
        "attack_family_breakdown": container.family_breakdown,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print(f"Total Records: {len(container.records)}")
    print(f"Total Time: {t_total:.2f}s ({t_total/60.0:.2f} min)")
    print(f"Average Run Time: {t_total/len(container.records):.4f}s / run")
    print(f"Throughput: {(len(container.records)*n_pulses)/t_total:.1f} pulses/s")
    print(f"Feasible: {container.feasibility_summary.get('Feasible', 0)} ({100.0 * container.feasibility_summary.get('Feasible', 0) / len(container.records):.1f}%)")
    print(f"Indeterminate: {container.feasibility_summary.get('Indeterminate', 0)} ({100.0 * container.feasibility_summary.get('Indeterminate', 0) / len(container.records):.1f}%)")
    print(f"Infeasible: {container.feasibility_summary.get('Infeasible', 0)} ({100.0 * container.feasibility_summary.get('Infeasible', 0) / len(container.records):.1f}%)")
    print(f"Protocol Aborts: {container.protocol_aborts_count} ({100.0 * container.protocol_aborts_count / len(container.records):.1f}%)")
    print(f"Dataset File: {output_path} ({file_size_kb:.2f} KB)")
    print(f"Metrics File: {metrics_path}")
    print("=" * 80)

    return container


if __name__ == "__main__":
    run_full_dataset_generation()
