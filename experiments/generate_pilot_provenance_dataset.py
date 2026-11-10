"""Generate the corrected pilot dataset with complete calibration provenance (k_calib, seed_calib).

Preserves results/pilot_dataset.json as historical artifact.
Outputs:
    - results/pilot_dataset_provenance.json
    - results/pilot_benchmark_metrics_provenance.json
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))
from experiments.generate_pilot_dataset import run_pilot_generation

if __name__ == "__main__":
    run_pilot_generation(
        output_path="results/pilot_dataset_provenance.json",
        metrics_path="results/pilot_benchmark_metrics_provenance.json",
        n_pulses=2000,
        seeds=(101, 102, 103),
    )

