# Cryptographic Provenance & Environmental Calibration

## 1. Provenance Architecture

To guarantee strict reproducibility across experimental platforms, all synthetic dataset generation runs record immutable provenance metadata (`results/pilot_dataset_provenance.json`):
- Random number generator seeds (NumPy, PyTorch, Python `random`).
- CPU architecture and execution environment details.
- Exact Git commit baseline.
- Dataset record count and field-level SHA-256 digests.

## 2. Invariant Preservation

The full training dataset hash is permanently recorded as:
`07C03E810470B3376EB48C51C13138DCF347D915F78E911F74A81E96F27CFB80`

Any modification or regeneration of the training dataset is strictly prohibited.
