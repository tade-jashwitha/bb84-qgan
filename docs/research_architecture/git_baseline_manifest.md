# INITIAL GIT BASELINE MANIFEST
**Project:** Adaptive QGAN-Based Eavesdropping in BB84 QKD: Resource-Constrained Conditional Quantum Generative Approach under NISQ Noise  
**Repository:** `C:\Users\lenovo\bb84-qgan-research`  
**Execution Date:** November 10, 2026  
**Status:** INITIAL GIT BASELINE COMMITTED & FULLY VERIFIED  

---

## 1. COMMIT RECORD & REPOSITORY STATUS

The initial scientific research baseline has been formally staged and committed in the primary repository:

- **Commit Hash (Full):** `23148a4956c2c1975d940f14fd064eaa3a9cb7ad`
- **Commit Hash (Short):** `23148a4`
- **Commit Date:** `Tue Nov 10 20:37:09 2026 +0530`
- **Commit Subject:** `feat: initial scientific baseline for adaptive QGAN eavesdropping research`
- **Branch:** `master`
- **Total Files Committed:** **85 files** (117,145 insertions, 0 deletions)
- **Total Repository Footprint:** **~6.90 MB** (7,224,310 bytes)
- **Final Git Status:** **Clean** (`git status --short` returns empty)
- **Git LFS Status:** **Not required** (Direct tracking verified; largest file 2.13 MB, checkpoints ~375 KB each, well below Git's 50 MB / 100 MB limits).

---

## 2. CRITICAL RESEARCH ARTIFACT HASHES (SHA-256)

All critical research assets were re-verified post-commit to confirm byte-for-byte preservation:

| Research Artifact | File Role | SHA-256 Hash | Post-Commit Status |
| :--- | :--- | :--- | :--- |
| `results/full_training_dataset.json` | Canonical 1,152-record research dataset | `07C03E810470B3376EB48C51C13138DCF347D915F78E911F74A81E96F27CFB80` | **VERIFIED MATCH** |
| `checkpoints/cgan_pilot_best.pt` | Historical Pilot Best Checkpoint (Epoch 16) | `F4050379F065E9FE3106A23D746396215EBF128F52249220013B07411ED827AA` | **VERIFIED MATCH** |
| `checkpoints/cgan_pilot_latest.pt` | Historical Pilot Latest Checkpoint (Epoch 20) | `0BAA0DBD0BE0FB4D75E752AFD2EB1E86045CBCDFB4D1468DA27B0A2BF618E1D5` | **VERIFIED MATCH** |
| `checkpoints/cgan_best.pt` | Preserved baseline best weights | `F4050379F065E9FE3106A23D746396215EBF128F52249220013B07411ED827AA` | **VERIFIED MATCH** |
| `checkpoints/cgan_latest.pt` | Preserved baseline latest weights | `606CB0B34D6A8102DED048953A51DAA3C58187544EB96374505EF539C8E4B57D` | **VERIFIED MATCH** |
| `src/cgan/models.py` | Generator $[0, \pi/4]$ bounds & discriminator | `E1B5B4E7A602C55568B73309B62F2D94C44EED27DD4FD94DD264F60791908033` | **VERIFIED MATCH** |
| `src/cgan/dataset.py` | Dataset loader & Wilson score filter | `98ADACCFD70A3A923E8B3AA5A30069BB3A9BC6F0EF62640E9DF72D644B8CF157` | **VERIFIED MATCH** |
| `src/cgan/trainer.py` | cGAN training loop & loss functions | `FF160CEF08B60BB9ADCB49C1A9A3D3B4DEFACC5E2201BE7524BD897276CB5AB4` | **VERIFIED MATCH** |
| `src/cgan/policy_engine.py` | Policy generation & Fallback Tri-Case | `62449A136124E853D9CB855B1CA314B76A2A3BD16134A5D0B2355BE802724CE5` | **VERIFIED MATCH** |
| `experiments/train_classical_cgan.py` | Authoritative 200-epoch training runner | `BA5787A8D4A24BA32358CD1B6B0A7C1A89A570466EBDD838D193052038B9F0B7` | **VERIFIED MATCH** |
| `tests/test_cgan_pipeline.py` | 24 unit tests including CRN channel coupling | `724A5A78D2217B08761ADC1C04D3BE126102A3AA546C1FC308B42C5B12213EAA` | **VERIFIED MATCH** |
| `.gitignore` | Project git ignore rules | `8749BF42B4F6498AF571CAD151BCA123B868BD5C8F0F5857E3564C55B2912458` | **VERIFIED MATCH** |

---

## 3. SENSITIVE CONTENT & PII VERIFICATION

- **API Keys / Secrets / Tokens:** 0 found.
- **Private Keys / Certificates:** 0 found.
- **Committer Information:** Standard identity `tadejashwi07-coder` / `tadejashwi07@gmail.com`.
- **Filesystem Paths:** Clean Windows local development paths (`C:\Users\lenovo\bb84-qgan-research`).

---

## 4. EXCLUDED FILES (49 FILES, CLEAN WORKTREE)

The `.gitignore` configuration successfully prevented all ephemeral, build, and scratch artifacts from being staged or committed:
- **`experiments/__pycache__/*` (1 file, 49.6 KB)**
- **`src/**/__pycache__/*` (17 files, 235.3 KB)**
- **`tests/**/__pycache__/*` (9 files, 203.2 KB)**
- **`results/fontlist-v3.11.0.json` (1 file, 141.7 KB)**
- **`scratch/*` (21 files, ~110 KB)**

---

## 5. COMPLETE COMMITTED FILE INVENTORY (85 FILES)

### 5.1 Core Production Source Code (`src/` — 20 files, 201.2 KB)
- `src/__init__.py` (43 B)
- `src/bb84/__init__.py` (1.6 KB), `alice.py` (7.5 KB), `attacks.py` (21.7 KB), `bb84_engine.py` (17.3 KB), `bob.py` (18.5 KB), `channel.py` (323 B)
- `src/cgan/__init__.py` (1.1 KB), `dataset.py` (8.6 KB), `models.py` (8.9 KB), `policy_engine.py` (9.0 KB), `trainer.py` (13.7 KB)
- `src/channel/__init__.py` (2.1 KB), `noise_models.py` (24.0 KB), `quantum_channel.py` (10.2 KB)
- `src/dataset/__init__.py` (1.2 KB), `generator.py` (16.3 KB), `policy_generator.py` (9.6 KB), `schema.py` (21.3 KB), `statistics.py` (8.2 KB)

### 5.2 Test Suite (`tests/` — 9 files, 117.8 KB, 149/149 Tests Passing)
- `tests/__init__.py` (49 B)
- `tests/test_alice.py` (10.4 KB), `test_attacks.py` (12.9 KB), `test_bb84_engine.py` (6.5 KB), `test_bob.py` (16.1 KB)
- `tests/test_cgan_pipeline.py` (25.6 KB, 24/24 tests including CRN channel coupling)
- `tests/test_dataset_pipeline.py` (20.8 KB), `test_noise_models.py` (15.1 KB), `test_noisy_channel.py` (10.4 KB)

### 5.3 Experiment Runners (`experiments/` — 12 files, 175.0 KB)
- `experiments/audit_full_dataset.py` (12.7 KB), `audit_pilot.py` (6.3 KB), `audit_pilot_provenance.py` (7.7 KB)
- `experiments/benchmark_baselines.py` (17.2 KB)
- `experiments/generate_full_dataset.py` (9.8 KB), `generate_pilot_dataset.py` (10.2 KB), `generate_pilot_provenance_dataset.py` (653 B)
- `experiments/run_bb84_simulation.py` (16.2 KB), `run_milestone2_experiments.py` (26.7 KB)
- `experiments/train_cgan.py` (7.2 KB), `train_cgan_pilot.py` (15.0 KB), `train_milestone4a.py` (45.4 KB)

### 5.4 Model Checkpoints (`checkpoints/` — 4 files, 1.47 MB)
- `checkpoints/cgan_best.pt` (374.7 KB), `cgan_latest.pt` (375.3 KB)
- `checkpoints/cgan_pilot_best.pt` (374.7 KB), `cgan_pilot_latest.pt` (375.8 KB)

### 5.5 Research Datasets (`results/` — 8 files, 3.04 MB)
- `results/full_training_dataset.json` (2.13 MB, SHA-256: `07C0...`)
- `results/pilot_dataset.json` (434.6 KB), `pilot_dataset_provenance.json` (446.9 KB)
- `results/cgan_pilot_training_history.json` (6.5 KB), `cgan_pilot_validation_evaluation.json` (40.5 KB)
- `results/full_dataset_benchmark_metrics.json` (1.9 KB), `pilot_benchmark_metrics.json` (2.3 KB), `pilot_benchmark_metrics_provenance.json` (2.3 KB)

### 5.6 Validation Results & Logs (`results/` — 6 files, 39.5 KB)
- `results/bb84_simulation_results.json` (7.5 KB), `milestone2_experimental_results.json` (21.0 KB)
- `results/classical_cgan/checkpoint_selection_verification.json` (3.4 KB), `data_isolation_verification.json` (2.3 KB)
- `results/classical_cgan/execution_estimate.json` (2.6 KB), `frozen_training_config.json` (2.6 KB)

### 5.7 Generated Sweep Figures (`results/` — 8 files, 1.54 MB)
- `results/exp1_depolarizing_sweep.png` through `exp5_unitary_probe_tradeoff.png` (8 figures)

### 5.8 Documentation & Reports (17 files, 335.5 KB)
- `final_report.md`, `architecture_design.md`, `dataset_acceptance_report.md`
- `pilot_training_report.md`, `post_pilot_audit.md`, `training_readiness.md`
- `pretraining_scientific_audit.md`, `training_authorization.md`
- `quantum_noise_models.md`, `pilot_dataset_validation.md`, `engineering_plan.md`
- `docs/git_baseline_manifest.md`, `docs/crn_and_baseline_closure.md`
- `docs/pretraining_scientific_audit.md`, `docs/execution_authorization_review.md`
- `docs/repository_consolidation.md`, `docs/migration_log.json`

### 5.9 Repository Configuration (1 file, 288 B)
- `.gitignore` (288 B)

---

## 6. SECONDARY WORKTREE BACKUP STATUS

- **Secondary Worktree Path:** `C:\Users\lenovo\.gemini\antigravity\worktrees\bb84-qgan-research\nisq_noise_modelling`
- **Total Files Audited:** 85 / 85
- **Identical Hash Matches:** **85 / 85 (100.00%)**
- **Integrity Statement:** The secondary worktree backup remains 100% intact and verified against the committed baseline.
