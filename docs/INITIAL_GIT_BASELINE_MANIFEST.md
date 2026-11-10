# INITIAL GIT BASELINE MANIFEST
**Project:** Adaptive QGAN-Based Eavesdropping in BB84 QKD: Resource-Constrained Conditional Quantum Generative Approach under NISQ Noise  
**Repository:** `C:\Users\lenovo\bb84-qgan-research`  
**Execution Date:** November 10, 2026  
**Status:** STAGING MANIFEST PREPARED — AWAITING EXPLICIT USER APPROVAL BEFORE COMMITTING  

---

## EXECUTIVE SUMMARY

This manifest establishes the formal inventory and scientific baseline for the initial Git commit of the primary repository `C:\Users\lenovo\bb84-qgan-research`. 

### Baseline Status Summary:
- **Repository Branch:** `master` (Initial commit, 0 historical commits).
- **Candidate Files to Stage:** **84 files** (Total size: **7,212,044 bytes / ~6.88 MB**).
- **Excluded Files:** **49 files** (Python cache `__pycache__/*`, Matplotlib font cache `results/fontlist-*.json`, and intermediate audit tools `scratch/*`).
- **Regression Suite:** **149 / 149 tests passing** (`python -m unittest discover -s tests`).
- **Secondary Worktree Backup:** **84 / 84 files byte-for-byte identical** in `C:\Users\lenovo\.gemini\antigravity\worktrees\bb84-qgan-research\nisq_noise_modelling`.
- **Sensitive Information & Credentials Scan:** **Clean** (0 API keys, 0 secrets, 0 private tokens detected).
- **Execution Safeguard:** **NO `git add` or `git commit` has been executed.**

---

## 1. CRITICAL RESEARCH ARTIFACT HASHES (SHA-256)

All critical research assets have been cryptographically verified:

| Research Artifact | File Role | SHA-256 Hash | Status |
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
| `experiments/train_milestone4a.py` | Authoritative 200-epoch training runner | `BA5787A8D4A24BA32358CD1B6B0A7C1A89A570466EBDD838D193052038B9F0B7` | **VERIFIED MATCH** |
| `tests/test_cgan_pipeline.py` | 24 unit tests including CRN channel coupling | `724A5A78D2217B08761ADC1C04D3BE126102A3AA546C1FC308B42C5B12213EAA` | **VERIFIED MATCH** |
| `.gitignore` | Project git ignore rules | `8749BF42B4F6498AF571CAD151BCA123B868BD5C8F0F5857E3564C55B2912458` | **VERIFIED MATCH** |

---

## 2. GIT LFS CONFIGURATION AUDIT

- **Git LFS Binary Version:** `git-lfs/3.7.1 (GitHub; windows amd64; go 1.25.1; git b84b3384)`
- **Global Configuration:** Filter hooks installed in `~/.gitconfig` (`filter.lfs.process = git-lfs filter-process`).
- **Local Repository Status:**
  - `.gitattributes` is currently **absent**.
  - `git lfs track` reports **0 active tracking patterns**.
  - **Currently configured for Git LFS:** **0 files**.

### Analysis & Recommendation:
1. **Direct Git Tracking (Recommended):**
   - The entire repository footprint across all 84 candidate files is only **6.88 MB**.
   - The single largest file is `results/full_training_dataset.json` at **2.13 MB**.
   - The model checkpoints (`checkpoints/*.pt`) are lightweight PyTorch MLP weights of **~375 KB each** (total 1.47 MB).
   - The validation sweep figures (`results/*.png`) total **~1.54 MB**.
   - Because all files are two orders of magnitude below GitHub's 50 MB warning threshold and 100 MB hard limit, direct tracking maintains complete repository portability, offline independence, and avoids third-party LFS storage/bandwidth quotas.
2. **Alternative (Git LFS for Binaries):**
   - If the user prefers Git LFS tracking for `.pt` weights and `.png` figures, a `.gitattributes` file can be introduced:
     ```gitattributes
     *.pt filter=lfs diff=lfs merge=lfs -text
     *.png filter=lfs diff=lfs merge=lfs -text
     ```

---

## 3. SENSITIVE CONTENT, CREDENTIALS & PII AUDIT

A regex-based security audit of all 84 files was conducted covering API keys, tokens, secret variables, and private credentials:
- **API Keys / Secrets / Tokens:** **0 found.**
- **Private Keys / Certificates:** **0 found.**
- **Personal Information:** Standard Git committer identity (`tadejashwi07-coder` / `tadejashwi07@gmail.com`) and local filesystem paths (`C:\Users\lenovo\bb84-qgan-research`) appear in documentation and audit logs. No private contact details, credentials, or sensitive organizational secrets exist in the repository.

---

## 4. EXCLUDED FILES AND RATIONALE

A total of **49 files** are excluded by `.gitignore` and will **NOT** be committed:

| Category | File Count | Size | Exclusion Rationale |
| :--- | :--- | :--- | :--- |
| **Python Bytecode (`__pycache__/*`)** | 27 files | ~487 KB | Ephemeral compiled `.pyc` artifacts regenerated dynamically by Python 3.11. |
| **Matplotlib Runtime Cache** | 1 file | 141.7 KB | `results/fontlist-v3.11.0.json`: dynamically generated OS font cache from matplotlib plotting routines. |
| **Scratch & Audit Scripts (`scratch/*`)** | 21 files | ~110 KB | Intermediate audit scripts (`audit_cuda.py`, `compare_repos.py`, `migrate_to_original.py`, etc.) used during audit and verification phases. |

---

## 5. COMPLETE PROPOSED STAGING INVENTORY (84 FILES, 6.88 MB)

### 5.1 Core Production Source Code (`src/` — 20 files, 201.2 KB)
| File Path | Size (Bytes) | Formatted Size | Role |
| :--- | :--- | :--- | :--- |
| `src/__init__.py` | 43 | 43 B | Package root |
| `src/bb84/__init__.py` | 1,634 | 1.6 KB | BB84 module exports |
| `src/bb84/alice.py` | 7,687 | 7.5 KB | Alice state prep & RNG stream |
| `src/bb84/attacks.py` | 22,218 | 21.7 KB | Intercept-resend & unitary probe circuits |
| `src/bb84/bb84_engine.py` | 17,730 | 17.3 KB | Aer simulation runner & carrier circuits |
| `src/bb84/bob.py` | 18,958 | 18.5 KB | Bob measurement & basis choice |
| `src/bb84/channel.py` | 323 | 323 B | Channel base interface |
| `src/cgan/__init__.py` | 1,176 | 1.1 KB | cGAN module exports |
| `src/cgan/dataset.py` | 8,764 | 8.6 KB | PyTorch policy dataset loader & Wilson filter |
| `src/cgan/models.py` | 9,147 | 8.9 KB | Differentiable Generator $[0, \pi/4]$ & Discriminator |
| `src/cgan/policy_engine.py` | 9,180 | 9.0 KB | 16-candidate sampling & Fallback Tri-Case |
| `src/cgan/trainer.py` | 14,057 | 13.7 KB | WGAN-GP / cGAN training step & loss |
| `src/channel/__init__.py` | 2,158 | 2.1 KB | Channel module exports |
| `src/channel/noise_models.py` | 24,592 | 24.0 KB | Depolarizing, amplitude/phase damping models |
| `src/channel/quantum_channel.py` | 10,408 | 10.2 KB | Quantum channel wrapper |
| `src/dataset/__init__.py` | 1,183 | 1.2 KB | Dataset module exports |
| `src/dataset/generator.py` | 16,696 | 16.3 KB | Synthetic dataset generator |
| `src/dataset/policy_generator.py` | 9,850 | 9.6 KB | Attack configuration instantiator |
| `src/dataset/schema.py` | 21,772 | 21.3 KB | Pydantic records, environmental conditions schema |
| `src/dataset/statistics.py` | 8,428 | 8.2 KB | Wilson score confidence interval & feasibility |

### 5.2 Test Suite (`tests/` — 9 files, 117.8 KB)
| File Path | Size (Bytes) | Formatted Size | Role |
| :--- | :--- | :--- | :--- |
| `tests/__init__.py` | 49 | 49 B | Test suite root |
| `tests/test_alice.py` | 10,603 | 10.4 KB | Alice state prep tests |
| `tests/test_attacks.py` | 13,220 | 12.9 KB | Eve attack unitary tests |
| `tests/test_bb84_engine.py` | 6,705 | 6.5 KB | BB84 execution tests |
| `tests/test_bob.py` | 16,445 | 16.1 KB | Bob measurement tests |
| `tests/test_cgan_pipeline.py` | 26,233 | 25.6 KB | cGAN unit, fallback & CRN channel coupling tests |
| `tests/test_dataset_pipeline.py` | 21,248 | 20.8 KB | Dataset pipeline & schema tests |
| `tests/test_noise_models.py` | 15,494 | 15.1 KB | NISQ noise channel tests |
| `tests/test_noisy_channel.py` | 10,650 | 10.4 KB | Noisy transmission integration tests |

### 5.3 Experiment Runners (`experiments/` — 12 files, 175.0 KB)
| File Path | Size (Bytes) | Formatted Size | Role |
| :--- | :--- | :--- | :--- |
| `experiments/audit_full_dataset.py` | 12,974 | 12.7 KB | Full dataset verification utility |
| `experiments/audit_pilot.py` | 6,447 | 6.3 KB | Pilot training audit script |
| `experiments/audit_pilot_provenance.py` | 7,854 | 7.7 KB | Pilot condition provenance script |
| `experiments/benchmark_baselines.py` | 17,649 | 17.2 KB | 4-method baseline benchmark runner |
| `experiments/generate_full_dataset.py` | 10,015 | 9.8 KB | 1,152-record dataset generator |
| `experiments/generate_pilot_dataset.py` | 10,406 | 10.2 KB | Pilot dataset generator |
| `experiments/generate_pilot_provenance_dataset.py` | 653 | 653 B | Provenance verification dataset generator |
| `experiments/run_milestone1b_experiments.py` | 16,558 | 16.2 KB | Milestone 1B noise sweep runner |
| `experiments/run_milestone2_experiments.py` | 27,318 | 26.7 KB | Milestone 2 attack sweep runner |
| `experiments/train_cgan.py` | 7,358 | 7.2 KB | Legacy training script |
| `experiments/train_cgan_pilot.py` | 15,394 | 15.0 KB | 20-epoch pilot training script |
| `experiments/train_milestone4a.py` | 46,524 | 45.4 KB | Frozen 200-epoch Milestone 4A runner |

### 5.4 Model Checkpoints (`checkpoints/` — 4 files, 1,500.5 KB)
| File Path | Size (Bytes) | Formatted Size | Role |
| :--- | :--- | :--- | :--- |
| `checkpoints/cgan_best.pt` | 383,688 | 374.7 KB | Baseline best weights |
| `checkpoints/cgan_latest.pt` | 384,316 | 375.3 KB | Baseline latest weights |
| `checkpoints/cgan_pilot_best.pt` | 383,688 | 374.7 KB | Historical Epoch 16 pilot best |
| `checkpoints/cgan_pilot_latest.pt` | 384,856 | 375.8 KB | Historical Epoch 20 pilot latest |

### 5.5 Research Datasets (`results/` — 8 files, 3,111.3 KB)
| File Path | Size (Bytes) | Formatted Size | Role |
| :--- | :--- | :--- | :--- |
| `results/cgan_pilot_training_history.json` | 6,646 | 6.5 KB | Epoch 1–20 pilot training history |
| `results/cgan_pilot_validation_evaluation.json` | 41,483 | 40.5 KB | Epoch 16 pilot validation metrics |
| `results/full_dataset_benchmark_metrics.json` | 1,947 | 1.9 KB | Dataset benchmark metrics |
| `results/full_training_dataset.json` | 2,228,503 | 2.13 MB | Authoritative 1,152-record research dataset |
| `results/pilot_benchmark_metrics.json` | 2,396 | 2.3 KB | Pilot benchmark metrics |
| `results/pilot_benchmark_metrics_provenance.json` | 2,390 | 2.3 KB | Provenance benchmark metrics |
| `results/pilot_dataset.json` | 444,998 | 434.6 KB | 240-record pilot dataset |
| `results/pilot_dataset_provenance.json` | 457,625 | 446.9 KB | Provenance audit dataset |

### 5.6 Validation Results & Verification Logs (`results/` — 6 files, 39.5 KB)
| File Path | Size (Bytes) | Formatted Size | Role |
| :--- | :--- | :--- | :--- |
| `results/milestone1b_experimental_results.json` | 7,691 | 7.5 KB | Milestone 1B experimental data |
| `results/milestone2_experimental_results.json` | 21,521 | 21.0 KB | Milestone 2 experimental data |
| `results/milestone4a/checkpoint_selection_verification.json` | 3,531 | 3.4 KB | Non-scalarized Pareto checkpoint selection log |
| `results/milestone4a/data_isolation_verification.json` | 2,395 | 2.3 KB | C21–C24 partition quarantine log |
| `results/milestone4a/execution_estimate.json` | 2,622 | 2.6 KB | Runtime & memory profiling estimate |
| `results/milestone4a/frozen_training_config.json` | 2,674 | 2.6 KB | Frozen 200-epoch hyperparameter config |

### 5.7 Generated Sweep Figures (`results/` — 8 files, 1,575.8 KB)
| File Path | Size (Bytes) | Formatted Size | Role |
| :--- | :--- | :--- | :--- |
| `results/exp1_depolarizing_sweep.png` | 210,382 | 205.5 KB | Experiment 1 depolarizing curve |
| `results/exp2_amplitude_damping_sweep.png` | 228,802 | 223.4 KB | Experiment 2 amplitude damping curve |
| `results/exp3_partial_intercept_resend.png` | 187,996 | 183.6 KB | Experiment 3 PIR trade-off curve |
| `results/exp3_phase_damping_sweep.png` | 192,420 | 187.9 KB | Experiment 3 phase damping curve |
| `results/exp4_basis_bias_analysis.png` | 226,587 | 221.3 KB | Experiment 4 basis bias analysis |
| `results/exp4_thermal_relaxation.png` | 246,109 | 240.3 KB | Experiment 4 thermal relaxation curve |
| `results/exp5_noise_vs_eavesdropping.png` | 138,871 | 135.6 KB | Experiment 5 noise vs eavesdropping |
| `results/exp5_unitary_probe_tradeoff.png` | 182,483 | 178.2 KB | Experiment 5 probe trade-off curve |

### 5.8 Documentation & Reports (16 files, 321.6 KB)
| File Path | Size (Bytes) | Formatted Size | Role |
| :--- | :--- | :--- | :--- |
| `FINAL_REPORT.md` | 13,162 | 12.9 KB | Cumulative research report |
| `M3_STAGE_A_REVISED_REPORT.md` | 48,029 | 46.9 KB | M3 Stage A design report |
| `M3_STAGE_B_FINAL_ACCEPTANCE_REPORT.md` | 21,801 | 21.3 KB | M3 Stage B acceptance report |
| `M3_STAGE_C_FINAL_PRETRAINING_AUDIT_REPORT.md` | 21,316 | 20.8 KB | Stage C audit report |
| `M3_STAGE_C_FINAL_TRAINING_AUTHORIZATION_REPORT.md` | 20,593 | 20.1 KB | Stage C authorization review |
| `M3_STAGE_C_PILOT_TRAINING_REPORT.md` | 18,346 | 17.9 KB | 20-epoch pilot empirical results |
| `M3_STAGE_C_POST_PILOT_AUDIT_REPORT.md` | 23,873 | 23.3 KB | Post-pilot constraint audit report |
| `M3_STAGE_C_TRAINING_READINESS_REPORT.md` | 19,949 | 19.5 KB | Pre-training readiness review |
| `MILESTONE_2_REPORT.md` | 30,655 | 29.9 KB | Milestone 2 attack model report |
| `PILOT_VALIDATION_REPORT.md` | 26,090 | 25.5 KB | Validation metrics report |
| `qkd_cqgan_engineering_plan.md` | 17,959 | 17.5 KB | Comprehensive engineering architecture |
| `docs/M3_STAGE_C_FINAL_CLOSURE_REPORT.md` | 11,265 | 11.0 KB | Final closure report with CRN coupling |
| `docs/M3_STAGE_C_FINAL_PRETRAINING_AUDIT_REPORT.md` | 18,295 | 17.9 KB | Mirrored stage C audit report |
| `docs/M4A_FINAL_EXECUTION_AUTHORIZATION_REVIEW.md` | 17,624 | 17.2 KB | Pre-execution configuration review |
| `docs/migration_log.json` | 8,486 | 8.3 KB | Repository consolidation migration log |
| `docs/REPOSITORY_CONSOLIDATION_REPORT.md` | 11,892 | 11.6 KB | Consolidation verification report |

### 5.9 Repository Configuration (1 file, 288 B)
| File Path | Size (Bytes) | Formatted Size | Role |
| :--- | :--- | :--- | :--- |
| `.gitignore` | 288 | 288 B | Ignores `__pycache__/`, `scratch/`, `results/fontlist-*.json` |

---

## 6. SECONDARY WORKTREE BACKUP STATUS

- **Secondary Worktree Path:** `C:\Users\lenovo\.gemini\antigravity\worktrees\bb84-qgan-research\nisq_noise_modelling`
- **Total Files Audited:** 84 / 84
- **Identical Hash Matches:** **84 / 84 (100.00%)**
- **Missing Files:** 0
- **Hash Mismatches:** 0
- **Integrity Statement:** The secondary worktree contains an exact byte-for-byte duplicate of all candidate staged files, serving as an uncorrupted secondary backup.

---

## 7. PROPOSED COMMIT COMMAND

Upon explicit user authorization, the proposed staging and initial commit command will be:

```powershell
Set-Location "C:\Users\lenovo\bb84-qgan-research"
git add .
git commit -m "feat: initial scientific baseline for adaptive QGAN eavesdropping research"
```

No git staging or commit command has been executed. Awaiting explicit user approval.

