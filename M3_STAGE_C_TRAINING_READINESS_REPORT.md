# MILESTONE 3 — STAGE C: DATASET COMPLETION GATE AND TRAINING READINESS AUDIT REPORT

**Project:** Adaptive QGAN-Based Eavesdropping in BB84 QKD: Resource-Constrained Conditional Quantum Generative Approach under NISQ Noise  
**Stage:** Milestone 3 — Stage C (Classical Conditional GAN)  
**Date & Timestamp:** 2026-11-10T17:55:00+05:30  
**Repository Worktree:** `C:\Users\lenovo\.gemini\antigravity\worktrees\bb84-qgan-research\nisq_noise_modelling`  
**Main Repository:** `C:\Users\lenovo\bb84-qgan-research`  
**Hardware Profile:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM), CUDA 12.1, PyTorch 2.5.1+cu121, Python 3.11.9, Qiskit 2.6.0, Qiskit Aer 0.18.0  
**Audit Status:** **ALL CHECKS PASSED (17/17 Audit Checks, 139/139 Unit Tests Passed)**  
**Readiness State:** **READY FOR TRAINING AUTHORIZATION — EXECUTION HALTED (AWAITING USER APPROVAL)**

---

## 1. Executive Summary and Authorization Status

Following formal authorization of Milestone 3 Stage C, the complete 24-condition primary grid simulation dataset (1,152 physical simulation records, 2,000 optical pulses each) was generated, validated, and subjected to a comprehensive scientific and data leakage audit. 

All core cGAN architecture components, dataset samplers, trainers, policy engines, and baseline benchmark runners have been implemented and validated against the approved Milestone 3 Stage A/B specifications. In strict compliance with instructions, **training has NOT been initiated**, benchmarking has NOT been executed, and no quantum GAN (M4) components have been implemented.

| Audit Category | Specification / Expectation | Measured / Verified | Audit Status |
| :--- | :--- | :--- | :---: |
| **Dataset Completeness** | 1,152 records (24 cond $\times$ 16 att $\times$ 3 seeds) | Exactly 1,152 records | **PASS** |
| **Simulation Pulses** | 2,000 pulses per record | 2,304,000 carrier pulses | **PASS** |
| **Condition Partitioning** | 16 Train / 4 Val / 4 Test (Zero Leakage) | Disjoint partitions verified | **PASS** |
| **Data Schema Integrity** | 0 missing fields, 0 duplicate sample IDs | 0 missing, 0 duplicates | **PASS** |
| **Statistical Calculations** | Wilson score intervals & sample-QBER rule | 100% independent match | **PASS** |
| **Information Boundary** | Strictly $c = [Q_{\text{budget}}, \hat{\gamma}_{\text{noise}}]$ | Hidden $p, Q_{\text{env}}$ excluded | **PASS** |
| **cGAN Architecture** | Differentiable masking, normalized outputs | 29,446 total parameters | **PASS** |
| **Loss & Objective** | Minimax GAN + density importance sampling | Non-saturating BCE loss | **PASS** |
| **Baseline Fairness** | Equal query budget $K=16$ for all 4 methods | Strict $K=16$ invariant | **PASS** |
| **Hardware Readiness** | GPU acceleration, Ampere capability (8, 6) | CUDA verified (4.21 ms/batch) | **PASS** |
| **Regression Stability** | 100% test pass rate across repository | 139/139 passed (96.4s) | **PASS** |

---

## 2. TASK 1: Full Dataset Completion & Validation Audit

The primary dataset generation script (`experiments/generate_full_dataset.py`) executed to completion with **exit code 0** (Task ID: `8d4550a6-7714-4940-8ded-4e7a6419f1f3/task-1360`).

### 2.1 Generation Performance and Hardware Telemetry
- **Total Execution Time:** 756.64 seconds (12.61 minutes).
- **Average Time per Run:** 0.6568 seconds per physical simulation.
- **Physical Carrier Pulses Simulated:** 2,304,000 pulses.
- **Simulation Throughput:** 3,045.0 pulses / second.
- **Initial Process RAM:** 399.68 MB $\to$ **Final Process RAM:** 525.03 MB ($\Delta = 125.34\text{ MB}$).
- **Peak VRAM Allocated during Simulation:** 0.0 MB (Aer simulation correctly executed on host CPU multi-threaded OpenMP backend).
- **Output Artifact:** `results/full_training_dataset.json` (2,176.27 KB).
- **Benchmark Metrics File:** `results/full_dataset_benchmark_metrics.json`.

### 2.2 Record Inventory and Partition Breakdown
The dataset contains exactly 1,152 unique simulation records across 24 environmental conditions (C01 to C24), 16 attack configurations (ATT-00 to ATT-15), and 3 independent simulator seeds (101, 102, 103).

```
Total Records: 1,152
├── Train Partition: 768 records (16 conditions x 16 attacks x 3 seeds)
├── Validation Partition: 192 records (4 conditions x 16 attacks x 3 seeds)
└── Test Partition: 192 records (4 conditions x 16 attacks x 3 seeds)
```

| Attack Family | Attack IDs | Count per Condition | Total Records | Share |
| :--- | :--- | :---: | :---: | :---: |
| **Null Attack** | ATT-00 | 3 | 72 | 6.25% |
| **Unitary Probe** | ATT-01 to ATT-07 | 21 | 504 | 43.75% |
| **Probabilistic Intercept-Resend (PIR)** | ATT-08 to ATT-15 | 24 | 576 | 50.00% |
| **Total** | ATT-00 to ATT-15 | 48 | 1,152 | 100.0% |

### 2.3 Statistical Feasibility and Protocol Abort Yield
In strict compliance with the Stage A mathematical specification:
- **Statistical Feasibility** is defined by the two-sided 95% Wilson score confidence interval calculated on $N_{\text{test}}$ sifted test bits:
  - **Feasible (Case A candidate):** $Q_{\text{UCB}} \le Q_{\text{budget}}$
  - **Infeasible (Case B fallback):** $Q_{\text{LCB}} > Q_{\text{budget}}$
  - **Indeterminate:** $Q_{\text{LCB}} \le Q_{\text{budget}} < Q_{\text{UCB}}$
- **Protocol Abort Outcome** is defined by Alice and Bob's operational decision rule: $\text{Abort} \iff Q_{\text{test}} > Q_{\text{budget}}$.

| Partition | Total Runs | Feasible ($Q_{\text{UCB}} \le Q_{\text{bud}}$) | Indeterminate | Infeasible ($Q_{\text{LCB}} > Q_{\text{bud}}$) | Protocol Aborts | Protocol Passes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (C01–C16)** | 768 | 368 (47.92%) | 249 (32.42%) | 151 (19.66%) | 252 (32.81%) | 516 (67.19%) |
| **Val (C17–C20)** | 192 | 99 (51.56%) | 62 (32.29%) | 31 (16.15%) | 65 (33.85%) | 127 (66.15%) |
| **Test (C21–C24)** | 192 | 79 (41.15%) | 61 (31.77%) | 52 (27.08%) | 84 (43.75%) | 108 (56.25%) |
| **Overall Dataset** | **1,152** | **546 (47.40%)** | **372 (32.29%)** | **234 (20.31%)** | **401 (34.81%)** | **751 (65.19%)** |

### 2.4 Integrity and Audit Verification Results
The standalone audit script (`experiments/audit_full_dataset.py`) verified all 1,152 records:
1. **Total Record Count:** Exactly 1,152 records.
2. **Duplicate IDs:** 0 duplicates; all sample IDs follow `M3-{COND_ID}-{ATT_ID}-S{SEED}`.
3. **Missing Fields:** 0 missing values; schema version `1.0.0` validated.
4. **Physical Parameters:** 
   - Null attack: $\alpha = \text{None}, \beta = \text{None}, \theta_E = 0.0$.
   - PIR attacks: $\alpha \in [0.10, 0.90] \subset [0, 1]$, $\beta \in [0.10, 0.90] \subset [0, 1]$, $\theta_E = \text{None}$.
   - Probe attacks: $\theta_E \in [0.08, 0.52] \text{ rad} \subset [0, \pi/4]$, $\alpha = \text{None}, \beta = \text{None}$.
5. **Calibration Provenance:** Every condition includes $N_{\text{calib}} = 1000$, calibration error count $k_{\text{calib}}$, observable estimate $\hat{\gamma}_{\text{noise}} = k_{\text{calib}} / N_{\text{calib}}$, and dedicated calibration seed.
6. **Wilson Confidence Bounds:** 100% of center, LCB, UCB, and margin values match independent reference recalculations to $< 10^{-5}$ tolerance.
7. **Independence of Aborts:** 0 records violate $Q_{\text{test}} > Q_{\text{budget}} \iff \text{Abort}$. Note that 401 runs aborted while 234 were statistically infeasible, demonstrating the physical divergence between finite-sample sample-QBER decisions and asymptotic confidence bounds.
8. **Information Boundary:** For every record, `get_observable_condition()` returns strictly $c = (Q_{\text{budget}}, \hat{\gamma}_{\text{noise}})$. Microscopic channel parameters ($p_{\text{depol}}, Q_{\text{env}}$) are strictly quarantined.

---

## 3. TASK 2: Zero-Leakage Partition Audit

Condition-level partitioning enforces that the generator and discriminator are trained and validated on disjoint physical environments, with test conditions completely held out for benchmark evaluation.

### 3.1 Condition Partition Mapping
- **Disjointness Invariant:** $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{val}} \cap \mathcal{C}_{\text{test}} = \emptyset$ (Verified: **TRUE**).

```
Condition Allocation:
├── Train (16 conditions, 768 records):
│   C01: p=0.00, Q_bud=0.05    C02: p=0.00, Q_bud=0.08    C03: p=0.00, Q_bud=0.11
│   C04: p=0.02, Q_bud=0.05    C05: p=0.02, Q_bud=0.08    C06: p=0.02, Q_bud=0.11
│   C07: p=0.04, Q_bud=0.05    C08: p=0.04, Q_bud=0.08    C09: p=0.04, Q_bud=0.11
│   C10: p=0.06, Q_bud=0.05    C11: p=0.06, Q_bud=0.08    C12: p=0.06, Q_bud=0.11
│   C13: p=0.01, Q_bud=0.08    C14: p=0.03, Q_bud=0.08    C15: p=0.05, Q_bud=0.08
│   C16: p=0.07, Q_bud=0.08
├── Validation (4 conditions, 192 records) [Interpolation Check]:
│   C17: p=0.01, Q_bud=0.05    C18: p=0.01, Q_bud=0.11
│   C19: p=0.03, Q_bud=0.05    C20: p=0.03, Q_bud=0.11
└── Test (4 conditions, 192 records) [Held-Out Extrapolation Check]:
    C21: p=0.05, Q_bud=0.05    C22: p=0.05, Q_bud=0.11
    C23: p=0.07, Q_bud=0.05    C24: p=0.07, Q_bud=0.11
```

### 3.2 Preprocessing and Data Leakage Quarantine
1. **Zero Fitted Preprocessing Parameters:** Normalization of the continuous policy vector is strictly analytic and fixed closed-form:
   $$\tilde{\theta}_E = \theta_E \times \frac{4}{\pi} \in [0, 1], \quad \alpha \in [0, 1], \quad \beta \in [0, 1]$$
   No empirical mean, variance, or quantile statistics are fitted from any dataset partition.
2. **Training Quarantine:** Model training draws exclusively from `train_loader`, which filters records where `partition == "train"` and `feasibility_category == FEASIBLE` (368 feasible exemplars).
3. **Validation Quarantine:** Best-checkpoint selection monitors validation loss exclusively on `val_loader` (C17–C20).
4. **Test Quarantine:** Conditions C21–C24 are never seen during training, backpropagation, or checkpoint saving; they are reserved strictly for downstream evaluation in `experiments/benchmark_baselines.py`.

---

## 4. TASK 3: cGAN Architecture and Mathematical Audit

All cGAN modules have been inspected in `src/cgan/models.py`, `src/cgan/trainer.py`, `src/cgan/dataset.py`, `src/cgan/policy_engine.py`, and `experiments/train_cgan.py`.

### 4.1 Generator Architecture (`ConditionalGenerator`)
- **Condition Vector:** $c = [Q_{\text{budget}}, \hat{\gamma}_{\text{noise}}] \in \mathbb{R}^2$
- **Latent Prior:** $z \sim \mathcal{N}(0, I_{16}) \in \mathbb{R}^{16}$
- **Shared Trunk:** Linear(18 $\to$ 128) $\to$ LayerNorm $\to$ LeakyReLU(0.2) $\to$ Linear(128 $\to$ 128) $\to$ LayerNorm $\to$ LeakyReLU(0.2) $\to$ Dropout(0.1)
- **Multi-Head Output:**
  1. *Family Head:* Linear(128 $\to$ 2) $\to$ Gumbel-Softmax($\tau$) during training, Softmax during sampling $\implies \mathbf{w} = [w_{\text{PIR}}, w_{\text{Probe}}] \in \Delta^1$.
  2. *PIR Head:* Linear(128 $\to$ 2) $\to$ $\text{Sigmoid} \implies [\alpha, \beta] \in [0, 1]^2$.
  3. *Probe Head:* Linear(128 $\to$ 1) $\to$ $\text{Sigmoid} \times \frac{\pi}{4} \implies \theta_E \in [0, \pi/4]$.
- **Canonical Masking Operator:**
  $$\mathbf{a}_{\text{masked}} = \begin{bmatrix} w_{\text{PIR}} \\ w_{\text{Probe}} \\ w_{\text{PIR}} \cdot \alpha \\ w_{\text{PIR}} \cdot \beta \\ w_{\text{Probe}} \cdot \tilde{\theta}_E \end{bmatrix} \in \mathbb{R}^5$$
  *Mathematical Guarantee:* When $w_{\text{PIR}} \to 1$ ($w_{\text{Probe}} \to 0$), $\frac{\partial \mathbf{a}_{\text{masked}}}{\partial \theta_E} = 0$, ensuring zero gradient leakage into inactive parameter branches.

### 4.2 Discriminator Architecture (`ConditionalDiscriminator`)
- **Input Vector:** $[\mathbf{a}_{\text{masked}}, c] \in \mathbb{R}^7$ (5 policy dimensions + 2 condition dimensions)
- **Layers:** 
  $$\text{spectral\_norm}(\text{Linear}(7 \to 128)) \to \text{LeakyReLU}(0.2) \to \text{spectral\_norm}(\text{Linear}(128 \to 64)) \to \text{LeakyReLU}(0.2) \to \text{spectral\_norm}(\text{Linear}(64 \to 1))$$
- **Lipshitz Bound:** Spectral normalization enforces 1-Lipschitz continuity, eliminating mode collapse and discriminator gradient explosion under small batch sizes.

### 4.3 Objective Function and Mutual Information Importance Sampling
The user specifically requested verification of the training objective and clarification of mutual-information importance sampling:

> **Audit Finding:** The cGAN is **NOT** a direct gradient optimizer of Eve's mutual information $I(A; E)$.

1. **Approved Minimax Distribution-Matching Objective:**
   The neural network loss functions are purely the standard non-saturating conditional GAN losses:
   $$\mathcal{L}_D = -\mathbb{E}_{(\mathbf{a}, c) \sim \tilde{p}_{\text{data}}} \left[\log \sigma(D(\mathbf{a}, c))\right] - \mathbb{E}_{z \sim p_z, c \sim p_c} \left[\log (1 - \sigma(D(G(z | c), c)))\right]$$
   $$\mathcal{L}_G = -\mathbb{E}_{z \sim p_z, c \sim p_c} \left[\log \sigma(D(G(z | c), c))\right]$$
   Both losses use `nn.BCEWithLogitsLoss()`.
2. **How Mutual Information Is Incorporated (Importance Sampling):**
   In `src/cgan/dataset.py`, each feasible training exemplar $i$ is assigned a static sampling weight:
   $$w_i = \exp\left(\frac{I(A; E)_i}{\tau_I}\right), \quad \tau_I = 0.10$$
   PyTorch's `WeightedRandomSampler` uses these weights to construct training mini-batches.
3. **Scientific Implication:**
   - Importance sampling modifies the empirical data density $\tilde{p}_{\text{data}}(\mathbf{a} | c) \propto p_{\text{feasible}}(\mathbf{a} | c) \cdot \exp(I(A; E) / \tau_I)$.
   - The generator is trained to match this re-weighted distribution, naturally generating candidates that concentrate near the high-information feasible frontier.
   - Because $I(A; E)$ is not inside the backpropagation graph of $G$, the generator cannot exploit gradient artifacts or adversarial boundary vulnerabilities, preserving the exact Stage A probabilistic formulation.

---

## 5. TASK 4: Baseline Fairness Verification

`experiments/benchmark_baselines.py` was inspected to confirm strict fairness across all four evaluated strategies.

| Fairness Invariant | Requirement | Implementation in `benchmark_baselines.py` | Verification |
| :--- | :--- | :--- | :---: |
| **Identical Condition Vector** | Same $c = [Q_{\text{budget}}, \hat{\gamma}_{\text{noise}}]$ | Passed directly to each engine via `EnvironmentalCondition` | **VERIFIED** |
| **Identical Query Budget** | Exactly $K = 16$ candidate forward simulations | Verified invariant `num_candidates = 16` across all engines | **VERIFIED** |
| **Identical Physical Simulator** | Same Aer engine, 2,000 pulses, same noise | All use `execute_single_simulation` ($N_{\text{pulse}} = 2000$) | **VERIFIED** |
| **Identical Feasibility Filter** | 95% Wilson score interval $Q_{\text{UCB}} \le Q_{\text{budget}}$ | All use identical `feasibility_category == FEASIBLE` | **VERIFIED** |
| **Identical Evaluation Seeds** | 5 independent seeds (901 to 905) | `eval_seeds = (901, 902, 903, 904, 905)` | **VERIFIED** |
| **Identical Decision Logic** | Case A (max $I_{AE}$), Case B (Null), Case C (Abort) | Identical Tri-Case fallback routine across all 4 methods | **VERIFIED** |
| **Test Condition Quarantine** | Evaluated only on held-out test set | Evaluated strictly on conditions C21, C22, C23, C24 | **VERIFIED** |

### Evaluated Methods:
1. **cGAN Policy Engine:** Proposes $K=16$ candidates from trained generator $G_\phi(z | c)$, filters feasible candidates, selects policy maximizing $I(A; E)$ (else Null fallback).
2. **Uniform Random Search:** Uniformly samples $K=16$ attack configurations across continuous parameter space ($50\%$ PIR, $50\%$ Probe), filters feasible candidates, selects policy maximizing $I(A; E)$ (else Null fallback).
3. **Deterministic Grid Search:** Evaluates the 16 canonical lattice configurations (ATT-00 to ATT-15), filters feasible candidates, selects policy maximizing $I(A; E)$ (else Null fallback).
4. **Analytical Probe Heuristic:** Calculates closed-form target angle $\theta_E^* = \frac{1}{2} \arcsin\left(\sqrt{2 \max\left(0, \frac{Q_{\text{budget}} - \hat{\gamma}_{\text{noise}}}{1 - 2\hat{\gamma}_{\text{noise}}}\right)}\right)$, generates 16 local perturbations across $[-0.15, +0.15]$, filters feasible candidates, selects policy maximizing $I(A; E)$ (else Null fallback).

---

## 6. TASK 5: GPU Profiling and Training Readiness

Hardware profiling was conducted using `experiments/train_cgan.py` configuration checks and dedicated PyTorch CUDA benchmarks.

### 6.1 Hardware and Runtime Telemetry
- **CUDA Device:** `NVIDIA GeForce RTX 3050 6GB Laptop GPU`
- **CUDA Capability:** Ampere (8, 6)
- **Total Dedicated VRAM:** 6.00 GB (5,984 MB available)
- **Current VRAM Allocation:** 0.00 MB (idle)
- **Trainable Parameters:**
  - Generator: 20,101 parameters
  - Discriminator: 9,345 parameters
  - Total Model Parameters: 29,446 parameters
- **Batch Size:** 64
- **Training Set Size:** 368 feasible records $\implies 6$ mini-batches per epoch.
- **Empirical GPU Latency:** 4.21 ms per batch forward/backward pass.
- **Estimated 200-Epoch Training Runtime:** Approximately 15 to 25 seconds.
- **Checkpoint Paths:** `checkpoints/cgan_best.pt` and `checkpoints/cgan_latest.pt`.
- **Reproducibility Configuration:** Fixed seed 42 set for `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`, and Python RNG.

---

## 7. Main Repository Regression Test Audit

A complete discovery regression run was executed across all test suites in the repository (`python -m unittest discover tests`):

```
...........................................................................................................................................
----------------------------------------------------------------------
Ran 139 tests in 96.400s

OK
```

### Module Breakdown:
1. `tests/test_alice.py`: 12 tests passed (Alice state preparation, basis randomization, and encoding).
2. `tests/test_bob.py`: 18 tests passed (Bob basis selection, measurement operators, detection statistics).
3. `tests/test_noisy_channel.py`: 14 tests passed (Depolarizing channel operators and Kraus representation).
4. `tests/test_bb84_engine.py`: 11 tests passed (End-to-end BB84 protocol coordination, sifting, error estimation).
5. `tests/test_attacks.py`: 24 tests passed (PIR attack, Fuchs-Peres-Brandt probe, basis-conditioned disturbance).
6. `tests/test_noise_models.py`: 18 tests passed (Noise channel models, ambient calibration, Kraus matrices).
7. `tests/test_dataset_pipeline.py`: 28 tests passed (Dataset schema, serialization, Wilson intervals, feasibility).
8. `tests/test_cgan_pipeline.py`: 14 tests passed (Masking, multi-head generator, spectral discriminator, trainer, policy engine).

**Final Regression Result:** **139 passed, 0 failed, 0 errors, 0 skipped (100% PASS RATE)**.

---

## 8. Historical Archival Integrity & Classification

In strict compliance with user instructions:
- Pre-Stage-B historical files without independently recoverable cryptographic snapshots remain classified as **`UNVERIFIED`**.
- Historical pilot files (`results/pilot_dataset.json`, `results/pilot_dataset_provenance.json`) are preserved byte-for-byte and have not been modified or overwritten.
- All code, dataset files, experiment scripts, and tests are synchronized between the workspace worktree and the main repository.

---

## 9. Conclusion and Next Steps

The Stage C infrastructure is completely implemented, verified, and audited:
1. Full 1,152-record primary dataset generated and verified with 0 defects.
2. Zero data leakage across condition partitions.
3. cGAN multi-head architecture and importance sampling objective verified.
4. Equal $K=16$ query budget benchmark framework ready.
5. All 139 unit tests passing.

**STOPPING POINT:** Per instructions, execution is now **HALTED**. We await your review and formal authorization before launching the 200-epoch cGAN training run (`python experiments/train_cgan.py`) and comparative baseline benchmarking (`python experiments/benchmark_baselines.py`).

