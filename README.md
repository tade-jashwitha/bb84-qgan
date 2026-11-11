# Adaptive QGAN-Based Eavesdropping in BB84 QKD Under NISQ Noise

[![Tests](https://img.shields.io/badge/Unit%20Tests-160%2F160%20Passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

A rigorous, end-to-end quantum simulation and generative adversarial machine learning research platform investigating resource-constrained eavesdropping in the BB84 Quantum Key Distribution (QKD) protocol under realistic Noisy Intermediate-Scale Quantum (NISQ) channel conditions.

---

## 1. Research Abstract

Quantum Key Distribution guarantees unconditional cryptographic security rooted in the fundamental laws of quantum mechanics—specifically the Heisenberg uncertainty principle and the no-cloning theorem. However, physical QKD implementations operate over noisy communication channels and lossy optical fibers where environmental perturbations induce a baseline Quantum Bit Error Rate (QBER). An advanced eavesdropper (Eve) can strategically hide disturbance beneath this environmental noise floor.

This project investigates whether conditional generative adversarial networks can learn optimal, adaptive eavesdropping strategies without triggering protocol aborts. We implement a high-fidelity BB84 simulation engine, model four distinct NISQ noise channels, construct a 24,000-sample offline policy dataset, train a classical conditional GAN (cGAN) with physical Pareto checkpointing, and execute a rigorous four-method comparative benchmark across quarantined held-out conditions ($C21$–$C24$). Our empirical findings reveal that while the classical cGAN achieves high statistical feasibility ($95.0\%$), its conservative mode behavior yields lower mutual information ($0.0451$ bits) than domain-tailored analytical heuristics ($0.0861$ bits) and grid search ($0.0760$ bits), providing empirical motivation for variational quantum generative circuits (QGAN).

---

## 2. Research Objectives

1. **High-Fidelity Quantum Simulation:** Construct a statevector-level 2D Hilbert space simulator of the BB84 protocol with Alice state preparation, quantum channel transmission, Bob projective measurement, and authenticated classical sifting.
2. **Comprehensive NISQ Noise Modeling:** Simulate four parameterized physical noise channels: depolarizing, amplitude damping, phase damping, and thermal relaxation ($T_1/T_2$).
3. **Physical Attack Formulations:** Model Partial/Probabilistic Intercept-Resend (PIR) and Fuchs-Peres-Brandt coherent unitary entangling probe attacks with exact disturbance derivations.
4. **Offline Policy Dataset Generation:** Build an authoritative 24-condition environmental matrix ($C01$–$C24$) and 16-lattice policy space under strict Common Random Numbers (CRN) and Wilson-score confidence bounds.
5. **Generative Adversarial Optimization:** Train a classical conditional GAN baseline ($G: \mathbb{R}^{16} \times \mathbb{R}^2 \to \mathbb{R}^3$) using continuous Gumbel-Softmax policy relaxations and multi-objective Pareto checkpoint selection.
6. **Comparative Benchmarking:** Evaluate the trained cGAN against Uniform Random Search, Deterministic Grid Search, and an Analytical Probe Heuristic under an identical query budget ($K=16$).
7. **Empirical Grounding for QGAN:** Rigorously analyze classical limitations to establish design criteria for Parameterized Quantum Circuit (PQC) generators in Milestone 4B.

---

## 3. BB84 and NISQ Simulation Architecture

The quantum physical foundation resides in `src/bb84/` and `src/channel/`:

```
+-------------------------------------------------------------------------------+
| Alice State Prep (Z & X bases)                                                |
|   |0>, |1> (Rectilinear);  |+>, |-> (Diagonal)                                |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| Quantum Channel (Parametric NISQ Noise)                                        |
|   - Depolarizing:  E(rho) = (1-p)rho + (p/3)[X rho X + Y rho Y + Z rho Z]     |
|   - Amplitude Damping: Kraus K0, K1 (Energy relaxation / loss)                 |
|   - Phase Damping: Kraus K0, K1 (Pure dephasing)                               |
|   - Thermal Relaxation: Combined T1/T2 Lindblad master equation               |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| Eavesdropping Attack (Eve Interaction)                                         |
|   - Partial Intercept-Resend: p_int in [0, 1]                                 |
|   - Unitary Entangling Probe: U_BE |psi>|w>, D(theta_E) = sin^2(theta_E)      |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
| Bob Measurement & Sifting Protocol                                            |
|   - Projective measurements in random bases                                   |
|   - Public classical reconciliation: sifted key fraction ~ 0.50               |
|   - QBER estimation: sample error rate E = N_err / N_sample                   |
|   - Abort Logic: IF E >= QBER_tol THEN Abort ELSE Privacy Amplification       |
+-------------------------------------------------------------------------------+
```

---

## 4. Classical Conditional GAN Architecture

Implemented in `src/cgan/models.py` and trained via `experiments/train_classical_cgan.py`:

- **Generator ($G$):** Multi-Layer Perceptron taking latent vector $\vec{z} \sim \mathcal{N}(0, I_{16})$ concatenated with environmental condition vector $\vec{c} = [p_{\text{noise}}, QBER_{\text{tol}}]^T \in \mathbb{R}^2$. Hidden dimensions: `[128, 128]` with LeakyReLU activations. Output heads produce continuous parameter proposals ($p_{\text{int}} \in [0, 0.40]$, $\theta_E \in [0, \pi/4]$) and categorical attack family logits passed through Gumbel-Softmax with temperature annealing $\tau_0 = 1.0 \to \tau_{\min} = 0.20$.
- **Discriminator ($D$):** Evaluates joint candidate-condition pairs $(\vec{a}, \vec{c})$ outputting validity and realism scores.
- **Pareto Checkpoint Engine:** Online multi-objective tracking maintains non-dominated checkpoints across 200 epochs:
  - `cgan_m4a_lexicographic_best.pt`: Primary operational model maximizing feasible mutual information $\bar{I}_{AE}$ subject to $Y_{\text{feas}} \ge 0.80$ and zero observed aborts (Selected at Epoch 150).
  - `cgan_m4a_max_feasibility.pt`: Anchor maximizing feasibility yield.
  - `cgan_m4a_max_information.pt`: Anchor maximizing mutual information.
  - `cgan_m4a_balanced_pareto.pt`: Scalar compromise maximizing $Y_{\text{feas}} \times \bar{I}_{AE}$.

---

## 5. Dataset Description

The authoritative dataset (`results/full_training_dataset.json`, SHA-256: `07C03E810470B3376EB48C51C13138DCF347D915F78E911F74A81E96F27CFB80`) contains 24,000 physical simulation records generated using `src/dataset/generator.py`:

- **Condition Matrix ($C01$–$C24$):**
  - Depolarizing noise $p \in [0.00, 0.08]$
  - Dephasing parameter $\lambda \in [0.00, 0.08]$
  - SPAM error $\epsilon \in [0.00, 0.03]$
  - Abort threshold $QBER_{\text{tol}} \in [0.08, 0.15]$
- **Strict Data Partitions:**
  - **Training Partition (C01–C16):** 16 conditions for generator/discriminator optimization.
  - **Validation Partition (C17–C20):** 4 conditions for epoch evaluation and checkpoint selection.
  - **Held-Out Test Partition (C21–C24):** 4 conditions completely quarantined for final benchmarking.
- **Statistical Feasibility Criterion:**
  $$Y_{\text{feas}} = \mathbb{I}\left(\text{UCB}_{95\%}(QBER) \le QBER_{\text{tol}}\right)$$

---

## 6. Benchmark Methodology

The four-method comparative benchmark is executed via `experiments/benchmark_baselines.py`:

- **Held-Out Conditions:** Quarantined conditions $C21, C22, C23, C24$.
- **Episodes:** 5 independent evaluation seeds per condition (20 episodes total per method).
- **Query Budget:** $K=16$ physical candidate evaluations per episode ($16 \times 80 = 1,280$ physical candidate evaluations across the benchmark).
- **Physical Validation:** $N=2,000$ quantum pulses per candidate simulation.
- **Evaluated Engines:**
  1. `cGAN_Policy_Engine`: Sampling 16 latent vectors through the Epoch 150 generator.
  2. `Uniform_Random_Search`: Uniform sampling across physical parameter boundaries.
  3. `Deterministic_Grid_Search`: $4 \times 4$ regular lattice grid.
  4. `Analytical_Probe_Heuristic`: Direct theoretical inversion $\theta_E^* = \arcsin(\sqrt{D^*})$ calibrated to estimated channel noise.

---

## 7. Actual Benchmark Results

Evaluated across all 80 episodes and 1,280 candidate simulations (`results/cgan_benchmark_comparison.json`):

| Policy Engine | Statistical Feasibility | Empirical Abort Rate | Mean Eve Info $\bar{I}_{AE}$ | Median $\bar{I}_{AE}$ | Max $\bar{I}_{AE}$ | Active Feasible (Case A) | Mean Proposal Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Analytical Probe Heuristic** | **100.0%** (20/20) | **0.0%** | **0.08611** | **0.08897** | 0.17786 | 19 / 20 | 1.10 ms |
| **Deterministic Grid Search** | **100.0%** (20/20) | **0.0%** | **0.07595** | **0.07050** | 0.17282 | 19 / 20 | 0.01 ms |
| **cGAN Policy Engine (Epoch 150)** | **95.0%** (19/20) | **0.0%** | **0.04512** | **0.04775** | 0.13100 | 18 / 20 | 803.82 ms |
| **Uniform Random Search** | **85.0%** (17/20) | **0.0%** | **0.06607** | **0.03611** | 0.20920 | 14 / 20 | 0.61 ms |

### Scientific Interpretation
1. **Safety & Zero Aborts:** All four methods achieved a 0.0% empirical abort rate during physical transmission by falling back to the null policy when no active candidate met the feasibility gate.
2. **Feasibility Yield:** The Analytical Heuristic and Grid Search achieved 100% feasibility. The cGAN achieved 95% feasibility, significantly outperforming Uniform Random Search (85%).
3. **Mutual Information Trade-off:** The classical cGAN achieved lower mean mutual information ($0.0451$) than the analytical heuristic ($0.0861$). Because the cGAN was trained under conservative penalization to avoid aborts, its generated policies cluster around low-angle probe configurations. While safe, it underutilizes the allowable error budget in relaxed noise environments.

---

## 8. Repository Structure

```
bb84-qgan/
├── README.md                                  # Publication-grade project documentation
├── .gitignore                                 # Git ignore configuration
│
├── src/                                       # Core scientific package
│   ├── bb84/                                  # BB84 protocol implementation
│   │   ├── alice.py                           # State preparation (Z and X bases)
│   │   ├── bob.py                             # Measurement and basis choice
│   │   ├── attacks.py                         # PIR and Fuchs-Peres-Brandt probe attacks
│   │   ├── channel.py                         # Quantum channel transmission
│   │   └── bb84_engine.py                     # Sifting and QBER estimation engine
│   ├── channel/                               # Parametric NISQ noise models
│   │   ├── noise_models.py                    # Kraus operators (depol, damp, dephase, thermal)
│   │   └── quantum_channel.py                 # Composite noisy quantum channel
│   ├── dataset/                               # Dataset generation & schema
│   │   ├── schema.py                          # DatasetRecord dataclass
│   │   ├── generator.py                       # Offline simulation generator
│   │   ├── policy_generator.py                # Policy lattice constructor
│   │   └── statistics.py                      # Wilson score confidence intervals
│   └── cgan/                                  # Classical conditional GAN
│       ├── models.py                          # Generator and Discriminator PyTorch modules
│       ├── policy_engine.py                   # Inference engine & analytical heuristic
│       ├── dataset.py                         # PyTorch Dataset & DataLoader
│       └── trainer.py                         # Baseline cGAN trainer
│
├── experiments/                               # Experiment runners
│   ├── run_bb84_simulation.py                 # BB84 protocol verification sweeps
│   ├── run_noise_experiments.py               # NISQ noise channel validation sweeps
│   ├── generate_full_dataset.py               # 24,000-sample dataset generator
│   ├── train_classical_cgan.py                # Authoritative 200-epoch cGAN training runner
│   └── benchmark_baselines.py                 # 4-method comparative benchmark harness
│
├── tests/                                     # Automated unit test suite (160 tests)
│   ├── test_alice.py                          # Alice state preparation tests
│   ├── test_bob.py                            # Bob measurement tests
│   ├── test_attacks.py                        # Attack disturbance & information tests
│   ├── test_bb84_engine.py                    # Complete BB84 protocol integration tests
│   ├── test_noise_models.py                   # Quantum noise Kraus operator tests
│   ├── test_noisy_channel.py                  # Noisy channel transmission tests
│   ├── test_dataset_pipeline.py               # Dataset generation and schema tests
│   ├── test_cgan_pipeline.py                  # cGAN architecture & selection tests
│   └── test_benchmark_baselines.py            # 4-method benchmark engine tests
│
├── checkpoints/                               # Serialized neural model weights
│   ├── cgan_pilot_best.pt                     # 20-epoch pilot validation model
│   └── classical_cgan/                        # 200-epoch training checkpoints
│       ├── cgan_m4a_lexicographic_best.pt     # Authoritative operational checkpoint (Epoch 150)
│       ├── cgan_m4a_max_feasibility.pt        # Pareto anchor: maximum feasibility
│       ├── cgan_m4a_max_information.pt        # Pareto anchor: maximum mutual information
│       ├── cgan_m4a_balanced_pareto.pt        # Pareto anchor: balanced scalar compromise
│       └── cgan_m4a_latest.pt                 # Final training state at Epoch 200
│
├── results/                                   # Experimental results & datasets
│   ├── full_training_dataset.json             # Authoritative 24,000-sample dataset
│   ├── cgan_benchmark_comparison.json         # Raw 4-method comparative benchmark data
│   ├── bb84_simulation_results.json           # BB84 simulation verification results
│   ├── quantum_noise_results.json             # Quantum noise model validation results
│   └── classical_cgan/                        # Detailed training history & evaluations
│
└── docs/                                      # Research documentation
    ├── research_architecture/                 # System architecture & engineering plans
    ├── bb84_simulation/                       # Simulation design, attack models, validation
    ├── noise_modelling/                       # Quantum noise models & validation
    ├── dataset/                               # Dataset design, generation, provenance
    ├── classical_cgan/                        # Architecture design, training reports, audits
    ├── benchmarking/                          # Benchmark methodology & comparative reports
    ├── quantum_cgan/                          # Variational quantum generator design (PQC)
    ├── final_report.md                        # Comprehensive consolidated research report
    └── repository_cleanup_report.md           # Professional reorganization & audit report
```

---

## 9. Installation Instructions

```powershell
# Clone the repository
git clone https://github.com/tade-jashwitha/bb84-qgan.git
cd bb84-qgan

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt  # Or pip install torch numpy scipy matplotlib
```

---

## 10. Environment Requirements

- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python Version:** Python 3.10 or 3.11
- **PyTorch:** >= 2.0.0 (CUDA support optional; full CPU fallback is verified)
- **Core Libraries:** `numpy >= 1.24`, `scipy >= 1.10`, `matplotlib >= 3.7`

---

## 11. How to Run Experiments

### 1. Verify BB84 Simulation Mechanics
```powershell
python experiments/run_bb84_simulation.py
```

### 2. Verify NISQ Noise Channels
```powershell
python experiments/run_noise_experiments.py
```

### 3. Generate Offline Training Dataset
```powershell
python experiments/generate_full_dataset.py
```

### 4. Train the Classical cGAN
```powershell
python experiments/train_classical_cgan.py --epochs 200 --batch-size 64
```

### 5. Execute 4-Method Baseline Benchmark
```powershell
python experiments/benchmark_baselines.py
```

---

## 12. Testing Instructions

The repository includes a comprehensive 160-test regression suite covering all quantum simulation mechanics, noise models, neural modules, and benchmarking baselines:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Expected output:
```
Ran 160 tests in ~50s
OK
```

---

## 13. Current Research Limitations

1. **Classical MLP Mode Collapse:** The classical generator exhibits conservative mode clustering, prioritizing feasibility over aggressive information extraction.
2. **Symmetric Attack Focus:** Current attack models emphasize individual and collective entangling probes; coherent multi-pulse coherent collective attacks are not currently modeled.
3. **Single-Qubit Channel Modeling:** Simulates single-photon polarization degrees of freedom. Continuous-variable QKD (CV-QKD) and multi-photon decoy-state protocols represent future research directions.

---

## 14. Future Quantum GAN Development (Milestone 4B)

The empirical limitations of the classical cGAN directly motivate Milestone 4B: implementing a Parameterized Quantum Circuit (PQC) generator using PennyLane:
- **Quantum-Native State Space:** Direct mapping between quantum circuit expectation values and non-classical policy correlations.
- **Entanglement-Assisted Exploration:** Investigating whether variational entanglement enables more expressive Pareto frontier sampling under severe query budgets ($K \le 16$).
- See [`docs/quantum_cgan/quantum_generator_design.md`](docs/quantum_cgan/quantum_generator_design.md) for full circuit specifications.

---

## 15. Citation and Reproducibility Information

### Checkpoint and Dataset Hashes (SHA-256)
- **Full Training Dataset:** `07C03E810470B3376EB48C51C13138DCF347D915F78E911F74A81E96F27CFB80`
- **Primary cGAN Checkpoint (Epoch 150):** `F26AEC880D4BAFAF67E22C3691B36450D9273DE2755135236010ECD4BDD2B81E`
- **Benchmark Output JSON:** `6E7DADBE1A6ED81C106B5FF54EBB3B80586CF1B76154329C51927B76C61B57F9`

```bibtex
@misc{tade2026bb84qgan,
  title={Adaptive QGAN-Based Eavesdropping in BB84 QKD: Resource-Constrained Conditional Generative Approach under NISQ Noise},
  author={Tade, Jashwitha},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/tade-jashwitha/bb84-qgan}}
}
```
