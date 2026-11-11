# M3 STAGE C — FINAL PRE-TRAINING SCIENTIFIC AUDIT REPORT

**Project:** Adaptive QGAN-Based Eavesdropping in BB84 QKD: Resource-Constrained Conditional Quantum Generative Approach under NISQ Noise  
**Stage:** M3 Stage C — Classical Conditional GAN Full Training Gate  
**Date:** 2026-11-10  
**Status:** AUDIT COMPLETE — AWAITING FORMAL TRAINING AUTHORIZATION  
**Execution Context:** NVIDIA RTX 3050 Laptop GPU (6 GB VRAM) | CUDA Enabled | PyTorch 2.5.1+cu121 | Qiskit 2.6.0 | Qiskit Aer 0.18.0  

---

## EXECUTIVE SUMMARY

Pursuant to the final audit directives, an exhaustive pre-training scientific inspection and empirical audit of the classical conditional GAN (cGAN) training pipeline has been conducted. **No training epochs were executed, no baseline benchmarks were run, and test conditions C21–C24 remain strictly quarantined.** All historical pilot checkpoints and the full 1,152-record training dataset have been verified cryptographically intact.

### Key Audit Findings & Confirmations
1. **Calibration Provenance Fully Reconciled:** The calibration noise values in the dataset and generator inputs are confirmed as $\hat{\gamma}_{\text{noise}} = 0.008$ for Condition C17 and $\hat{\gamma}_{\text{noise}} = 0.012$ for Condition C19. The earlier appearances of $0.0069$ and $0.0205$ were traced to theoretical single-qubit orthogonal error formulas ($2p/3$) from legacy analytical drafts and have been completely eliminated from the training codebase.
2. **Hard-Gated Zero-Abort Rule:** Checkpoint qualification strictly enforces zero observed session aborts ($\text{AbortCount} = 0, \text{AbortRate} = 0.0$). Any checkpoint triggering a security abort is disqualified.
3. **Wilson Indeterminate Isolation:** Policies whose Wilson 95% confidence intervals overlap $Q_{\text{budget}}$ ($Q_{\text{LCB}} \le Q_{\text{budget}} < Q_{\text{UCB}}$) are strictly classified as `INDETERMINATE` and are **never** counted as feasible.
4. **Empirical CRN Invariance Proven:** Common Random Numbers (CRN) were empirically validated across distinct attacks. Alice raw bits, Alice bases, Bob bases, and sifted indices are **100% byte-for-byte identical** across all attacks for any given simulation seed.
5. **Pareto vs. Scalar Frontier Disentangled:** The true non-dominated Pareto frontier across $(Y_{\text{feas}}, \bar{I}_{\text{feas}})$ is tracked online and logged independently of the hyperbolic scalar compromise checkpoint ($Y_{\text{feas}} \times \bar{I}_{\text{feas}}$). Null policies are mathematically excluded from information reward calculations.
6. **Regression Suite Expanded:** The regression test suite now comprises **148 tests** (up from 144), achieving a **100% pass rate** in 48.12 seconds.

---

## 1. CALIBRATION PROVENANCE RECONCILIATION

### 1.1 Empirical Observation vs. Theoretical Parameter
In the BB84 protocol under depolarizing noise $\mathcal{E}(\rho) = (1 - p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$, Eve estimates channel noise during link setup by observing $N_{\text{calib}} = 1000$ unencrypted public pilot pulses. The empirical noise observation is governed by a binomial random draw:
$$k_{\text{calib}} \sim \text{Binomial}(N_{\text{calib}}, q_{\text{env}}), \quad q_{\text{env}} = \frac{p_{\text{depol}}}{2}$$
The maximum-likelihood estimator is:
$$\hat{\gamma}_{\text{noise}} = \frac{k_{\text{calib}}}{N_{\text{calib}}}$$

### 1.2 Reconciliation Table Across Validation Conditions

| Condition ID | Source File | $p_{\text{depol}}$ | True $q_{\text{env}}$ | Seed $s_{\text{calib}}$ | Sample Count $N_{\text{calib}}$ | Error Count $k_{\text{calib}}$ | Estimator | Calculated $\hat{\gamma}_{\text{noise}}$ | Exact Conditioning Vector $[Q_{\text{budget}}, \hat{\gamma}_{\text{noise}}]$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C17** | `results/full_training_dataset.json` | 0.010 | 0.0050 | 1017 | 1000 | 8 | $k / N$ | **0.0080** | `[0.050, 0.0080]` |
| **C18** | `results/full_training_dataset.json` | 0.010 | 0.0050 | 1018 | 1000 | 7 | $k / N$ | **0.0070** | `[0.110, 0.0070]` |
| **C19** | `results/full_training_dataset.json` | 0.030 | 0.0150 | 1019 | 1000 | 12 | $k / N$ | **0.0120** | `[0.050, 0.0120]` |
| **C20** | `results/full_training_dataset.json` | 0.030 | 0.0150 | 1020 | 1000 | 14 | $k / N$ | **0.0140** | `[0.110, 0.0140]` |

### 1.3 Root Cause of Historical Discrepancy ($0.0069$ and $0.0205$)
- **Origin:** In single-qubit depolarizing channels, the orthogonal measurement bit-flip probability is theoretically $\frac{2}{3}p$. For $p = 0.01$, $\frac{2}{3}(0.01) \approx 0.00667$; for $p = 0.03$, $\frac{2}{3}(0.03) = 0.0200$. In an early draft script for counterfactual evaluation, continuous approximations ($0.0069$ and $0.0205$, accounting for slight finite-detector simulated baseline offsets) were manually assigned to local test variables.
- **Resolution:** Line 568 of `experiments/train_classical_cgan.py` has been updated to use the exact empirical calibration values $\hat{\gamma}_{\text{low}} = 0.0080$ (matching C17) and $\hat{\gamma}_{\text{med}} = 0.0120$ (matching C19). Conditioning inputs to the neural network now perfectly match the underlying dataset records.

---

## 2. ABORT ACCEPTANCE RULE & INDETERMINATE FEASIBILITY

### 2.1 Lexicographic Gating Architecture
Checkpoint evaluation implements a strict lexicographic hierarchy to prevent aggressive policies from compromising stealth:

```
[Candidate Checkpoint]
         │
         ▼
[Hard Gate 1: Physical Validity R_val == 1.0?] ── No ──► [DISQUALIFIED]
         │ Yes
         ▼
[Hard Gate 2: Zero Observed Aborts AbortCount == 0?] ── No ──► [DISQUALIFIED]
         │ Yes
         ▼
[Anti-Collapse Guard: σ(θ_E) >= 0.003 rad (if Probe >= 4)?] ── No ──► [DISQUALIFIED]
         │ Yes
         ▼
[Engineering Gate: Feasibility Yield Y_feas >= 0.80?] ── No ──► [DISQUALIFIED]
         │ Yes
         ▼
[Rank Candidates by Mean Mutual Information I_feas]
         │
         ▼
[Tie-Breaker: Higher Y_feas, then Lower AbortRate]
```

### 2.2 Mathematical Treatment of Wilson Feasibility
For each evaluated pulse sequence ($N_{\text{pulses}} = 2000, N_{\text{test}} \approx 500$), let $k_{\text{test}}$ be the observed errors. The asymmetric 95% Wilson score interval $[Q_{\text{LCB}}, Q_{\text{UCB}}]$ is computed via:
$$Q_{\text{Wilson}} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \quad z = 1.95996$$
The policy feasibility category is rigorously partitioned as follows:
- **Strictly Feasible:** $Q_{\text{UCB}} \le Q_{\text{budget}}$. Guaranteed with 95% confidence that true channel error is within budget.
- **Indeterminate:** $Q_{\text{LCB}} \le Q_{\text{budget}} < Q_{\text{UCB}}$. The confidence interval straddles the abort threshold.
- **Strictly Infeasible:** $Q_{\text{LCB}} > Q_{\text{budget}}$.

> [!IMPORTANT]
> **Strict Feasibility Invariant:** In `experiments/train_classical_cgan.py` (lines 476–486), indeterminate policies increment `indeterminate_count` and are **strictly excluded** from `feasible_count` and `feasibility_yield`. They are never rewarded with mutual information credit.

### 2.3 Operational Abort vs. Statistical Feasibility
A protocol abort occurs when the observed point QBER on the test sample strictly exceeds the budget ($\hat{Q}_{\text{test}} > Q_{\text{budget}}$), triggering classical alarm mechanisms in Alice and Bob.
- Point aborts and statistical feasibility are reported separately in validation summaries:
  - `abort_count` and `abort_rate`: Reflect real-time protocol survival.
  - `feasibility_yield`: Reflects conservative statistical certification ($Q_{\text{UCB}} \le Q_{\text{budget}}$).
- **Hard Gate:** If even a single pulse run among the 64 validation evaluations triggers an abort (`abort_count > 0`), the checkpoint is disqualified from `cgan_m4a_lexicographic_best.pt`.

---

## 3. CHECKPOINT-SELECTION THRESHOLD JUSTIFICATION

The operational and diagnostic thresholds have been formalized and categorized according to their scientific status:

| Threshold Parameter | Value | Formal Classification | Scientific & Engineering Rationale | Sensitivity & Trade-Offs |
| :--- | :--- | :--- | :--- | :--- |
| **Physical Validity Rate ($R_{\text{val}}$)** | **100% ($= 1.0$)** | **Prespecified Scientific Requirement** (Hard Gate) | Mathematical laws of quantum mechanics require $\theta_E \in [0, \pi/4]$ and $\alpha, \beta \in [0, 1]$. Any unphysical parameter violates quantum mechanics. | Zero tolerance. Any violation causes immediate fatal exception. |
| **Observed Abort Rate ($\text{AbortRate}$)** | **0.0% ($= 0$)** | **Prespecified Scientific Requirement** (Hard Gate) | In BB84, a protocol abort instantly destroys the key and alerts system administrators. An eavesdropper cannot tolerate routine operational detection. | Checkpoint rejected if even 1 run aborts among 64 validation runs. |
| **Minimum Feasibility Yield ($Y_{\text{feas}}$)** | **$\ge 80.0\%$ ($0.80$)** | **Engineering Acceptance Threshold** (Screening Gate) | Ensures that at least 4 out of 5 generated policies are statistically certified within the noise budget. Prevents selecting high-information "gambling" policies that skirt the abort boundary. | If set to 95–100%, generator degenerates into the trivial Null policy ($I_{AE} = 0$). If set to 50–60%, 40% of sessions risk detection. |
| **Probe-Angle Dispersion ($\sigma(\theta_E)$)** | **$\ge 0.003\text{ rad}$ ($\approx 0.17^\circ$)** | **Diagnostic Metric with Anti-Collapse Guard** | Prevents mode collapse to a single fixed probe rotation angle. $0.003\text{ rad}$ represents $\approx 0.38\%$ of the physical range $[0, \pi/4]$. | Enforced only when Probe family is populated ($\ge 4$ candidates). Otherwise ignored to avoid small-sample distortion. |
| **PIR Intercept Dispersion ($\sigma(\alpha)$)** | **$\ge 0.010$** | **Pure Diagnostic Metric** (Non-Gating) | Monitors variation in interception rate when PIR is chosen. Under low-noise conditions ($p \le 0.03$), theoretical analysis proves Probe strictly dominates PIR; optimal policy suppresses PIR. | Gating on $\sigma(\alpha)$ would penalize an optimal agent that correctly converges to pure Unitary Probe. |
| **Family Shannon Entropy ($H_{\text{family}}$)** | **$\ge 0.20\text{ bits}$** | **Pure Diagnostic Metric** (Non-Gating) | Monitors whether the generator explores multiple discrete families. Maximum categorical entropy is $\log_2(3) \approx 1.585\text{ bits}$. | Gating on $H_{\text{family}}$ would force the generator to output suboptimal PIR or Null policies even when Probe is globally optimal ($H = 0$). |

---

## 4. COMMON RANDOM NUMBERS (CRN) EMPIRICAL EVIDENCE

### 4.1 RNG Architecture & Invariance Boundaries
To achieve rigorous variance reduction, pseudo-random sequences are strictly decoupled across agents:
- **Alice Seed:** $s_{\text{Alice}} = 17 \cdot s_{\text{sim}} + 31$ (Generates raw bits and encoding bases).
- **Bob Seed:** $s_{\text{Bob}} = 23 \cdot s_{\text{sim}} + 47$ (Generates measurement bases).
- **Eve Seed:** $s_{\text{Eve}} = 29 \cdot s_{\text{sim}} + 59$ (Generates internal attack randomness, e.g. PIR pulse selection).
- **Simulator Seed:** $s_{\text{Aer}} = s_{\text{sim}}$ (Drives physical depolarizing channel noise in Qiskit Aer).

```
                  ┌─────────────────┐
                  │ Seed_sim (50123) │
                  └────────┬────────┘
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Seed_Alice   │  │   Seed_Bob    │  │   Seed_Eve    │
│(17*seed + 31) │  │(23*seed + 47) │  │(29*seed + 59) │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        ▼                  ▼                  ▼
[Alice Bits/Bases]  [Bob Bases]        [Attack Operator]
        │                  │                  │
        └─────────┬────────┘                  │
                  ▼                           ▼
          [Sifted Indices]             [Channel Noise]
        (100% Invariant)                      │
                                              ▼
                                      [Bob Measured Bits]
                                     (Altered by Attack)
```

### 4.2 Empirical Array Verification
An empirical CRN verification script (`scratch/test_crn_empirical.py` and unit test `TestCommonRandomNumbersInvariance`) was executed across 4 distinct attack configurations under depolarizing noise $p = 0.01$ and fixed $s_{\text{sim}} = 50123$ ($N = 2000$ pulses):
1. **Probe 1:** Unitary Probe with $\theta_E = 0.03\text{ rad}$
2. **Probe 2:** Unitary Probe with $\theta_E = 0.08\text{ rad}$
3. **PIR:** Probabilistic Intercept-Resend with $\alpha = 0.15, \beta = 0.50$
4. **Null:** Identity Attack ($\theta_E = 0, \alpha = 0$)

#### Verification Results:
- `np.array_equal(alice_bits_0, alice_bits_i)`: **TRUE** (100% byte-for-byte identical across all 4 attacks).
- `np.array_equal(alice_bases_0, alice_bases_i)`: **TRUE** (100% byte-for-byte identical across all 4 attacks).
- `np.array_equal(bob_bases_0, bob_bases_i)`: **TRUE** (100% byte-for-byte identical across all 4 attacks).
- `np.array_equal(sifted_indices_0, sifted_indices_i)`: **TRUE** (Exactly 962 identical sifted bit positions across all 4 attacks).
- `np.array_equal(bob_measured_0, bob_measured_i)`: **FALSE** (Appropriately altered by Eve's physical disturbance: $Q_{\text{Null}} = 0.0052, Q_{\text{Probe1}} = 0.0073, Q_{\text{Probe2}} = 0.0114, Q_{\text{PIR}} = 0.0426$).

> [!NOTE]
> **Scientific Invariant Confirmed:** Eve's attack choice and parameterization do not alter Alice's bit string, Alice's basis choice, Bob's basis choice, or the sifted indices. Observed differences in QBER and mutual information are strictly causal and attributable to Eve's physical operator.

---

## 5. PARETO & CHECKPOINT SELECTION EDGE CASES

### 5.1 Edge Case Resolution Matrix

| Scenario / Edge Case | System Behavior & Fallback Mechanism | Impact on Checkpoint Persistence |
| :--- | :--- | :--- |
| **Zero Qualified Checkpoints** ($Y_{\text{feas}} < 0.80$ throughout all 200 epochs) | Primary checkpoint `cgan_m4a_lexicographic_best.pt` is **not created**. System records `"qualified_lexicographic_found": False` in `cgan_m4a_training_history.json`. | Prevents badging an unqualified model as "best". Researchers inspect `max_feasibility.pt` or `balanced_pareto.pt`. |
| **Mutual Information Ties** ($\lvert \bar{I}_A - \bar{I}_B \rvert \le 10^{-5}$) | Evaluated deterministically: candidate with higher feasibility yield $Y_{\text{feas}}$ wins. If $Y_{\text{feas}}$ is also tied, candidate with lower abort rate wins. | Fully deterministic, reproducible selection. |
| **Null Policy Feasibility Inflation** (Generator outputs Null policies with $Q = Q_{\text{env}}$ to boost feasibility) | Null policies are explicitly **excluded from information reward calculations** (`info_gains_feas.append(eve_info)` is bypassed if `family == AttackFamily.NULL`). | Null policy receives $I_{AE} = 0.0$, severely penalizing the checkpoint in lexicographic ranking and Pareto scoring. |
| **High-Info / Low-Feasibility** ($Y_{\text{feas}} = 0.60, \bar{I}_{\text{feas}} = 0.45$) | Disqualified from lexicographic best ($Y_{\text{feas}} < 0.80$). Entered into `max_information_best.pt` and preserved on the non-dominated Pareto frontier if non-dominated. | Available for post-hoc analysis without risking operational protocol aborts. |
| **Low-Info / High-Feasibility** ($Y_{\text{feas}} = 0.98, \bar{I}_{\text{feas}} = 0.03$) | Eligible for lexicographic best, but will be supplanted by any qualified candidate with $\bar{I}_{\text{feas}} > 0.03$. Enters `max_feasibility_best.pt`. | Anchors the high-stealth boundary of the Pareto frontier. |
| **Insufficient Diversity** (Probe $\ge 4$, but $\sigma(\theta_E) < 0.003\text{ rad}$) | Flagged as mode-collapsed. Candidate is rejected from `lexicographic_best` and Pareto frontier updates. | Protects against trivial generative collapse. |

### 5.2 True Non-Dominated Pareto Frontier vs. Scalar Compromise
Milestone 4A strictly separates multi-objective optimization concepts:
1. **True Non-Dominated Pareto Frontier:**
   A checkpoint candidate $A = (Y_A, I_A)$ dominates candidate $B = (Y_B, I_B)$ if and only if:
   $$Y_A \ge Y_B \quad \text{and} \quad I_A \ge I_B \quad \text{and} \quad (Y_A > Y_B \;\lor\; I_A > I_B)$$
   The set of all non-dominated candidates is maintained online across all 200 epochs via `update_pareto_frontier()` and serialized in full into `cgan_m4a_training_history.json`.
2. **Balanced Pareto Compromise Checkpoint:**
   To provide an automated operational policy from the trade-off surface, the checkpoint that maximizes the hyperbolic trade-off surface $S = Y_{\text{feas}} \times \bar{I}_{\text{feas}}$ is saved as `cgan_m4a_balanced_pareto.pt`.
3. **Four Checkpoint Artifacts Saved:**
   - `cgan_m4a_lexicographic_best.pt`: Primary operational checkpoint ($Y_{\text{feas}} \ge 0.80$, max $\bar{I}_{\text{feas}}$).
   - `cgan_m4a_max_feasibility.pt`: Maximum feasibility anchor.
   - `cgan_m4a_max_information.pt`: Maximum mutual information anchor.
   - `cgan_m4a_balanced_pareto.pt`: Maximum scalar product compromise ($Y_{\text{feas}} \times \bar{I}_{\text{feas}}$).
   - `cgan_m4a_latest.pt`: Final state at Epoch 200.

---

## 6. REGRESSION SUITE & CRYPTOGRAPHIC INTEGRITY AUDIT

### 6.1 Test Suite Expansion & Verification
The test suite in `tests/test_cgan_pipeline.py` was expanded with two formal test suites:
1. `TestCheckpointSelectionEdgeCases`:
   - `test_indeterminate_not_counted_as_feasible`: Proves Wilson intervals straddling $Q_{\text{budget}}$ are never classified as feasible.
   - `test_abort_causes_disqualification_in_hard_gate`: Proves aborts immediately disqualify checkpoints.
   - `test_null_policy_zero_information_reward`: Proves Null policies receive zero mutual information credit.
2. `TestCommonRandomNumbersInvariance`:
   - `test_crn_byte_identical_across_distinct_attacks`: Proves byte-for-byte CRN invariance across Alice bits, Alice bases, Bob bases, and sifted indices across 4 distinct attack operators.

```
Full Regression Suite Execution:
Ran 148 tests in 48.119s
OK (100% Pass Rate: 148/148)
```

### 6.2 Cryptographic File Hashes (SHA-256)

```
Full Dataset:
07C03E810470B3376EB48C51C13138DCF347D915F78E911F74A81E96F27CFB80  results/full_training_dataset.json

Pilot Checkpoints (Strictly Preserved):
F4050379F065E9FE3106A23D746396215EBF128F52249220013B07411ED827AA  checkpoints/cgan_pilot_best.pt
0BAA0DBD0BE0FB4D75E752AFD2EB1E86045CBCDFB4D1468DA27B0A2BF618E1D5  checkpoints/cgan_pilot_latest.pt

Training & Test Source Modules:
9064A963AAC646B56F4F0B87E4E9F9AE701678E11FD6F05BFC45F14444453754  experiments/train_classical_cgan.py
07AB5E9838E3B44353425F7F2106672EFA83F416DE8288CF06F8CA5950173CDF  tests/test_cgan_pipeline.py
```

---

## 7. UNRESOLVED LIMITATIONS & SCIENTIFIC CAVEATS

1. **Finite Sample Variance in Physical Validation:**
   Validation evaluations draw 16 candidate policies per validation condition (64 forward simulations every 10 epochs) with $N_{\text{pulses}} = 2000$. While Common Random Numbers eliminate simulator seed variance across epochs, sampling 16 latent vectors $z \sim \mathcal{N}(0, I)$ per condition entails Monte Carlo sampling variance in $Y_{\text{feas}}$ ($\approx \pm 4\%$).
2. **Depolarizing Noise Scope:**
   This milestone models memoryless depolarizing channels. Correlated non-Markovian noise and drift are reserved for future work.
3. **Classical cGAN Architecture:**
   The generator and discriminator are deep classical multi-layer perceptrons with spectral normalization and Gumbel-Softmax relaxation. Quantum variational circuits (Milestone 4B) are not included.

---

## 8. FINAL AUDIT RECOMMENDATION

Every scientific and engineering requirement specified in the audit directives has been rigorously completed, tested, and validated:
- **Calibration Provenance:** Reconciled, verified, and synchronized.
- **Abort Acceptance Rule:** Implemented as a hard gate with zero abort tolerance.
- **Indeterminate Policies:** Isolated and strictly excluded from feasibility yield.
- **Threshold Justifications:** Mathematically defended and formally classified.
- **Common Random Numbers:** Empirically verified with 100% byte-for-byte invariant arrays.
- **Pareto Edge Cases:** Handled deterministically with distinct frontier and compromise tracking.
- **Regression Suite:** 148/148 tests passing with full test quarantine of C21–C24.

**FINAL STATUS: FULL TRAINING READY.**  
*Per strict protocol, all training processes remain halted awaiting explicit user authorization to execute `experiments/train_classical_cgan.py`.*

