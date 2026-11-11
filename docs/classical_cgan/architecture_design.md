# MILESTONE 3: STAGE A — FINAL SCIENTIFIC CONSISTENCY AUDIT & DESIGN REPORT
## Classical Conditional GAN (cGAN) Baseline for Adaptive BB84 Eavesdropping under NISQ Channel Noise

**Project:** Adaptive QGAN-Based Eavesdropping in BB84 QKD: Resource-Constrained Conditional Quantum Generative Approach under NISQ Noise  
**Author:** Quantum Computing & Machine Learning Research Team  
**Date:** November 10, 2026  
**Document Status:** Stage A Final Scientific Design (Audited Pre-Implementation Specification)  
**Target System Hardware & Software Stack:**
- **OS:** Windows 11
- **Python:** 3.11.9
- **Qiskit Core:** 2.6.0
- **Qiskit Aer:** 0.18.0 (CPU Multi-threaded OpenMP backend verified; available devices: `('CPU', )`)
- **PyTorch:** 2.5.1+cu121 (CUDA accelerated; GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU)
- **Scientific Stack:** NumPy 2.4.6, SciPy 1.17.1

---

## Executive Summary of Audit Resolutions

This report provides the final, mathematically audited scientific specification for **Milestone 3: Stage A**, resolving all consistency issues identified in the scientific audit. 

### Key Scientific Revisions Implemented
1. **Exact Finite-Sample QBER Feasibility & Outcome Separation:** Replaced the approximate Wald interval with the exact **Wilson Score Interval** at explicit confidence level $1 - \alpha_{\text{stat}} = 0.95$. Transmitted optical pulses ($N_{\text{pulse}}$) and sifted test bits ($N_{\text{test}}$) are strictly distinguished. A rigorous three-state feasibility partition is established (*Feasible*, *Infeasible*, *Indeterminate*). We explicitly remove and reject any assertion that Wilson feasibility alone mathematically guarantees a $95\%$ empirical protocol pass probability; empirical pass/abort frequencies are determined by repeated simulation trials under Alice and Bob's deterministic sample-QBER rule.
2. **Clarification of Conditioning Variables & Explicit Limitations:** Retained the 2D condition vector $\mathbf{c} = [Q_{\text{budget}}, \hat{\gamma}_{\text{noise}}]^T$. Defined $\hat{\gamma}_{\text{noise}}$ as Eve's empirical estimate of the **ambient channel QBER** ($Q_{\text{env}} = p_{\text{depol}}/2$) measured via unencrypted calibration pilot pulses, explicitly distinguishing it from the microscopic depolarizing parameter $p_{\text{depol}}$. Formally documented the inability of a scalar estimate to distinguish anisotropic noise families (Phase Damping vs. Amplitude Damping vs. Depolarizing) as an explicit scientific limitation of the classical baseline.
3. **Exact Deterministic Dataset Sampling Grid:** Specified an exact 16-point attack table (Null attack, 7 Unitary Probe angles in radians, 8 PIR joint $(\alpha, \beta)$ configurations) evaluated across 24 environmental conditions (8 depolarizing noise rates $\times$ 3 error budgets) with $R = 3$ independent simulation seeds per configuration (1,152 total simulation executions). Defined a strict condition-level partitioning rule that prevents seed leakage across train, validation, and test splits.
4. **Formal Definition of Learned Conditional Distribution & Tri-Case Fallback:** Delineated the 6-stage lifecycle (Dataset Construction $\to$ Target Distribution $\to$ Candidate Generation $\to$ Simulation Evaluation $\to$ Statistical Filtering $\to$ Final Ranking). Clarified that GAN minimax training minimizes adversarial divergence to feasible exemplars and does **not** directly maximize mutual information via objective gradients. Formulated a rigorous three-case operational fallback (Case A: Active Feasible; Case B: Null Feasible; Case C: Null-attack reference condition infeasible under configured budget) and acknowledged that the primary sampling grid ($Q_{\text{env}} \le 0.035 < Q_{\text{budget}}$) does not exercise Case C.
5. **Fair Baseline Benchmarking under Equal Simulation Query Budget:** Enforced an identical budget of $K = 16$ simulator queries across cGAN, Uniform Random Search, Grid Search, and the **Analytical Probe Heuristic** (formerly "Optimal Analytical Policy"), documenting that the heuristic is an approximate initialization covering only the Unitary Probe family.
6. **Empirically Measured Computational Pilot & Weighted Runtime:** Executed an empirical pilot on the local machine verifying that Qiskit Aer operates on CPU OpenMP, measuring exact per-attack runtimes at 3,000 pulses (Null: 1.1838s, PIR: 1.6289s, Unitary Probe: 1.8076s). Calculated the true weighted average runtime ($\bar{t}_{\text{weighted, 3000}} \approx 1.6793\,\text{s}$) across the 16 dataset attack configurations, yielding an expected 3,000-pulse runtime of **32.2 minutes** (or **35.5 minutes** with 10% overhead). Linearly extrapolated 2,000-pulse runtime to **21.5 minutes** (or **23.6 minutes** with overhead).

---

## 1. Correct Finite-Sample QBER Feasibility

### 1.1 Distinction: Transmitted Pulses vs. Sifted Test Bits

In the BB84 protocol without decoy states, error estimation is conducted on a dedicated subset of sifted key bits, not on the raw transmitted optical pulse stream:

```
  +---------------------------------------------------------------------------------+
  | Transmitted Pulses (N_pulse = 2,000 or 3,000)                                   |
  +---------------------------------------+-----------------------------------------+
                                          |
                        [50% Conjugate Basis Sifting]
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | Sifted Bits (N_sift ≈ 1,000 or 1,500)                                           |
  +---------------------------------------+-----------------------------------------+
                     /                                                 \
  [Test Sift Partition: ρ_test = 0.50]                 [Key Sift Partition: 1 - ρ_test]
                   /                                                     \
                  v                                                       v
  +---------------------------------------+   +-------------------------------------+
  | Error Estimation Bits (N_test)        |   | Raw Key Generation Bits (N_key)     |
  | (N_test = 500 or 750 bits)            |   | (N_key = 500 or 750 bits)           |
  | Publicly disclosed by Alice & Bob     |   | Kept secret for Privacy Amplification|
  +---------------------------------------+   +-------------------------------------+
```

1. **Transmitted Pulses ($N_{\text{pulse}}$):** Total single-photon optical carrier pulses prepared by Alice and transmitted over the quantum link ($N_{\text{pulse}} = 2,000$ in standard runtime; $3,000$ in high-precision mode).
2. **Sifted Key Bits ($N_{\text{sift}}$):** Pulses where Alice's preparation basis matches Bob's measurement basis ($x_A = x_B$):
   $$\mathbb{E}[N_{\text{sift}}] = \frac{1}{2} N_{\text{pulse}} \approx 1,000 \text{ bits (for } N_{\text{pulse}}=2,000\text{)}$$
3. **Sifted Test Bits ($N_{\text{test}}$):** A fraction $\rho_{\text{test}} = 0.50$ of sifted bits randomly sampled and publicly compared over the authenticated classical channel:
   $$N_{\text{test}} = \lfloor \rho_{\text{test}} \cdot N_{\text{sift}} \rfloor \approx 500 \text{ bits (or } 750 \text{ bits for } N_{\text{pulse}}=3,000\text{)}$$
4. **Secret Key Bits ($N_{\text{key}}$):** The remaining unrevealed sifted bits ($N_{\text{key}} \approx 500$ or $750$ bits) subjected to error correction and privacy amplification if the session is not aborted.

### 1.2 Mathematical Formulation of the Wilson Score Interval

The standard Wald interval $\hat{p} \pm z \sqrt{\hat{p}(1-\hat{p})/n}$ breaks down severely when error rates are small ($Q < 0.05$) and sample sizes are finite ($N_{\text{test}} \approx 500$), frequently generating unphysical negative lower bounds and actual coverage probabilities far below nominal confidence.

We implement the **Wilson Score Interval** at explicit confidence level $1 - \alpha_{\text{stat}} = 0.95$ (two-sided standard normal quantile $z = 1.95996 \approx 1.960$).

Let $k$ be the observed number of bit errors among the $N_{\text{test}}$ disclosed test bits ($k \sim \text{Binomial}(N_{\text{test}}, Q_{\text{true}})$). The sample QBER is:
$$\hat{Q} = \frac{k}{N_{\text{test}}}$$

The Wilson center point $Q_{\text{center}}$ and margin $\Delta Q_{\text{Wilson}}$ are defined as:
$$Q_{\text{center}} = \frac{\hat{Q} + \frac{z^2}{2 N_{\text{test}}}}{1 + \frac{z^2}{N_{\text{test}}}}$$
$$\Delta Q_{\text{Wilson}} = \frac{z}{1 + \frac{z^2}{N_{\text{test}}}} \sqrt{\frac{\hat{Q}(1 - \hat{Q})}{N_{\text{test}}} + \frac{z^2}{4 N_{\text{test}}^2}}$$

The Lower Confidence Bound ($Q_{\text{LCB}}$) and Upper Confidence Bound ($Q_{\text{UCB}}$) are strictly bounded in $[0, 1]$:
$$Q_{\text{LCB}} = \max\left(0.0, \; Q_{\text{center}} - \Delta Q_{\text{Wilson}}\right)$$
$$Q_{\text{UCB}} = \min\left(1.0, \; Q_{\text{center}} + \Delta Q_{\text{Wilson}}\right)$$

### 1.3 Separation of Statistical Feasibility from Empirical Protocol Outcomes

We strictly separate the concept of **statistical feasibility** evaluated on candidate policies from the **empirical protocol execution outcomes** experienced by Alice and Bob:

#### A. Statistical Feasibility Classification (Policy Evaluation)
Using the two-sided $95\%$ Wilson score interval $[Q_{\text{LCB}}, Q_{\text{UCB}}]$ evaluated over $N_{\text{test}}$ test bits, every candidate attack policy $a$ under condition $c = (Q_{\text{budget}}, \hat{\gamma}_{\text{noise}})$ is classified into one of three mutually exclusive categories:

$$\text{Category}(a \mid c) = \begin{cases} 
\mathbf{Statistically\;Feasible} & \text{if } Q_{\text{UCB}}(a; c) \le Q_{\text{budget}} \\
\mathbf{Statistically\;Infeasible} & \text{if } Q_{\text{LCB}}(a; c) > Q_{\text{budget}} \\
\mathbf{Statistically\;Indeterminate} & \text{if } Q_{\text{LCB}}(a; c) \le Q_{\text{budget}} < Q_{\text{UCB}}(a; c)
\end{cases}$$

- **Statistically Feasible (Category A):** $Q_{\text{UCB}} \le Q_{\text{budget}}$. The entire upper confidence bound lies at or below the protocol threshold under the specified $95\%$ Wilson criterion.
- **Statistically Infeasible (Category B):** $Q_{\text{LCB}} > Q_{\text{budget}}$. The entire lower confidence bound lies strictly above the protocol threshold.
- **Statistically Indeterminate (Category C):** $Q_{\text{LCB}} \le Q_{\text{budget}} < Q_{\text{UCB}}$. The confidence interval straddles the threshold; sample size is insufficient to decisively accept or reject feasibility at the specified confidence level.

```
   0.0%                    Q_LCB           Q_budget           Q_UCB              1.0
----+------------------------+----------------+-----------------+-----------------> QBER
    |                        |                |                 |
    |  STATISTICALLY         |                |                 |  STATISTICALLY
    |  FEASIBLE              |  INDETERMINATE |  INDETERMINATE  |  INFEASIBLE
    |  Q_UCB <= Q_budget     |                |                 |  Q_LCB > Q_budget
    |  Full interval below   | Interval spans | Interval spans  |  Full interval above
    |                        | threshold      | threshold       |
```

> **Critical Statistical Clarification:** We explicitly remove and reject any assertion that Wilson feasibility ($Q_{\text{UCB}} \le Q_{\text{budget}}$) alone mathematically guarantees an empirical protocol pass probability of $\ge 95\%$. The Wilson score interval is a parameter confidence interval for the underlying true error rate $Q_{\text{true}}$, not a tolerance prediction interval for individual future sample test statistics $\hat{Q}_{\text{test}}$.

#### B. Empirical Protocol Execution & Outcome Measurement
In operational execution, Alice and Bob apply their deterministic sample-QBER rule on disclosed test bits:
$$\text{Protocol Action} = \begin{cases} \mathbf{ABORT} & \text{if } \hat{Q}_{\text{test}} > Q_{\text{budget}} \\ \mathbf{PASS} & \text{if } \hat{Q}_{\text{test}} \le Q_{\text{budget}} \end{cases}$$

To determine actual protocol pass and abort frequencies:
- Alice and Bob's abort rule is evaluated across $R$ repeated independent simulation seeds.
- We report the exact number of simulation trials $R$.
- The empirical pass frequency $\hat{P}_{\text{pass}}$ and empirical abort frequency $\hat{P}_{\text{abort}}$ are calculated as:
  $$\hat{P}_{\text{pass}} = \frac{1}{R} \sum_{r=1}^R \mathbb{I}\left( \hat{Q}_{\text{test}}^{(r)} \le Q_{\text{budget}} \right), \quad \hat{P}_{\text{abort}} = \frac{1}{R} \sum_{r=1}^R \mathbb{I}\left( \hat{Q}_{\text{test}}^{(r)} > Q_{\text{budget}} \right)$$
- When presenting these empirical frequencies, we report an appropriate binomial confidence interval (such as the Wilson score interval on $\hat{P}_{\text{pass}}$ over the $R$ trials).

---

## 2. Clarification of Conditioning Variables & Physical Noise Estimator

### 2.1 The Two-Dimensional Condition Vector

The primary classical baseline retains the compact two-dimensional condition vector:
$$\mathbf{c} = \begin{bmatrix} Q_{\text{budget}} \\ \hat{\gamma}_{\text{noise}} \end{bmatrix} \in \mathbb{R}^2$$

1. **Protocol Abort Budget ($Q_{\text{budget}}$):**
   - Public standard specification parameter known prior to key exchange ($Q_{\text{budget}} \in [0.04, 0.11]$).
   - Zero measurement error: $Q_{\text{budget}}^{\text{Eve}} = Q_{\text{budget}}$.
2. **Ambient Noise Estimator ($\hat{\gamma}_{\text{noise}}$):**
   - Defined explicitly as Eve's empirical estimate of the **ambient channel QBER** ($Q_{\text{env}}$) experienced by carrier pulses traversing the quantum link.

### 2.2 Physical Disambiguation: Microscopic Noise vs. Channel QBER

We strictly distinguish the four interrelated physical quantities:

```
+--------------------------------------------------------------------------------------------------+
|                              PHYSICAL NOISE QUANTITY DISAMBIGUATION                              |
+--------------------------+-----------------------+-----------------------------------------------+
| Quantity                 | Mathematical Symbol   | Physical Definition & Operational Context     |
+--------------------------+-----------------------+-----------------------------------------------+
| Microscopic Depolarizing | p_depol in [0, 1]     | Kraus operator probability parameter in Aer:   |
| Parameter                |                       | E(rho) = (1 - p)*rho + p*(I/2)                |
+--------------------------+-----------------------+-----------------------------------------------+
| True Ambient Channel     | Q_env = p_depol / 2   | Physical bit-flip probability experienced by   |
| QBER                     |                       | carrier qubits in matching bases without Eve  |
+--------------------------+-----------------------+-----------------------------------------------+
| Empirical Calibration    | Q_hat_calib =         | Observed bit error rate measured across       |
| Error Rate               | k_calib / N_calib     | N_calib = 1,000 public pilot pulses           |
+--------------------------+-----------------------+-----------------------------------------------+
| Estimated Channel Noise  | gamma_hat_noise =     | The conditioning variable supplied to Eve's   |
| Parameter (Condition)    | Q_hat_calib           | cGAN (in sifted-error units: gamma_hat ~ Q_env|
+--------------------------+-----------------------+-----------------------------------------------+
```

**Operational Estimation Protocol:**
- During optical link alignment prior to key transmission, Alice and Bob exchange $N_{\text{calib}} = 1,000$ unencrypted public pilot pulses across matching conjugate bases.
- Eve passively monitors this calibration stream, counting the number of erroneous bits $k_{\text{calib}} \sim \text{Binomial}(N_{\text{calib}}, Q_{\text{env}})$.
- Eve's estimator is:
  $$\hat{\gamma}_{\text{noise}} \equiv \hat{Q}_{\text{calib}} = \frac{k_{\text{calib}}}{N_{\text{calib}}}$$
  $$\mathbb{E}[\hat{\gamma}_{\text{noise}}] = Q_{\text{env}} = \frac{p_{\text{depol}}}{2}$$
- If Eve requires an estimate of the microscopic depolarizing parameter, she computes $\hat{p}_{\text{depol}} = 2\hat{\gamma}_{\text{noise}}$.
- **Measurement Uncertainty:**
  $$\sigma_{\text{calib}}^2 = \frac{Q_{\text{env}}(1 - Q_{\text{env}})}{N_{\text{calib}}}$$
  For $p_{\text{depol}} = 0.04 \implies Q_{\text{env}} = 0.02$, $\sigma_{\text{calib}} = \sqrt{0.02 \times 0.98 / 1000} \approx 0.00443$ ($0.44\%$).
- In the condition vector $\mathbf{c} = [Q_{\text{budget}}, \hat{\gamma}_{\text{noise}}]^T$, both entries share the identical physical unit of sifted error rates, rendering the remaining allowable error budget directly accessible:
  $$\Delta Q_{\text{allowable}} \approx Q_{\text{budget}} - \hat{\gamma}_{\text{noise}}$$

### 2.3 Inability to Distinguish Noise Families (Explicit Baseline Limitation)

A single scalar estimate $\hat{\gamma}_{\text{noise}}$ compresses all environmental decoherence into a one-dimensional magnitude. However, physical NISQ channels exhibit distinct tensor structures:
- **Depolarizing Noise:** Isotropic bit-flips and phase-flips across all bases ($Q_Z = p/2, Q_X = p/2$).
- **Pure Phase Damping:** Coherence decay without energy relaxation ($Q_Z = 0.0, Q_X = \frac{1 - \sqrt{1-\lambda}}{2}$). Computational $Z$-basis is completely undisturbed.
- **Amplitude Damping:** Energy dissipation $|1\rangle \to |0\rangle$ ($Q_Z = f_1 \gamma, Q_X = \frac{1 - \sqrt{1-\gamma}}{2}$). Asymmetric bit-flips in $Z$ and loss in $X$.

**Explicit Scientific Limitation:**
> **Scientific Limitation Notice:** A scalar estimate $\hat{\gamma}_{\text{noise}}$ cannot communicate basis asymmetry ($Q_Z \neq Q_X$) or distinguishing Kraus operator representations to the generator. Therefore, **we make no claim of zero-shot adaptation** to non-depolarizing noise families (such as Amplitude Damping or Phase Damping). Evaluating the classical cGAN on these unseen noise families is conducted explicitly as a stress test to expose the failure modes of scalar classical conditioning, providing direct empirical motivation for the genuine quantum conditional GAN (cQGAN) developed in Milestone 4.

---

## 3. Dataset Sampling Grid, Repeated Seeds & Partitioning

### 3.1 Exact Table of Attack Configurations

The dataset sampling grid evaluates 16 unique attack configurations across physical interaction libraries:

| Attack ID | Family | $\alpha$ | $\beta$ | $\theta_E$ (rad) | $\theta_E$ (deg) | Theoretical $D(\text{Attack})$ | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ATT-00** | **Null** | $0.00$ | $0.50$ | $0.0000$ | $0.00^\circ$ | $0.0000$ | Unattacked reference channel (No Eve) |
| **ATT-01** | **Probe** | --- | --- | $0.0500$ | $2.86^\circ$ | $\sin^2(0.05) \approx 0.0025$ | Weak unitary entangling probe |
| **ATT-02** | **Probe** | --- | --- | $0.1000$ | $5.73^\circ$ | $\sin^2(0.10) \approx 0.0099$ | Low-disturbance probe |
| **ATT-03** | **Probe** | --- | --- | $0.1500$ | $8.59^\circ$ | $\sin^2(0.15) \approx 0.0223$ | Moderate probe |
| **ATT-04** | **Probe** | --- | --- | $0.2000$ | $11.46^\circ$ | $\sin^2(0.20) \approx 0.0395$ | Medium-intensity probe |
| **ATT-05** | **Probe** | --- | --- | $0.2500$ | $14.32^\circ$ | $\sin^2(0.25) \approx 0.0612$ | Near-threshold probe for $Q_{\text{budget}}=0.08$ |
| **ATT-06** | **Probe** | --- | --- | $0.3000$ | $17.19^\circ$ | $\sin^2(0.30) \approx 0.0873$ | High-intensity probe |
| **ATT-07** | **Probe** | --- | --- | $0.3398$ | $19.47^\circ$ | $\sin^2(0.3398) \approx 0.1100$ | Critical threshold probe ($D = 11.0\%$) |
| **ATT-08** | **PIR** | $0.05$ | $0.50$ | --- | --- | $\alpha / 4 = 0.0125$ | Intercept $5\%$ with unbiased basis |
| **ATT-09** | **PIR** | $0.10$ | $0.50$ | --- | --- | $\alpha / 4 = 0.0250$ | Intercept $10\%$ with unbiased basis |
| **ATT-10** | **PIR** | $0.20$ | $0.50$ | --- | --- | $\alpha / 4 = 0.0500$ | Intercept $20\%$ with unbiased basis |
| **ATT-11** | **PIR** | $0.30$ | $0.50$ | --- | --- | $\alpha / 4 = 0.0750$ | Intercept $30\%$ with unbiased basis |
| **ATT-12** | **PIR** | $0.40$ | $0.50$ | --- | --- | $\alpha / 4 = 0.1000$ | Intercept $40\%$ with unbiased basis |
| **ATT-13** | **PIR** | $0.44$ | $0.50$ | --- | --- | $\alpha / 4 = 0.1100$ | Critical PIR threshold ($\alpha=0.44 \implies D=11\%$) |
| **ATT-14** | **PIR** | $0.20$ | $0.30$ | --- | --- | $\approx 0.0500$ (biased) | Intercept $20\%$ with $X$-basis bias ($\beta=0.3$) |
| **ATT-15** | **PIR** | $0.20$ | $0.70$ | --- | --- | $\approx 0.0500$ (biased) | Intercept $20\%$ with $Z$-basis bias ($\beta=0.7$) |

**Distribution of Attack Families:**
- 1 Null attack configuration (`ATT-00`)
- 8 PIR configurations (`ATT-08` through `ATT-15`: 6 symmetric + 2 biased)
- 7 Unitary probe configurations (`ATT-01` through `ATT-07`)
- Total: 16 unique attack configurations.

### 3.2 Environmental Conditions Grid

The environmental conditions grid $\mathcal{C}$ comprises 24 unique configurations:
- **Depolarizing Noise Rates (8 levels):**
  $$p_{\text{depol}} \in \{0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07\}$$
  $$\implies Q_{\text{env}} = \frac{p_{\text{depol}}}{2} \in \{0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035\}$$
- **Target Error Budgets (3 levels):**
  $$Q_{\text{budget}} \in \{0.05, 0.08, 0.11\}$$
  (representing conservative, intermediate, and standard BB84 abort thresholds).
- **Total Unique Environmental Conditions:** $|\mathcal{C}| = 8 \times 3 = 24$.
- **Total Unique Condition-Policy Pairs:** $24 \text{ conditions} \times 16 \text{ attacks} = \mathbf{384\text{ unique configurations}}$.

### 3.3 Repeated Seeds & Finite-Shot Uncertainty Estimation

To directly measure finite-shot statistical variability, every unique configuration $(c, a)$ is executed across $R = 3$ independent simulation runs using distinct random seeds:
$$\text{Seeds: } S \in \{101, 102, 103\}$$

- **Total Simulator Executions:**
  $$N_{\text{sim}} = 384 \text{ configurations} \times 3 \text{ seeds} = \mathbf{1,152\text{ executions}}$$

### 3.4 Zero-Leakage Condition-Level Partitioning

To strictly prevent data leakage across dataset partitions:
- **Partitioning Rule:** Splitting into Train, Validation, and Test sets is performed **strictly at the environmental condition tuple level $c = (p, Q_{\text{budget}})$**. Repeated seeds of the same condition are **never** split across partitions.
- **Disjoint Partition Sets:**
  - $\mathcal{C}_{\text{train}}$ (16 conditions, $66.7\%$):
    $p \in \{0.00, 0.02, 0.04, 0.06\} \times Q_{\text{budget}} \in \{0.05, 0.08, 0.11\}$ plus $\{0.01, 0.03, 0.05, 0.07\} \times \{0.08\}$. Total: 16 conditions $\times 16$ attacks $\times 3$ seeds = $768$ runs.
  - $\mathcal{C}_{\text{val}}$ (4 conditions, $16.7\%$):
    $p \in \{0.01, 0.03\} \times Q_{\text{budget}} \in \{0.05, 0.11\}$. Held-out interpolation conditions. Total: 4 conditions $\times 16$ attacks $\times 3$ seeds = $192$ runs.
  - $\mathcal{C}_{\text{test}}$ (4 conditions, $16.7\%$):
    $p \in \{0.05, 0.07\} \times Q_{\text{budget}} \in \{0.05, 0.11\}$. Held-out high-noise test conditions. Total: 4 conditions $\times 16$ attacks $\times 3$ seeds = $192$ runs.
- **Guarantee:** $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{val}} \cap \mathcal{C}_{\text{test}} = \emptyset$. No test condition shares either a training condition or repeated simulation seeds.

---

## 4. Formal Definition of the Learned Conditional Distribution

We explicitly distinguish the six distinct operational stages of the generative eavesdropping pipeline:

```
  +---------------------------------------------------------------------------------+
  | STAGE A: Dataset Construction & Feasibility Filtering                           |
  | Raw Simulations (1,152 runs) -> Feasibility Filter (Q_UCB <= Q_budget)          |
  +---------------------------------------------------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | STAGE B: Target Distribution & cGAN Density Learning                            |
  | Training Exemplars: p_target(a|c) -> Minimax Training: G_phi(z, c)              |
  +---------------------------------------------------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | STAGE C: Candidate Generation at Runtime                                        |
  | Eve samples z_1 ... z_16 ~ N(0, I) -> Proposes 16 Candidates {a_1, ..., a_16}   |
  +---------------------------------------------------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | STAGE D: Physical BB84 Forward Evaluation                                       |
  | 16 Simulator Executions under Channel c -> Measures (Q_hat, Q_UCB, I_AE)        |
  +---------------------------------------------------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | STAGE E: Statistical Feasibility Filtering                                      |
  | Feasible Pool: F(c) = { a_k | Q_UCB_k <= Q_budget }                             |
  +---------------------------------------------------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | STAGE F: Final Candidate Ranking & Tri-Case Operational Selection               |
  | Case A: argmax I_AE | Case B: Null Attack | Case C: Null Reference Infeasible   |
  +---------------------------------------------------------------------------------+
```

### 4.1 Stage A & B: Target Distribution & Divergence Minimization

The offline simulation sweep populates the raw dataset $\mathcal{D}_{\text{raw}} = \{(c_i, a_i, \bar{Q}_i, Q_{\text{LCB}, i}, Q_{\text{UCB}, i}, \bar{I}_{AE, i})\}_{i=1}^{384}$. The training partition is filtered to retain only statistically feasible exemplars:
$$\mathcal{D}_{\text{feas}} = \left\{ (c_i, a_i) \in \mathcal{D}_{\text{raw}} \;\middle|\; c_i \in \mathcal{C}_{\text{train}} \land Q_{\text{UCB}}(a_i; c_i) \le Q_{\text{budget}, i} \right\}$$

The target training distribution $p_{\text{target}}(a \mid c)$ is defined as the empirical distribution over feasible attacks:
$$p_{\text{target}}(a \mid c) = \frac{1}{Z(c)} \sum_{(c_i, a_i) \in \mathcal{D}_{\text{feas}}, c_i = c} \exp\left(\frac{\bar{I}_{AE}(a_i)}{\tau_I}\right) \delta(a - a_i)$$

**Mathematical Training Objective:** The discriminator $D_\psi(a, c)$ and generator $G_\phi(z, c)$ optimize the standard non-saturating conditional minimax game:
$$\mathcal{L}_D = -\mathbb{E}_{(c, a) \sim \mathcal{D}_{\text{feas}}} \left[ \log D_\psi(\mathbf{a}_{\text{masked}}, c) \right] - \mathbb{E}_{c \sim \mathcal{C}_{\text{train}}, z \sim \mathcal{N}(0, I)} \left[ \log\left(1 - D_\psi(G_\phi(z, c), c)\right) \right]$$
$$\mathcal{L}_G = -\mathbb{E}_{c \sim \mathcal{C}_{\text{train}}, z \sim \mathcal{N}(0, I)} \left[ \log D_\psi(G_\phi(z, c), c) \right]$$

> **Critical Mathematical Distinction:** GAN minimax training minimizes the Jensen-Shannon divergence between the generated candidate distribution $p_G(a \mid c)$ and the reference distribution $p_{\text{target}}(a \mid c)$. **GAN training itself does not directly compute or maximize an analytical mutual information gradient $\nabla_\phi I_{AE}$.** Information maximization is achieved through the construction of the weighted target distribution $p_{\text{target}}$ and by the post-generation ranking engine (Stage F).

### 4.2 Tri-Case Operational Fallback Logic (Stage F)

At runtime, the generator proposes candidate pool $\{a_1, \dots, a_{16}\}$. Each candidate is evaluated through the BB84 simulator, forming the feasible candidate pool $\mathcal{F}(c) = \{a_k \mid Q_{\text{UCB}}(a_k; c) \le Q_{\text{budget}}\}$. 

We strictly distinguish three mutually exclusive operational cases:

```
+--------------------------------------------------------------------------------------------------+
|                               TRI-CASE OPERATIONAL FALLBACK LOGIC                                |
+--------+----------------------------+-----------------------+------------------------------------+
| Case   | Condition Status           | Policy Selection a*   | Protocol & Security Outcome        |
+--------+----------------------------+-----------------------+------------------------------------+
| Case A | Feasible non-null policy   | a* = argmax I_AE(a_k) | Active candidate satisfies         |
|        | exists: F(c) \ {a_null} != ∅| for a_k in F(c)       | Q_UCB <= Q_budget; I_AE > 0        |
+--------+----------------------------+-----------------------+------------------------------------+
| Case B | No non-null feasible, but  | a* = a_null           | Null reference satisfies           |
|        | null is feasible:          | (θ_E = 0, α = 0)      | Q_UCB <= Q_budget; I_AE = 0        |
|        | a_null in F(c)             |                       |                                    |
+--------+----------------------------+-----------------------+------------------------------------+
| Case C | Null-attack reference      | Protocol aborts under | NULL REFERENCE INFEASIBLE;         |
|        | condition is infeasible:   | null reference;       | The null-attack reference condition|
|        | a_null NOT in F(c)         | a* = a_null (forced)  | is statistically infeasible;       |
|        |                            |                       | Alice & Bob abort under reference. |
+--------+----------------------------+-----------------------+------------------------------------+
```

1. **Case A (Active Policy Feasible):** At least one active attack satisfies the statistical feasibility criterion $Q_{\text{UCB}} \le Q_{\text{budget}}$. Eve selects $a^* = \arg\max_{a_k \in \mathcal{F}(c)} \hat{I}_{AE}(a_k)$, extracting positive mutual information.
2. **Case B (Null-Attack Feasible, Active Infeasible):** No active candidate satisfies the feasibility criterion, but the null-attack reference condition satisfies $Q_{\text{UCB}}(a_{\text{null}}; c) \le Q_{\text{budget}}$. Eve selects $a^* = a_{\text{null}}$ ($\alpha = 0, \theta_E = 0$). Alice and Bob pass under the null condition, but Eve extracts $I_{AE} = 0$.
3. **Case C (Null-Attack Reference Condition Infeasible):** The null-attack reference condition is statistically infeasible under the configured QBER budget ($Q_{\text{LCB}}(a_{\text{null}}; c) > Q_{\text{budget}}$). Alice and Bob abort under this null-attack reference condition. We do not claim that every possible active attack policy must produce the same QBER without separately establishing that property. The session is intrinsically unviable under the reference condition; Eve extracts $I_{AE} = 0$.

### 4.3 Acknowledgment: Primary Grid Boundaries vs. Case C
In our defined primary sampling grid:
$$p_{\text{depol}} \in [0.00, 0.07] \implies Q_{\text{env}} = \frac{p_{\text{depol}}}{2} \in [0.000, 0.035]$$
$$Q_{\text{budget}} \in \{0.05, 0.08, 0.11\}$$
- Because $\max(Q_{\text{env}}) = 0.035 < \min(Q_{\text{budget}}) = 0.05$, the condition $Q_{\text{env}} \ge Q_{\text{budget}}$ is **never satisfied** in the primary dataset grid.
- **Explicit Acknowledgment:** The primary training grid does not exercise Case C or the complete collapse of $p_{\text{target}}(a \mid c)$ to $a_{\text{null}}$. Instead, Case C and strict distribution collapse are evaluated strictly during **Tier 2 Parameter Extrapolation** ($p_{\text{depol}} \in \{0.08, 0.10, 0.12\} \implies Q_{\text{env}} \in \{0.04, 0.05, 0.06\}$ evaluated against $Q_{\text{budget}} = 0.05$).

---

## 5. Fair Baseline Comparison under an Equal Simulation Query Budget

To establish whether the cGAN candidate generator provides genuine scientific utility over classical search methods, all baselines are constrained to an **identical query budget of $K = 16$ BB84 simulation queries** per evaluation condition:

```
+--------------------------------------------------------------------------------------------------+
|                            QUERY-BUDGET COMPARISON PROTOCOL (K = 16)                             |
+--------------------------+------------------------------------+----------------------------------+
| Method                   | Candidate Proposal Mechanism       | Query Allocation Rule            |
+--------------------------+------------------------------------+----------------------------------+
| 1. cGAN Generator        | 16 independent latent vectors      | 16 unique candidate policy       |
|                          | z_1 ... z_16 ~ N(0, I) -> G_phi    | forward simulations              |
+--------------------------+------------------------------------+----------------------------------+
| 2. Uniform Random Search | 16 random points sampled           | 16 unique candidate policy       |
|                          | uniformly across valid parameter   | forward simulations              |
|                          | bounds: w ~ Bern(0.5),             |                                  |
|                          | α,β ~ U[0,1], θ_E ~ U[0, π/4]      |                                  |
+--------------------------+------------------------------------+----------------------------------+
| 3. Deterministic Grid    | Fixed 4x4 uniform lattice:         | 16 unique candidate policy       |
|    Search                | 8 PIR (α in {0.05,0.15,0.25,0.35}  | forward simulations              |
|                          |        x β in {0.5, 0.7}) +        |                                  |
|                          | 8 Probe (θ_E in 8 equidistant pts) |                                  |
+--------------------------+------------------------------------+----------------------------------+
| 4. Analytical Probe      | Approximate heuristic probe:       | Single heuristic candidate       |
|    Heuristic             | θ_E = arcsin(sqrt(max(0,           | evaluated across 16 independent  |
|                          |       Q_budget - γ_hat)))          | simulation batches               |
+--------------------------+------------------------------------+----------------------------------+
```

### 5.1 Formulation of the Analytical Probe Heuristic
- **Baseline Name:** **Analytical Probe Heuristic** (formerly "Optimal Analytical Policy").
- **Mathematical Specification:**
  $$\theta_E^{\text{heuristic}} = \arcsin\left(\sqrt{\max\left(0.0, \; Q_{\text{budget}} - \hat{\gamma}_{\text{noise}}\right)}\right)$$
  where $\hat{\gamma}_{\text{noise}} = \hat{Q}_{\text{calib}}$ is the estimated ambient channel QBER.
- **Physical Caveats & Scope:**
  1. **Approximate Initialization:** This formula is derived under an idealized additive disturbance assumption ($Q_{\text{total}} \approx Q_{\text{env}} + \sin^2\theta_E$), ignoring nonlinear compounding cross-terms ($Q_{\text{total}} = Q_{\text{env}} + D - 2 Q_{\text{env}} D$), finite-sample variance margins, and asymmetric channel noise. **It is not globally optimal.**
  2. **Simulation Validation:** The candidate proposed by this heuristic is **not accepted blind**; it is evaluated directly through the BB84 simulator and subjected to the exact Wilson score feasibility test ($Q_{\text{UCB}} \le Q_{\text{budget}}$).
  3. **Family Scope:** This heuristic covers **exclusively the Unitary Probe family**. It does not propose PIR attacks (as an equivalent PIR heuristic $\alpha^{\text{heuristic}} = 4 \max(0, Q_{\text{budget}} - \hat{\gamma}_{\text{noise}})$ is not standard).

### 5.2 Selection and Post-Selection Validation Protocol
1. **Query Budget Allocation:** The $K = 16$ queries are strictly consumed during the **candidate selection stage**. Each candidate $a_k \in \{a_1, \dots, a_{16}\}$ receives exactly one simulation run ($N_{\text{pulse}} = 2,000$).
2. **Selection Rule:** Every method selects its winning policy using the identical selection logic:
   $$a^* = \begin{cases} \arg\max_{a_k \in \mathcal{F}(c)} \hat{I}_{AE}(a_k) & \text{if } \mathcal{F}(c) \neq \emptyset \\ a_{\text{null}} & \text{if } \mathcal{F}(c) = \emptyset \end{cases}$$
3. **Post-Selection Validation:** Once winning policy $a^*$ is selected, it is evaluated across an independent evaluation suite of $R_{\text{eval}} = 5$ runs with independent evaluation seeds ($S_{\text{eval}} \in \{901, \dots, 905\}$) to measure mean information gain $\bar{I}_{AE}(a^*)$, empirical abort rate $\hat{P}_{\text{abort}}$, and selection stability. Post-selection evaluation is an offline benchmarking step and does not participate in candidate selection.

---

## 6. Empirical Computational Pilot & Recalculated Runtimes

### 6.1 Pilot Execution & Hardware Environment Verification

A diagnostic benchmark script was executed on the target environment to eliminate speculative hardware assumptions.

**Hardware & Backend Verification:**
- `qiskit_aer.AerSimulator`: Available devices: `('CPU', )`. Backend name: `aer_simulator`.
  - **Result:** Qiskit Aer 0.18.0 is compiled for CPU OpenMP multi-threading. It does **not** execute quantum simulation on CUDA GPU.
- `torch.cuda.is_available()`: `True`. Device: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`.
  - **Result:** PyTorch neural network training and tensor operations execute on CUDA GPU.

### 6.2 Empirical Measurements at 3,000 Pulses

Measurements were performed on the verified BB84 engine (`BB84Engine.run`) across three attack conditions under depolarizing noise ($p = 0.04$) at $N_{\text{pulse}} = 3,000$ pulses:

```
+--------------------------------------------------------------------------------------------------+
|                            EMPIRICAL MEASUREMENTS (N_pulse = 3,000)                              |
+-----------------------+--------------------+---------------------+-------------------------------+
| Configuration         | Pulses Transmitted | Execution Time (s)  | Simulation Throughput         |
+-----------------------+--------------------+---------------------+-------------------------------+
| 1. Null Attack        | 3,000 pulses       | 1.1838 s            | 2,534.2 pulses / second       |
| 2. PIR Attack         | 3,000 pulses       | 1.6289 s            | 1,841.7 pulses / second       |
| 3. Unitary Probe      | 3,000 pulses       | 1.8076 s            | 1,659.7 pulses / second       |
+-----------------------+--------------------+---------------------+-------------------------------+
| Arithmetic Mean       | 3,000 pulses       | 1.5401 s            | (1.1838 + 1.6289 + 1.8076)/3  |
+-----------------------+--------------------+---------------------+-------------------------------+
| System RAM Usage      | Baseline: 399.4 MB | Peak: 447.8 MB      | Delta: +48.4 MB               |
| Aer GPU VRAM Usage    | Baseline: 0.0 MB   | Peak: 0.0 MB        | Delta: 0.0 MB (CPU execution) |
+-----------------------+--------------------+---------------------+-------------------------------+
```

### 6.3 Weighted Average Runtime across Dataset Attack Distribution

The dataset evaluates 16 unique attack configurations:
- 1 Null attack configuration (`ATT-00`)
- 8 PIR configurations (`ATT-08` through `ATT-15`)
- 7 Unitary probe configurations (`ATT-01` through `ATT-07`)

The true weighted average runtime per simulation run at $N_{\text{pulse}} = 3,000$ is:
$$\bar{t}_{\text{weighted, 3000}} = \frac{1 \times 1.1838\,\text{s} + 8 \times 1.6289\,\text{s} + 7 \times 1.8076\,\text{s}}{16} = \frac{26.8682\,\text{s}}{16} \approx \mathbf{1.6793\text{ seconds}}$$

### 6.4 Total Dataset Generation Runtime Calculations

The complete dataset requires $N_{\text{sim}} = 24 \text{ conditions} \times 16 \text{ attacks} \times 3 \text{ seeds} = \mathbf{1,152\text{ executions}}$.

#### A. Empirical 3,000-Pulse Dataset Runtime
- Direct execution time:
  $$T_{\text{direct, 3000}} = 1,152 \times 1.6793\,\text{s} \approx 1,934.5\,\text{seconds} \approx \mathbf{32.24\text{ minutes}}$$
- Expected execution time with $10\%$ Python serialization and logging overhead:
  $$T_{\text{total, 3000, overhead}} = 32.24 \times 1.10 \approx \mathbf{35.5\text{ minutes}}$$

#### B. Linearly Extrapolated 2,000-Pulse Dataset Runtime
Assuming linear scaling with pulse count (scaling factor $2,000 / 3,000 = 2/3$):
- Extrapolated runtimes per run:
  - Null Attack: $1.1838 \times \frac{2}{3} \approx 0.7892\,\text{s}$
  - PIR Attack: $1.6289 \times \frac{2}{3} \approx 1.0859\,\text{s}$
  - Unitary Probe: $1.8076 \times \frac{2}{3} \approx 1.2051\,\text{s}$
- Extrapolated weighted average runtime:
  $$\bar{t}_{\text{weighted, 2000, extrap}} = 1.6793 \times \frac{2}{3} \approx \mathbf{1.1195\text{ seconds}}$$
- Direct execution time:
  $$T_{\text{direct, 2000, extrap}} = 1,152 \times 1.1195\,\text{s} \approx 1,289.7\,\text{seconds} \approx \mathbf{21.50\text{ minutes}}$$
- Expected execution time with $10\%$ overhead:
  $$T_{\text{total, 2000, overhead}} = 21.50 \times 1.10 \approx \mathbf{23.65\text{ minutes}}$$

- **Dataset Storage Footprint:** 1,152 JSON records $\times \approx 320$ bytes/record $\approx \mathbf{368\text{ KB}}$, fully resident in RAM with zero disk I/O bottlenecks.

---

## 7. Multi-Metric Validation Suite (Beyond Discriminator Loss)

Discriminator loss $\mathcal{L}_D$ oscillates in adversarial minimax games and is mathematically uncorrelated with generative utility or stealth compliance. Model checkpointing and early stopping are governed strictly by five quantitative metrics evaluated on held-out validation conditions $\mathcal{C}_{\text{val}}$:

```
+--------------------------------------------------------------------------------------------------+
|                                    VALIDATION METRIC SUITE                                       |
+-----+---------------------------+-----------------------------------------+----------------------+
| ID  | Metric                    | Mathematical Definition                 | Target Threshold     |
+-----+---------------------------+-----------------------------------------+----------------------+
| M1  | Feasibility Yield         | Fraction of generated candidates with   | Y_feas >= 75%        |
|     | (Y_feas)                  | Q_UCB <= Q_budget                       |                      |
+-----+---------------------------+-----------------------------------------+----------------------+
| M2  | Generative Pareto         | I_theoretical*(ΔQ) -                    | E_Pareto <= 0.03     |
|     | Proximity (E_Pareto)      | max_{a in F(c)} I_AE(a)                 | bits                 |
+-----+---------------------------+-----------------------------------------+----------------------+
| M3  | Diversity Continuous      | Mean pairwise distance across generated | D_pw >= 0.15         |
|     | Distance (D_pw)           | parameter vectors                       |                      |
+-----+---------------------------+-----------------------------------------+----------------------+
| M4  | Family Selection          | Entropy across attack family choices    | H_fam >= 0.50 bits   |
|     | Entropy (H_fam)           | H = -sum(w_k * log2(w_k))               | (Anti-mode-collapse) |
+-----+---------------------------+-----------------------------------------+----------------------+
| M5  | Physical Validity         | Fraction strictly within physical       | V_phys = 100.0%      |
|     | Rate (V_phys)             | parameter bounds                        | (Strict)             |
+-----+---------------------------+-----------------------------------------+----------------------+
```

### Composite Checkpoint Score
The checkpoint selection criterion across training epochs is the composite score:
$$\mathcal{S}_{\text{val}} = \frac{1}{|\mathcal{C}_{\text{val}}|} \sum_{c \in \mathcal{C}_{\text{val}}} \left[ Y_{\text{feas}}(c) \cdot \max_{a_k \in \mathcal{F}(c)} \bar{I}_{AE}(a_k) \right]$$
This scalar reward balances generating stealthy attacks against maximizing extracted information.

---

## 8. Summary Comparison of Design Revisions

| Dimension | Previous Draft | Audited Final Stage A Specification | Scientific Justification |
| :--- | :--- | :--- | :--- |
| **Statistical Interval** | Approximate Wald interval | Exact Wilson Score Interval at $95\%$ confidence | Prevents negative bounds and severe undercoverage at small $Q$. |
| **Bit Accounting** | Pulses and test bits blurred | Clear hierarchy: $N_{\text{pulse}} \to N_{\text{sift}} \to N_{\text{test}}$ | Essential for rigorous finite-sample error rate estimation. |
| **Feasibility States** | Naive $Q \le Q_{\text{budget}} + 0.015$ with speculative probabilities | Tri-state partition (*Feasible*, *Infeasible*, *Indeterminate*) with empirical pass/abort frequencies | Eliminates ungrounded probability claims; anchors decisions in protocol rule. |
| **Threat Model** | Ambiguous channel observability | 2D vector $[Q_{\text{budget}}, \hat{\gamma}_{\text{noise}}]$; explicit limitation on scalar noise | Prevents unphysical clairvoyance; formally bounds classical baseline claims. |
| **Noise Estimator** | Equated $\hat{\gamma}$ with $p_{\text{depol}}$ | Disambiguated $p_{\text{depol}}$, $Q_{\text{env}}$, $\hat{Q}_{\text{calib}}$, $\hat{\gamma}_{\text{noise}}$ | Mathematical rigor: $\hat{\gamma} = \hat{Q}_{\text{calib}}$ has expectation $p/2$. |
| **Fallback Taxonomy** | Simple fallback to $a_{\text{null}}$ | Tri-case taxonomy (Cases A, B, C); acknowledged grid does not reach Case C | Rigorous operational semantics; does not overclaim active attack QBER invariance in Case C. |
| **Analytical Baseline** | "Optimal Analytical Policy" | "Analytical Probe Heuristic" (approximate probe rule) | Avoids overclaiming global optimality; explicitly covers only Unitary Probe. |
| **Sampling Table** | Approximate parameter ranges | Exact 16-point attack table with explicit $(\alpha, \beta, \theta_E)$ | Guaranteed reproducibility and deterministic grid geometry. |
| **Variability & Seeds** | Single-seed execution | $R = 3$ independent seeds per configuration (1,152 total runs) | Directly estimates finite-shot variance and empirical abort probabilities. |
| **Data Partitioning** | Random row splitting | Condition-level splitting ($\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{val}} \cap \mathcal{C}_{\text{test}} = \emptyset$) | Prevents data leakage between repeated simulation seeds. |
| **Baseline Benchmarking**| Unequal comparisons | Identical $K = 16$ simulation query budget across all baselines | Guarantees fair, statistically valid empirical comparison. |
| **Runtime Calculation** | Unweighted mean claimed as 2,000 pulses | Weighted mean $\bar{t} \approx 1.679\,\text{s}$ at 3,000 pulses; extrapolated to 2,000 | Separates empirical reality ($32.2\text{ min}$) from linear extrapolation ($21.5\text{ min}$). |

---

## 9. Stage Boundary & Sign-off Request

This document represents the finalized, fully audited scientific design for **Milestone 3: Stage A**, resolving all specific audit items.

**Report-Only Revision Confirmation:**
- No source code files in `src/`, `tests/`, or `experiments/` were modified.
- No neural network model implementations (`src/cgan/`) have been created.
- No dataset generation scripts (`experiments/generate_m3_dataset.py`) have been executed.
- Existing Milestone 1 and Milestone 2 implementations, tests, and experimental results remain untouched and fully passing (historical context: **97/97 tests pass**).

We request your review and final approval of this revised Stage A scientific specification to commence **Stage B: Code Implementation, Unit Testing, and Pilot Validation**.