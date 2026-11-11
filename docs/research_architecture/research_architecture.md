# BB84-QGAN: Research Architecture Specification

## 1. Executive Summary & Objective

This research investigates resource-constrained adversarial eavesdropping in Quantum Key Distribution (QKD) protocols under realistic Noisy Intermediate-Scale Quantum (NISQ) environments. Specifically, we examine how a conditional generative model (classical conditional GAN and variational quantum GAN) can learn to infer optimal, stealthy eavesdropping policies against the BB84 protocol without triggering the protocol abort threshold.

The primary scientific challenge is navigating the fundamental physics trade-off in quantum measurement: extracting maximal mutual information $I(A:E)$ regarding the Alice-Bob raw key while constraining the induced Quantum Bit Error Rate (QBER) strictly below the detection abort threshold $QBER_{\text{tol}}$:

$$\max_{\vec{a} \in \mathcal{A}} I(A:E)(\vec{a}; \vec{c}) \quad \text{subject to} \quad QBER(\vec{a}; \vec{c}) \le QBER_{\text{tol}}$$

where $\vec{c} = (p_{\text{noise}}, QBER_{\text{tol}})$ represents the environmental noise condition.

---

## 2. Four-Tier System Architecture

The research framework is implemented across four tightly integrated, modular layers:

```
+-----------------------------------------------------------------------------------+
|                        Tier 4: Adversarial Policy Optimization                     |
|   Classical cGAN (MLP Generator)  |  Variational Quantum GAN (PQC Generator)     |
|   Inputs: Latent z ~ N(0, I_16), Condition c = [p_noise, QBER_tol]               |
|   Outputs: Policy candidate a = [p_intercept, theta_E, attack_family]             |
+-----------------------------------------------------------------------------------+
                                       |
                                       v Policy Candidate Selection (Budget K=16)
+-----------------------------------------------------------------------------------+
|                        Tier 3: Policy Engine & Evaluation Harness                  |
|   - Policy Validation & Feasibility Filter (Wilson UCB_95(QBER) <= QBER_tol)      |
|   - 4-Method Benchmark: cGAN, Uniform Random, Grid Search, Analytical Probe       |
|   - Common Random Numbers (CRN) for Rigorous Variance Reduction                  |
+-----------------------------------------------------------------------------------+
                                       |
                                       v Simulated Pulse Transmission
+-----------------------------------------------------------------------------------+
|                        Tier 2: Parametric Quantum Channel & NISQ Noise            |
|   - Depolarizing Channel: E(rho) = (1-p)rho + (p/3)(X rho X + Y rho Y + Z rho Z)   |
|   - Amplitude Damping Channel (Photon Loss / Relaxation: Kraus K0, K1)            |
|   - Phase Damping Channel (Dephasing: Kraus K0, K1)                               |
|   - Thermal Relaxation (Lindblad T1 / T2 Master Equation)                        |
+-----------------------------------------------------------------------------------+
                                       |
                                       v 2D Hilbert Space States
+-----------------------------------------------------------------------------------+
|                        Tier 1: Physical BB84 Simulation Engine                    |
|   - Alice: Polarization state prep in Z {|0>, |1>} and X {|+>, |->}               |
|   - Bob: Projective measurements in randomly chosen conjugate bases               |
|   - Classical Sifting: Basis reconciliation, sample QBER estimation, abort check  |
+-----------------------------------------------------------------------------------+
```

---

## 3. Data Flow & Condition Vector Pipeline

1. **Environmental Characterization:** The channel environment is specified by vector $\vec{c} = (p_{\text{noise}}, QBER_{\text{tol}}) \in [0, 0.08] \times [0.08, 0.15]$.
2. **Policy Generation:** The policy generator samples $K=16$ candidate eavesdropping parameter sets:
   $$\vec{a}_k = [p_{\text{intercept}}, \theta_E, \text{family}]_k, \quad k=1, \dots, K$$
3. **Execution & Measurement:** In simulation, Alice transmits $N$ polarized photons ($N=2,000$ per pulse evaluation). Eve intercepts or entangles according to $\vec{a}_k$. Bob measures in random bases.
4. **Sifting & Verification:** Sifted bits are compared over an authenticated public channel. Alice and Bob compute empirical error rate $\hat{E}$.
5. **Feasibility Gate:** A policy is strictly *feasible* if and only if:
   $$\text{UCB}_{95\%}(\hat{E}) \le QBER_{\text{tol}} \quad \text{and} \quad \text{Abort} = 0$$

---

## 4. Benchmark Partitions & Strict Experimental Isolation

To prevent data contamination and ensure publication-grade statistical validity, the 24 environmental conditions are strictly partitioned:
- **Training Set (C01–C16):** 16 conditions used for offline dataset generation and cGAN backpropagation.
- **Validation Set (C17–C20):** 4 conditions used strictly for epoch evaluation, learning rate monitoring, and lexicographic checkpoint selection.
- **Held-Out Test Set (C21–C24):** 4 conditions completely quarantined until the final four-method comparative benchmark.
