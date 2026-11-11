# Milestone 2 Technical Research Report
## Realistic NISQ Noise Modelling & Quantum Channel Integration

**Project Title:** Adaptive QGAN-Based Eavesdropping in BB84 QKD: Resource-Constrained Conditional Quantum Generative Approach under NISQ Noise  
**Domain:** Quantum Cryptography, NISQ Simulation, Quantum Open Systems, Quantum Information Theory  
**Target Environment:** Python 3.11.9 | Qiskit 2.6.0 | Qiskit-Aer 0.18.0 | PyTorch 2.5.1+cu121 | NumPy 2.4.6 | SciPy 1.17.1  
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) | 16 GB Host RAM  
**Project Workspace:** `C:\Users\lenovo\bb84-qgan-research`  
**Milestone Status:** Completed & Fully Validated (97/97 Unit Tests Passing)  

---

## 1. Executive Summary

Milestone 2 establishes a mathematically rigorous, fully validated open-quantum-systems simulation framework for realistic **Noisy Intermediate-Scale Quantum (NISQ)** environments within the BB84 Quantum Key Distribution (QKD) architecture.

Prior to Milestone 2, the BB84 simulation engine (`BB84Engine`) operated exclusively in an idealized or analytically perturbed setting. In physical optical fibers and free-space quantum channels, environmental dissipation, thermal relaxation, and phase decoherence occur concurrently with potential adversarial eavesdropping. To enable realistic adversarial optimization in subsequent milestones (Milestones 3–5), the quantum channel must simulate:
1. **Depolarizing Noise:** Symmetric isotropic noise affecting all conjugate bases identically.
2. **Amplitude Damping:** Energy relaxation modeling photon loss and non-symmetric decay $|1\rangle \to |0\rangle$.
3. **Phase Damping:** Pure dephasing destroying quantum coherence while preserving computational basis populations.
4. **Thermal Relaxation:** Physical combined $T_1$ (longitudinal relaxation) and $T_2$ (transverse dephasing) decay under strict quantum positivity constraints ($T_2 \le 2 T_1$).
5. **Coupled Multi-Source Channels:** Compound noise interacting with active eavesdropping (Intercept-Resend and Entangling Unitary Probes).

All noise mechanisms have been implemented in `src/channel/noise_models.py`, integrated into the physical transmission link `src/channel/quantum_channel.py`, and verified against exact analytical Kraus models and end-to-end Monte Carlo simulations in `BB84Engine`.

---

## 2. Mandatory Pre-Implementation Audit & Mathematical Correction

### 2.1 The Fuchs-Peres-Brandt Unitary Probe Disturbance Discrepancy

A critical pre-implementation audit was conducted on the Fuchs-Peres-Brandt probe results from Milestone 1B:
- **Historical Report Text:** Mentioned $P_{\text{error}} = \sin^2(\theta_E)$.
- **Code Function in `src/bb84/attacks.py`:** Defined $D(\theta_E) = \frac{1}{2}\sin^2(\theta_E)$.
- **Observed Simulation Output in `results/bb84_simulation_results.json`:** Produced $\text{QBER} \approx 0.147989$ at $\theta_E = \pi / 8 = 22.5^\circ$, which is approximately double $0.073223$.

### 2.2 Rigorous Derivation from Reduced Density Matrices

Let the carrier qubit $A$ (qubit 0) interact with Eve's probe ancilla $E$ (qubit 1, initialized in $|0\rangle_E$) via the unitary operator $U_E(\theta_E)$ implemented in `build_probe_unitary`:
$$U_E(\theta_E) |0\rangle_A |0\rangle_E = \cos\theta_E |0\rangle_A |0\rangle_E + \sin\theta_E |1\rangle_A |1\rangle_E$$
$$U_E(\theta_E) |1\rangle_A |0\rangle_E = \cos\theta_E |1\rangle_A |0\rangle_E - \sin\theta_E |0\rangle_A |1\rangle_E$$

In the computational basis $\{|0\rangle, |1\rangle\}$, when Alice transmits $|0\rangle_A$, the joint state after the probe is:
$$|\Psi_{\text{out}}(0)\rangle = \cos\theta_E |0\rangle_A |0\rangle_E + \sin\theta_E |1\rangle_A |1\rangle_E$$
Tracing out Eve's ancilla subsystem $E$:
$$\rho_A(0) = \text{Tr}_E \left( |\Psi_{\text{out}}(0)\rangle \langle \Psi_{\text{out}}(0)| \right) = \cos^2\theta_E |0\rangle\langle 0| + \sin^2\theta_E |1\rangle\langle 1|$$

When Bob measures in the matching $Z$-basis, the error probability (measuring $|1\rangle$) is:
$$P_{\text{error}}(0, Z) = \langle 1| \rho_A(0) |1\rangle = \sin^2(\theta_E)$$

Similarly, when Alice transmits $|1\rangle_A$:
$$|\Psi_{\text{out}}(1)\rangle = \cos\theta_E |1\rangle_A |0\rangle_E - \sin\theta_E |0\rangle_A |1\rangle_E$$
$$\rho_A(1) = \cos^2\theta_E |1\rangle\langle 1| + \sin^2\theta_E |0\rangle\langle 0|$$
$$P_{\text{error}}(1, Z) = \langle 0| \rho_A(1) |0\rangle = \sin^2(\theta_E)$$
Thus, in the $Z$-basis, $\text{QBER}_Z = \sin^2(\theta_E)$.

Now consider the Hadamard basis $\{|+\rangle, |-\rangle\}$:
When Alice transmits $|+\rangle_A = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$:
$$|\Psi_{\text{out}}(+)\rangle = \frac{1}{\sqrt{2}} \left[ \cos\theta_E |00\rangle + \sin\theta_E |11\rangle + \cos\theta_E |10\rangle - \sin\theta_E |01\rangle \right]$$
$$= \cos\theta_E |+\rangle_A |0\rangle_E - \sin\theta_E |-\rangle_A |1\rangle_E$$
Tracing out Eve's ancilla:
$$\rho_A(+) = \cos^2\theta_E |+\rangle\langle +| + \sin^2\theta_E |-\rangle\langle -|$$
When Bob measures in the matching $X$-basis, the error probability (measuring $|-\rangle$) is:
$$P_{\text{error}}(+, X) = \langle -| \rho_A(+) |-\rangle = \sin^2(\theta_E)$$
By symmetry, for Alice transmitting $|-\rangle_A$, $P_{\text{error}}(-, X) = \sin^2(\theta_E)$.

### 2.3 Mathematical Proof of Disturbance Identity

Across all four BB84 states and both conjugate bases on sifted pulses:
$$D(\theta_E) \equiv \text{QBER} = \sin^2(\theta_E)$$

### 2.4 Consistency with the Fuchs Information-Disturbance Relation

According to Fuchs and Peres (Phys. Rev. A, 1996), Eve's mutual information bound satisfies:
$$I_{AE} = 1 - h_2\left( \frac{1 + \sqrt{1 - (1 - 2D)^2}}{2} \right)$$
Substituting $D = \sin^2(\theta_E)$:
$$1 - 2D = 1 - 2\sin^2(\theta_E) = \cos(2\theta_E)$$
$$\sqrt{1 - (1 - 2D)^2} = \sqrt{1 - \cos^2(2\theta_E)} = \sin(2\theta_E)$$
$$p_{\text{succ}} = \frac{1 + \sin(2\theta_E)}{2}$$
$$I_{AE} = 1 - h_2\left( \frac{1 + \sin(2\theta_E)}{2} \right)$$

This confirms that the information gain formula implemented in `attacks.py` ($p_{\text{succ}} = \frac{1 + \sin(2\theta_E)}{2}$) is strictly consistent with $D(\theta_E) = \sin^2(\theta_E)$, and mathematically inconsistent with $\frac{1}{2}\sin^2(\theta_E)$.

### 2.5 Audit Resolution

1. **Root Cause:** The earlier code formula `0.5 * sin^2(theta_E)` was an analytical misattribution originating from half-angle trigonometric formulas ($\sin^2\theta = \frac{1 - \cos 2\theta}{2}$) or asymmetric coupling models.
2. **Probe Circuit Integrity:** The quantum circuit and unitary matrix implementation in `attacks.py` was already physically exact and was left untouched.
3. **Correction:** The theoretical method `FuchsPeresBrandtProbe.theoretical_disturbance()` in `src/bb84/attacks.py` was updated to return $\sin^2(\theta_E)$, unifying analytical predictions and empirical simulation results.
4. **Milestone 1 Historical Preservation:** Milestone 1B JSON artifacts in `results/bb84_simulation_results.json` remain unmodified as historical records.

---

## 3. Mathematical Models & Kraus Representations

### 3.1 Depolarizing Channel

The single-qubit depolarizing channel models isotropic environmental decoherence.

#### Kraus Operators
$$\mathcal{E}_{\text{depol}}(\rho) = (1 - p)\rho + p \frac{I}{2}$$
In Kraus operator form $\mathcal{E}(\rho) = \sum_{k=0}^3 K_k \rho K_k^\dagger$:
$$K_0 = \sqrt{1 - \frac{3p}{4}} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad K_1 = \sqrt{\frac{p}{4}} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad K_2 = \sqrt{\frac{p}{4}} \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad K_3 = \sqrt{\frac{p}{4}} \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

#### Qiskit Aer Convention Audit
In Qiskit Aer `depolarizing_error(p, 1)`, input state $|0\rangle\langle 0|$ transforms to:
$$\mathcal{E}(|0\rangle\langle 0|) = \begin{pmatrix} 1 - \frac{p}{2} & 0 \\ 0 & \frac{p}{2} \end{pmatrix}$$
This confirms that Qiskit Aer adopts the $(1-p)\rho + p\frac{I}{2}$ parameterization.

#### Analytical Error Predictions
$$\text{QBER}_Z = \frac{p}{2}, \quad \text{QBER}_X = \frac{p}{2}, \quad \text{QBER}_{\text{total}} = \frac{p}{2}$$
$$\bar{F} = 1 - \frac{p}{2}$$

---

### 3.2 Amplitude Damping Channel

Models non-unitary energy dissipation (e.g., spontaneous photon emission or fiber attenuation) transitioning $|1\rangle \to |0\rangle$.

#### Kraus Operators
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$
Completeness check:
$$K_0^\dagger K_0 + K_1^\dagger K_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 - \gamma \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & \gamma \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I_2$$

#### Basis Asymmetry & Bit Distribution Dependence
For an input carrier density matrix $\rho$:
$$\mathcal{E}_{\text{amp}}(\rho) = \begin{pmatrix} \rho_{00} + \gamma \rho_{11} & \sqrt{1 - \gamma} \rho_{01} \\ \sqrt{1 - \gamma} \rho_{10} & (1 - \gamma) \rho_{11} \end{pmatrix}$$

1. **Computational Basis ($Z$):**
   - Input $|0\rangle$: Error is 0 (ground state is invariant).
   - Input $|1\rangle$: Error probability is $\gamma$.
   - For an arbitrary fraction $f_1 = \Pr(\text{Alice transmits } |1\rangle)$:
     $$\text{QBER}_Z(f_1) = f_1 \cdot \gamma$$
   - For unbiased BB84 ($f_1 = 0.5$): $\text{QBER}_Z = 0.5 \gamma$.

2. **Diagonal Basis ($X$):**
   - Inputs $|+\rangle$ and $|-\rangle$:
     $$\text{QBER}_X = \frac{1 - \sqrt{1 - \gamma}}{2}$$

3. **Total Sifted QBER:**
   $$\text{QBER}_{\text{total}} = \frac{1}{2} \left[ f_1 \gamma + \frac{1 - \sqrt{1 - \gamma}}{2} \right]$$

---

### 3.3 Phase Damping Channel

Models loss of phase coherence (pure dephasing) without energy loss or state transitions.

#### Kraus Operators
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \lambda} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}$$
Completeness check:
$$K_0^\dagger K_0 + K_1^\dagger K_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 - \lambda \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & \lambda \end{pmatrix} = I_2$$

#### Computational Basis Immunity
$$\mathcal{E}_{\text{phase}}(\rho) = \begin{pmatrix} \rho_{00} & \sqrt{1 - \lambda}\rho_{01} \\ \sqrt{1 - \lambda}\rho_{10} & \rho_{11} \end{pmatrix}$$
- Diagonal terms $\rho_{00}, \rho_{11}$ are completely invariant.
- **$\text{QBER}_Z \equiv 0.0000$** for all values of $\lambda \in [0, 1]$.
- In the $X$-basis:
  $$\text{QBER}_X = \frac{1 - \sqrt{1 - \lambda}}{2}$$
- Total QBER:
  $$\text{QBER}_{\text{total}} = \frac{1 - \sqrt{1 - \lambda}}{4}$$
- $l_1$-norm coherence: $C_{l_1}(\rho_{\text{out}}(+)) = 2 |\rho_{01}| = \sqrt{1 - \lambda}$.
- Average state fidelity: $\bar{F} = \frac{3 + \sqrt{1 - \lambda}}{4}$.

---

### 3.4 Thermal Relaxation Channel

Models physical relaxation under finite thermal environments defined by:
- $T_1$: Longitudinal (energy) relaxation time.
- $T_2$: Transverse (dephasing) relaxation time.
- $t$: Channel transit / gate duration.
- $p_{\text{excited}}$: Equilibrium thermal population ($p_{\text{excited}} = 0$ for optical frequencies $\hbar\omega \gg k_B T$).

#### Positivity Constraint
The physical boundary for complete positivity mandates:
$$T_2 \le 2 T_1$$
Violations ($T_2 > 2 T_1$) correspond to non-positive dynamical maps and are rejected with `ValueError`.

#### Analytical Error Formulations ($p_{\text{excited}} = 0$)
- Longitudinal relaxation parameter: $\gamma = 1 - e^{-t / T_1}$.
- Transverse coherence decay factor: $e^{-t / T_2}$.
- Basis-specific error rates:
  $$\text{QBER}_Z = \frac{1 - e^{-t / T_1}}{2}$$
  $$\text{QBER}_X = \frac{1 - e^{-t / T_2}}{2}$$
  $$\text{QBER}_{\text{total}} = \frac{1}{4} \left( 2 - e^{-t / T_1} - e^{-t / T_2} \right)$$
  $$\bar{F} = \frac{2 + e^{-t / T_1} + e^{-t / T_2}}{4}$$

---

### 3.5 Noise Compounding Law

When multiple independent error processes (e.g., environmental channel noise $Q_{\text{env}}$ and eavesdropping disturbance $Q_{\text{Eve}}$) act on the carrier, errors compound according to the binary symmetric convolution:
$$Q_{\text{comp}} = Q_1 + Q_2 - 2 Q_1 Q_2$$

---

## 4. System Architecture & Modular Design

```mermaid
flowchart TD
    subgraph Alice_Domain [1. Alice: Transmitter]
        A_Bits[Bits b_A in {0, 1}^N] --> A_Prep[BB84 State Preparation]
        A_Bases[Bases x_A in {Z, X}^N] --> A_Prep
        A_Prep --> CarrierPulses["Carrier Circuits (Qubit 0)"]
    end

    subgraph Quantum_Link [2. Quantum Channel Link]
        CarrierPulses --> Segment1[Link Segment 1: Alice to Eve]
        Segment1 --> Eve_Station{Eve Present?}
        
        Eve_Station -->|No: Direct Transit| TransitNoise["Carrier Transit Noise (Qubit 0 id Gate)"]
        
        Eve_Station -->|Yes: Intercept-Resend| PIR_Attack["PIR Attack (Alpha, Beta)<br/>Measures Q0 & Prepares Replacement"]
        PIR_Attack --> TransitNoise
        
        Eve_Station -->|Yes: Unitary Probe| FPB_Attack["Fuchs-Peres-Brandt Probe (Theta_E)<br/>Entangles Carrier Q0 with Ancilla Q1"]
        FPB_Attack --> ProbeNoise["Carrier Transit Noise (Q0 id Gate)<br/>Strict Ancilla Q1 Isolation"]
    end

    subgraph Bob_Domain [3. Bob: Receiver]
        TransitNoise --> Bob_Meas[Bob: Random Basis Measurement]
        ProbeNoise --> Bob_Meas
        Bob_Bases[Bases x_B in {Z, X}^N] --> Bob_Meas
        Bob_Meas --> RawBits[Raw Outcomes b_B]
    end

    subgraph Sifting_Diagnostics [4. Reconciliation & Metrics Engine]
        A_Bases -.-> Sift[Classical Basis Reconciliation]
        Bob_Bases -.-> Sift
        Sift --> SiftedKey[Sifted Key Extraction]
        SiftedKey --> QBER_Calc["QBER Engine: Total, QBER_Z, QBER_X"]
        QBER_Calc --> AbortCheck{"QBER >= 11%?"}
        AbortCheck -->|Yes| Abort[Protocol Aborted]
        AbortCheck -->|No| SecureKey[Secure Key Extracted]
    end
```

### 4.1 Strict Carrier-Ancilla Isolation

A critical design requirement is that channel noise must act strictly on the optical carrier propagating through the channel, and never corrupt Eve's private ancilla:
- In `QuantumChannel`, carrier pulses are single-qubit circuits (qubit 0).
- Channel noise is registered on the carrier transit gate `id(0)`.
- When `FuchsPeresBrandtProbe` attaches an ancilla (qubit 1), `BB84Engine` appends `id(0)` exclusively on carrier qubit 0.
- Qubit 1 never receives an `id` instruction and remains in an uncorrupted quantum state until Eve's delayed measurement.

---

## 5. Experimental Results & Verification

All five experiments were executed using the dedicated harness `experiments/run_noise_experiments.py` with results archived in `results/quantum_noise_results.json`.

### Experiment 1: Depolarizing Noise Sweep

- **Sample Size:** $N = 10,000$ pulses per point ($M \approx 5,000$ sifted bits).
- **Sweep Range:** $p \in [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]$.

| $p$ | Observed QBER | Theoretical QBER ($p/2$) | Z-QBER | X-QBER | Observed Fidelity | Theoretical Fidelity ($1 - p/2$) | Runtime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 100.00% | 100.00% | 2.76 s |
| **0.01** | 0.5455% | 0.5000% | 0.5136% | 0.5788% | 99.50% | 99.50% | 2.89 s |
| **0.03** | 1.9737% | 1.5000% | 1.9178% | 2.0317% | 98.50% | 98.50% | 3.40 s |
| **0.05** | 2.1796% | 2.5000% | 2.0243% | 2.3340% | 97.50% | 97.50% | 2.96 s |
| **0.10** | 5.1267% | 5.0000% | 4.9469% | 5.3006% | 95.00% | 95.00% | 3.37 s |
| **0.20** | 9.8137% | 10.0000% | 10.0731% | 9.5614% | 90.00% | 90.00% | 3.23 s |

**Findings:**
- Depolarizing noise produces completely symmetric error rates across $Z$ and $X$ conjugate bases ($\text{QBER}_Z \approx \text{QBER}_X \approx p/2$).
- Average state fidelity follows $F = 1 - p/2$ with sub-$10^{-14}$ precision.
- At $p = 0.20$, QBER reaches $9.81\%$, approaching the $11\%$ abort threshold.

---

### Experiment 2: Amplitude Damping Sweep & Bit Asymmetry Analysis

- **Sample Size:** $N = 10,000$ pulses per point.
- **Sweep Range:** $\gamma \in [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]$.

| $\gamma$ | Observed Total QBER | Theoretical Total | Observed Z-QBER | Theoretical Z ($0.5\gamma$) | Observed X-QBER | Theoretical X | Asymmetry $\Delta Q$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| **0.01** | 0.4979% | 0.3753% | 0.5792% | 0.5000% | 0.4224% | 0.2506% | +0.1568% |
| **0.03** | 1.2070% | 1.1279% | 1.5458% | 1.5000% | 0.8692% | 0.7557% | +0.6766% |
| **0.05** | 1.6634% | 1.8830% | 2.1781% | 2.5000% | 1.1295% | 1.2660% | +1.0487% |
| **0.10** | 4.0461% | 3.7829% | 5.8029% | 5.0000% | 2.2250% | 2.5658% | +3.5779% |
| **0.20** | 7.0969% | 7.6393% | 9.0909% | 10.0000% | 5.1220% | 5.2786% | +3.9689% |

#### Input Bit Distribution Dependence ($\gamma = 0.10, N = 8,000$)
To verify that amplitude damping errors scale linearly with the fraction of excited state $|1\rangle$ transmissions ($f_1$):

| Fraction $f_1 = \Pr(\|1\rangle)$ | Empirical $f_1$ | Observed $\text{QBER}_Z$ | Theoretical $\text{QBER}_Z = f_1 \gamma$ | Absolute Discrepancy |
| :---: | :---: | :---: | :---: | :---: |
| **0.25** | 0.2441 | 1.9240% | 2.4413% | 0.517% |
| **0.50** | 0.4921 | 4.5881% | 4.9213% | 0.333% |
| **0.75** | 0.7519 | 6.8081% | 7.5188% | 0.711% |

**Findings:**
- Amplitude damping produces strong conjugate basis asymmetry: $\text{QBER}_Z$ consistently exceeds $\text{QBER}_X$ because relaxation $|1\rangle \to |0\rangle$ directly corrupts the computational basis.
- The input bit distribution test experimentally confirms the linear dependence $\text{QBER}_Z(f_1) = f_1 \cdot \gamma$.

---

### Experiment 3: Phase Damping Sweep & Coherence Loss

- **Sample Size:** $N = 10,000$ pulses per point.
- **Sweep Range:** $\lambda \in [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]$.

| $\lambda$ | Observed Total QBER | Theoretical Total | Observed Z-QBER | Theoretical Z-QBER | Observed X-QBER | Theoretical X-QBER | Coherence $C_{l_1}$ | State Fidelity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | 0.0000% | 0.0000% | **0.0000%** | **0.0000%** | 0.0000% | 0.0000% | 1.0000 | 100.00% |
| **0.01** | 0.1983% | 0.1253% | **0.0000%** | **0.0000%** | 0.3949% | 0.2506% | 0.9950 | 99.87% |
| **0.03** | 0.3168% | 0.3779% | **0.0000%** | **0.0000%** | 0.6223% | 0.7557% | 0.9849 | 99.62% |
| **0.05** | 0.6517% | 0.6330% | **0.0000%** | **0.0000%** | 1.2821% | 1.2660% | 0.9747 | 99.37% |
| **0.10** | 1.5562% | 1.2829% | **0.0000%** | **0.0000%** | 3.0645% | 2.5658% | 0.9487 | 98.72% |
| **0.20** | 2.9649% | 2.6393% | **0.0000%** | **0.0000%** | 5.9289% | 5.2786% | 0.8944 | 97.36% |

**Findings:**
- Computational basis immunity is strictly verified: **$\text{QBER}_Z = 0.0000\%$ across all $\lambda$ values** (0 errors observed out of $> 2,500$ sifted Z pulses at each point).
- All errors are strictly confined to the conjugate diagonal $X$-basis.
- Off-diagonal coherence $C_{l_1}$ decays exactly as $\sqrt{1 - \lambda}$.

---

### Experiment 4: Physical Thermal Relaxation across Regimes

- **Fixed Parameter:** $T_1 = 50\ \mu\text{s}$.
- **Transit Duration Sweep:** $t \in [0.5, 1.0, 2.0, 5.0, 10.0]\ \mu\text{s}$.
- **Regimes:**
  - **Regime A:** $T_2 = 100\ \mu\text{s} = 2 T_1$ (Pure relaxation limit).
  - **Regime B:** $T_2 = 50\ \mu\text{s} = T_1$ (Balanced decay).
  - **Regime C:** $T_2 = 25\ \mu\text{s} = 0.5 T_1$ (Dephasing dominated).

| Regime | Transit Time $t$ | Observed Total QBER | Theoretical Total | Observed Z-QBER | Observed X-QBER | State Fidelity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regime A** ($T_2 = 2T_1$) | 0.5 $\mu$s | 0.4042% | 0.3735% | 0.6586% | 0.1500% | 99.63% |
| | 1.0 $\mu$s | 0.7538% | 0.7425% | 1.0401% | 0.4854% | 99.26% |
| | 2.0 $\mu$s | 1.2281% | 1.4658% | 1.4542% | 0.9935% | 98.52% |
| | 5.0 $\mu$s | 3.1764% | 3.5593% | 3.9265% | 2.4700% | 96.40% |
| | 10.0 $\mu$s | 7.2173% | 6.8406% | 8.8288% | 5.5186% | 93.09% |
| **Regime B** ($T_2 = T_1$) | 0.5 $\mu$s | 0.5312% | 0.4975% | 0.4578% | 0.6027% | 99.50% |
| | 1.0 $\mu$s | 1.0772% | 0.9901% | 1.0060% | 1.1579% | 99.01% |
| | 2.0 $\mu$s | 2.2687% | 1.9605% | 2.2534% | 2.2831% | 98.04% |
| | 5.0 $\mu$s | 3.8825% | 4.7581% | 4.1841% | 3.5901% | 95.24% |
| | 10.0 $\mu$s | 9.1517% | 9.0635% | 9.8152% | 8.4712% | 90.94% |
| **Regime C** ($T_2 = 0.5T_1$) | 0.5 $\mu$s | 0.7483% | 0.7425% | 0.5979% | 0.9050% | 99.26% |
| | 1.0 $\mu$s | 1.2922% | 1.4753% | 0.9766% | 1.6194% | 98.52% |
| | 2.0 $\mu$s | 3.1939% | 2.9024% | 1.9895% | 4.3243% | 97.10% |
| | 5.0 $\mu$s | 6.6304% | 6.9108% | 4.0163% | 9.1084% | 93.09% |
| | 10.0 $\mu$s | **11.4679%** | **12.7737%** | 8.3290% | **14.5153%** | 87.23% |

**Findings:**
- In Regime A ($T_2 > T_1$), longitudinal decay dominates: $\text{QBER}_Z > \text{QBER}_X$.
- In Regime B ($T_2 = T_1$), errors are symmetric across bases.
- In Regime C ($T_2 \ll T_1$), severe transverse dephasing causes $\text{QBER}_X$ to surge to $14.52\%$, causing Total QBER ($11.47\%$) to **cross the $11.0\%$ abort threshold** at $t = 10\ \mu\text{s}$, even in the complete absence of an eavesdropper!

---

### Experiment 5: Combined Noise and Eavesdropping Compounding

- **Sample Size:** $N = 12,000$ pulses per condition.
- **Parameters:** Depolarizing noise $p = 0.04$ ($\text{QBER}_{\text{env}} = 2.0\%$), Intercept-Resend $\alpha = 0.40$ ($\text{QBER}_{\text{Eve}} = 10.0\%$), Unitary probe $\theta_E = \pi/10 \approx 18^\circ$ ($\text{QBER}_{\text{Eve}} = \sin^2(18^\circ) \approx 9.55\%$).

| Condition | Configuration | Observed QBER | Theoretical QBER | Eve Info $I_{AE}$ | Protocol Status | Threshold Margin |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | Ideal Channel | 0.0000% | 0.0000% | 0.0000 bits | **SECURE KEY** | Safe ($-11.00\%$) |
| **B** | Noise Only ($p = 0.04$) | 1.7304% | 2.0000% | 0.0000 bits | **SECURE KEY** | Safe ($-9.27\%$) |
| **C1** | PIR Attack Only ($\alpha = 0.40$) | 10.6847% | 10.0000% | 0.0290 bits | **SECURE KEY** | Safe ($-0.32\%$) |
| **C2** | Unitary Probe Only ($\theta = \pi/10$) | 9.1167% | 9.5492% | 0.2660 bits | **SECURE KEY** | Safe ($-1.88\%$) |
| **D1** | **Noise + PIR Attack** | **11.3903%** | **11.6000%** | 0.0290 bits | **ABORT PROTOCOL** | **Breached (+0.39%)** |
| **D2** | **Noise + Unitary Probe** | **11.1848%** | **11.1672%** | 0.2660 bits | **ABORT PROTOCOL** | **Breached (+0.18%)** |

#### Theoretical Compounding Verification
Using the compounding relation $Q_{\text{total}} = Q_{\text{env}} + Q_{\text{Eve}} - 2 Q_{\text{env}} Q_{\text{Eve}}$:
1. **Noise + Intercept-Resend:**
   $$Q_{\text{theo}} = 0.02 + 0.10 - 2(0.02)(0.10) = 0.1160 \quad (11.60\%)$$
   Empirical observation: **$11.39\%$** ($\Delta = 0.21\%$).
2. **Noise + Unitary Probe:**
   $$Q_{\text{theo}} = 0.02 + 0.09549 - 2(0.02)(0.09549) = 0.11167 \quad (11.17\%)$$
   Empirical observation: **$11.18\%$** ($\Delta = 0.01\%$).

**Crucial Scientific Discovery for Milestone 3 & 4 (cQGAN Optimization):**
In Conditions C1 and C2, Eve operates *stealthily* beneath the $11\%$ abort boundary ($\text{QBER} < 11\%$). However, when deployed onto a realistic channel with even mild background noise ($p = 0.04$), the compounded QBER breaches the $11\%$ threshold, causing Alice and Bob to abort the exchange.
This proves that an adaptive eavesdropper *must explicitly condition its generative attack parameters on the observed channel noise rate $\gamma_{\text{noise}}$* to avoid detection—the exact foundational premise of this research project!

---

## 6. Embedded Visualizations

All five high-resolution research figures were generated using non-interactive Matplotlib and are stored in the project's `results/` directory:

1. **Depolarizing Noise Sweep:**  
   ![Experiment 1: Depolarizing Noise Sweep](file:///c:/Users/lenovo/bb84-qgan-research/results/exp1_depolarizing_sweep.png)  
   *Figure 1: (Left) Total, Z-basis, and X-basis QBER vs. Depolarizing parameter $p$ against the $11\%$ abort boundary. (Right) Average carrier state fidelity degradation following $F = 1 - p/2$.*

2. **Amplitude Damping & Bit Asymmetry:**  
   ![Experiment 2: Amplitude Damping Sweep](file:///c:/Users/lenovo/bb84-qgan-research/results/exp2_amplitude_damping_sweep.png)  
   *Figure 2: (Left) Asymmetric QBER response between $Z$ and $X$ conjugate bases vs. $\gamma$. (Right) Linear dependence of Z-basis QBER on the transmitted excited state fraction $f_1$.*

3. **Phase Damping & Coherence Loss:**  
   ![Experiment 3: Phase Damping Sweep](file:///c:/Users/lenovo/bb84-qgan-research/results/exp3_phase_damping_sweep.png)  
   *Figure 3: (Left) Demonstration of complete computational basis immunity ($\text{QBER}_Z \equiv 0$) with all errors manifesting in the $X$-basis. (Right) Normalized $l_1$-norm coherence decay.*

4. **Thermal Relaxation Physical Regimes:**  
   ![Experiment 4: Thermal Relaxation](file:///c:/Users/lenovo/bb84-qgan-research/results/exp4_thermal_relaxation.png)  
   *Figure 4: (Left) Total QBER progression across transit times $t$ for Regimes A ($T_2 = 2T_1$), B ($T_2 = T_1$), and C ($T_2 = 0.5T_1$). (Right) State fidelity decay.*

5. **Noise vs. Eavesdropping Compounding:**  
   ![Experiment 5: Noise vs Eavesdropping](file:///c:/Users/lenovo/bb84-qgan-research/results/exp5_noise_vs_eavesdropping.png)  
   *Figure 5: (Left) Compounded QBER across experimental conditions demonstrating threshold crossing when stealthy attacks combine with ambient noise. (Right) Intercepted mutual information $I_{AE}$.*

---

## 7. Complete Test Suite & Regression Verification

The complete regression and validation test suite was executed:
```powershell
python -m unittest discover -s tests -v
```

### Full Test Execution Summary
- **Total Test Cases:** 97
- **Passed:** 97
- **Failed / Errored:** 0
- **Execution Time:** 48.21 s

### Breakdown by Test Module
1. `tests/test_alice.py`: **14 tests passed**
   - Single-qubit state preparations ($|0\rangle, |1\rangle, |+\rangle, |-\rangle$).
   - Mutually unbiased basis orthogonality.
   - Batch pulse generation and RNG seeding.
2. `tests/test_bob.py`: **18 tests passed**
   - Independent random basis selection.
   - Projective measurements across $Z$ and $X$ bases.
   - Public basis reconciliation (sifting efficiency $\approx 50\%$).
   - Classical QBER calculation routines.
3. `tests/test_bb84_engine.py`: **9 tests passed**
   - End-to-end protocol simulation without eavesdropper ($\text{QBER} = 0.0\%$).
   - State isolation guarantees.
   - Sifted key consistency.
4. `tests/test_attacks.py`: **18 tests passed**
   - Intercept-Resend attack ($\alpha, \beta$ sweeps, bit-replacement fidelity).
   - Fuchs-Peres-Brandt entangling unitary probe ($U^\dagger U = I_4$, joint normalization, partial trace positivity, audited disturbance $D(\theta_E) = \sin^2\theta_E$).
5. `tests/test_noise_models.py`: **22 tests passed**
   - Parameter range validation & physical boundary rejection ($T_2 \le 2T_1$).
   - Kraus completeness ($\sum K_i^\dagger K_i = I_2$).
   - Density matrix trace preservation and positive semi-definiteness.
   - Physical limiting cases ($p=0, p=1, \gamma=1, \lambda=1$).
   - Analytical QBER predictor validation against density matrix expectations.
   - State fidelity and $l_1$-coherence decay validation.
   - Qiskit Aer `NoiseModel` construction.
6. `tests/test_noisy_channel.py`: **16 tests passed**
   - Transit noise injection on carrier qubit 0.
   - Symmetric depolarizing error execution ($\text{QBER} \approx p/2$).
   - Amplitude damping basis asymmetry.
   - Phase damping computational basis immunity ($\text{QBER}_Z = 0$).
   - Thermal relaxation physical decay across regimes.
   - Simulation reproducibility with fixed random seeds.
   - Compounded noise and eavesdropping conditions (Conditions A–F).

---

## 8. Hardware Resources & Execution Benchmarks

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM).
- **Execution Strategy:** Qiskit Aer single-qubit density matrix and statevector simulation.
- **Circuit Batching:** Circuits are evaluated in batches of 5,000–12,000 pulses with 1 shot per circuit.
- **Resource Footprint:**
  - Peak RAM Allocation: $< 420$ MB.
  - VRAM Consumption: Negligible ($< 150$ MB utilized, avoiding memory thrashing).
  - Overall Milestone 2 Test Suite Runtime: **48.2 s**.
  - All 5 Experimental Sweeps Runtime: **94.6 s**.

---

## 9. Scientific Limitations & Boundaries

1. **Single-Photon Assumption:** Pulses are treated as ideal single qubits. Weak coherent pulses with Poissonian photon number distributions (multiphoton pulses subject to Photon Number Splitting attacks) are not modeled.
2. **Channel Memory:** Noise channels are memoryless (Markovian). Non-Markovian temporal correlations are out of scope.
3. **Eve Ancilla Lifetime:** Eve's probe ancilla is assumed to have an infinite coherence time between carrier transit and classical basis announcement, isolating channel noise to the optical link.

---

## 10. Milestone Status & Transition Roadmap

| Milestone | Deliverables | Status |
| :--- | :--- | :--- |
| **Milestone 0** | Mathematical derivations, system architecture, engineering plan | **Completed** |
| **Milestone 1A** | Alice and Bob core protocol modules (`alice.py`, `bob.py`) | **Completed** |
| **Milestone 1B** | BB84 Protocol Runner & Physical Attacks (`attacks.py`, `bb84_engine.py`) | **Completed** |
| **Milestone 2** | NISQ Noise Modeling & Channel Integration (`noise_models.py`, `quantum_channel.py`) | **COMPLETED & AUDITED** |
| **Milestone 3** | Classical cGAN Baseline (PyTorch MLP, Condition Vector $c = (Q_{\text{budget}}, \gamma_{\text{noise}})$) | **Scheduled Next** |
| **Milestone 4** | Genuine Quantum cQGAN (Qiskit PQC Generator + Hybrid Parameter-Shift Optimization) | Scheduled |
| **Milestone 5** | Experimental Benchmarking, Pareto Frontier Mapping, Final Defense Report | Scheduled |

With Milestone 2 successfully validated, the NISQ quantum channel simulation framework is fully operational and ready to serve as the noisy training and evaluation medium for the classical cGAN (Milestone 3) and quantum cQGAN (Milestone 4) adaptive eavesdropping agents.
