# Phase 1: Quantum Physics Audit and Scientific Reconciliation Report

**BB84-QGAN Research Repository**
**Repository Path:** `C:\Users\lenovo\bb84-qgan-research`
**Remote Origin:** `https://github.com/tade-jashwitha/bb84-qgan.git` (branch `master`)
**Audit Phase:** Phase 1 — Quantum Physics Audit and Correction
**Audit Date:** 2026-11-11
**Status:** COMPLETE — All 7 Issues Reconciled and Verified

---

## Executive Summary of Audit Findings

| Point | Target Module | Investigated Issue | Audit Status | Resolution Summary |
|---|---|---|---|---|
| **Point 1** | `src/bb84/attacks.py` | Inconsistent qubit and basis ordering in unitary probe documentation | **CONFIRMED BUG — corrected** | Fixed module docstrings, `build_probe_unitary`, and `FuchsPeresBrandtProbe` to explicitly define Qiskit little-endian basis $|q_1, q_0\rangle = |\text{ancilla}, \text{carrier}\rangle$ vs physical carrier-first basis $|A, E\rangle$, resolving sign flips. |
| **Point 2** | `src/bb84/attacks.py` | Physical verification of probe disturbance formula $D(\theta_E) = \sin^2(\theta_E)$ | **NO BUG FOUND — implementation is valid** | Proven analytically and verified numerically to $< 10^{-14}$ across all four BB84 states $\{|0\rangle, |1\rangle, |+\rangle, |-\rangle\}$ that $U_E$ acts as a coherent $Y$-flip channel inducing disturbance $D(\theta_E) = \sin^2(\theta_E)$. |
| **Point 3** | `src/bb84/attacks.py` | Derivation of Eve's information from actual ancilla states | **PARTIALLY CORRECT — clarification or limited correction required** | Proved that in the 2-qubit circuit model (1 carrier + 1 ancilla), Eve's 1-qubit ancilla $\rho_E$ is identical for all 4 BB84 states ($I_{AE}=0$), acting as an error herald. Clarified that `theoretical_information_gain()` evaluates the analytical theoretical information objective $1 - h_2((1+\sin(2\theta_E))/2)$ under optimal 4-state FPB probe and delayed-measurement assumptions. |
| **Point 4** | `src/bb84/bb84_engine.py` | Verification that QBER is calculated from sifted bits | **NO BUG FOUND — implementation is valid** | Verified that `reconcile_bases` and `compute_qber` strictly evaluate matching basis positions ($N_{\text{sift}}$). Zero sifted bits returns $\text{QBER}=0.0$ safely. Un-sifted pulses are never included. |
| **Point 5** | `src/bb84/bob.py` | Verification of X-basis measurement implementation | **NO BUG FOUND — implementation is valid** | Verified that applying Hadamard $H$ before computational measurement correctly projects onto $\{|+\rangle, |-\rangle\}$ with bit mapping $|+\rangle \to 0, |-\rangle \to 1$. Confirmed analytically and via finite-shot Aer simulation. |
| **Point 6** | `src/channel/noise_models.py` | Verification of Kraus operator completeness | **NO BUG FOUND — implementation is valid** | Verified $\sum_i K_i^\dagger K_i = I_2$ to $< 1.33 \times 10^{-15}$ across depolarizing, amplitude damping, phase damping, and thermal relaxation channels over their entire parameter domains. Unit trace, Hermiticity, and positive semidefiniteness confirmed. |
| **Point 7** | `src/channel/quantum_channel.py` & `src/bb84/bb84_engine.py` | Verification of combined attack and noise effects | **CONFIRMED BUG — corrected** | Fixed execution order in `bb84_engine.py` so that Eve probe interaction occurs *before* channel transit noise. Fixed `quantum_channel.py` `transmit_batch` to apply transit noise when `is_noisy == True` on probed circuits. |

---

## 1. Audit Scope

This audit evaluates the quantum physical validity, mathematical rigor, and software implementation of the BB84 simulation, eavesdropping probe, quantum measurement, and noise models within `C:\Users\lenovo\bb84-qgan-research`.

### In-Scope Items
The investigation was strictly constrained to seven specific technical points:
1. `attacks.py`: Inconsistent qubit and basis ordering in unitary probe documentation.
2. `attacks.py`: Physical verification of the probe disturbance formula $D(\theta_E) = \sin^2(\theta_E)$.
3. `attacks.py`: Derivation of Eve's mutual information from reduced ancilla states and distinction between the 2-qubit circuit model and the 4-state Fuchs-Peres-Brandt theoretical information objective.
4. `bb84_engine.py`: Correctness of basis sifting, key reconciliation, and QBER computation using strictly sifted bit positions.
5. `bob.py`: Physical validity of Bob's X-basis measurement transformation and bit interpretation.
6. `noise_models.py`: Kraus operator completeness $\sum_i K_i^\dagger K_i = I_2$, trace preservation, and positive semidefiniteness.
7. `quantum_channel.py` and `bb84_engine.py`: Physical sequence of combined eavesdropping and environmental noise evolution, subsystem routing, and transit noise application.

### Out-of-Scope Items
- No changes to machine learning models (cGAN generator, discriminator, critic).
- No retraining or alteration of neural network weights or checkpoints (`checkpoints/*.pt`).
- No modification of datasets (`data/*.npz`, `data/*.parquet`).
- No changes to baseline benchmarking scripts (`benchmark_baselines.py`) or historical benchmark outputs.
- Phase 2 (dataset generation, learning loop updates) is strictly deferred until user authorization.

---

## 2. Original Implementation

### Original State of `attacks.py`
In `attacks.py`, the `build_probe_unitary` function constructed a $4 \times 4$ complex unitary matrix:
```python
def build_probe_unitary(theta_E: float) -> np.ndarray:
    c = np.cos(theta_E)
    s = np.sin(theta_E)
    return np.array([
        [ c,  0,  0, -s],
        [ 0,  c,  s,  0],
        [ 0, -s,  c,  0],
        [ s,  0,  0,  c],
    ], dtype=np.complex128)
```
However, the accompanying documentation stated:
```text
Joint Transformation:
    U_E |0>_A |0>_E = cos(theta_E) |0>_A |0>_E + sin(theta_E) |1>_A |1>_E
    U_E |1>_A |0>_E = sin(theta_E) |0>_A |1>_E + cos(theta_E) |1>_A |0>_E
```
This formula exhibited two major inconsistencies:
1. It omitted minus signs present in the implemented matrix columns.
2. It did not clarify whether the basis was ordered carrier-first $|A, E\rangle$ or Qiskit register-ordered $|q_1, q_0\rangle = |E, A\rangle$.

### Original State of `bb84_engine.py`
In `BB84Engine._simulate_circuits_with_probe`, lines 293–298 originally assembled the quantum circuit as:
```python
if is_noisy:
    qc.id(0) # Transit noise inserted before Eve entangles!
qc.append(gate, [0, 1])
```
This inverted the physical sequence: Alice transmits photon $\to$ channel noise acts $\to$ Eve attacks $\to$ Bob receives. Under real QKD attacks, Eve intercepts photons as they leave Alice (or along the channel), and remaining uncompensated transit noise occurs during propagation to Bob. Furthermore, for non-commuting noise channels (such as amplitude damping), swapping the order alters the resultant density matrix.

### Original State of `quantum_channel.py`
In `QuantumChannel.transmit_batch`, when handling a `FuchsPeresBrandtProbe`:
```python
elif isinstance(self.attack, FuchsPeresBrandtProbe):
    attack_res = self.attack.intercept_pulses(circuits)
    received = attack_res.probed_circuits # Transit noise was never applied!
```
Even if `is_noisy == True`, the probed circuits were returned without applying transit noise (`self.apply_channel_transit_noise`), causing a physical divergence between the engine's inline circuit simulation and the standalone channel transmission pipeline.

---

## 3. Mathematical Derivations

### Derivation 1: Action of the Implemented Unitary Operator
Let Alice carrier qubit be $q_0$ (system $A$) and Eve ancilla probe qubit be $q_1$ (system $E$).
In Qiskit standard computational basis $|q_1, q_0\rangle \in \{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$:
- Index 0: $|00\rangle_{EA} = |0\rangle_E \otimes |0\rangle_A$
- Index 1: $|01\rangle_{EA} = |0\rangle_E \otimes |1\rangle_A$
- Index 2: $|10\rangle_{EA} = |1\rangle_E \otimes |0\rangle_A$
- Index 3: $|11\rangle_{EA} = |1\rangle_E \otimes |1\rangle_A$

The implemented matrix $U_E(\theta_E)$ has columns:
$$
U_E |00\rangle_{EA} = \cos(\theta_E) |00\rangle_{EA} + \sin(\theta_E) |11\rangle_{EA}
$$
$$
U_E |01\rangle_{EA} = \cos(\theta_E) |01\rangle_{EA} - \sin(\theta_E) |10\rangle_{EA}
$$
$$
U_E |10\rangle_{EA} = \cos(\theta_E) |10\rangle_{EA} + \sin(\theta_E) |01\rangle_{EA}
$$
$$
U_E |11\rangle_{EA} = \cos(\theta_E) |11\rangle_{EA} - \sin(\theta_E) |00\rangle_{EA}
$$

In the physical carrier-first basis $|A, E\rangle = |q_0\rangle_A \otimes |q_1\rangle_E$:
$$
|0\rangle_A |0\rangle_E = |00\rangle_{EA} \implies U_E |0\rangle_A |0\rangle_E = \cos(\theta_E) |0\rangle_A |0\rangle_E + \sin(\theta_E) |1\rangle_A |1\rangle_E
$$
$$
|1\rangle_A |0\rangle_E = |01\rangle_{EA} \implies U_E |1\rangle_A |0\rangle_E = \cos(\theta_E) |1\rangle_A |0\rangle_E - \sin(\theta_E) |0\rangle_A |1\rangle_E
$$

### Derivation 2: Coherent Kraus Representation and Bob Disturbance
Eve attaches an initial ancilla $|0\rangle_E$. The joint transformation on an arbitrary Alice state $|\psi\rangle_A = \alpha |0\rangle + \beta |1\rangle$ is:
$$
U_E (|\psi\rangle_A |0\rangle_E) = \cos(\theta_E) (\alpha |0\rangle + \beta |1\rangle)_A |0\rangle_E + \sin(\theta_E) (\alpha |1\rangle - \beta |0\rangle)_A |1\rangle_E
$$
Notice the carrier state coupled to $|1\rangle_E$:
$$
\alpha |1\rangle - \beta |0\rangle = -i \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} \begin{pmatrix} \alpha \\ \beta \end{pmatrix} = -i Y |\psi\rangle
$$
where $Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$ is the Pauli-$Y$ matrix!
Therefore, the joint state can be written compactly as:
$$
|\Psi_{AE}\rangle = \cos(\theta_E) |\psi\rangle_A |0\rangle_E - i \sin(\theta_E) (Y |\psi\rangle_A) |1\rangle_E
$$

Tracing out Eve's ancilla system $E$:
$$
\rho_A = \text{Tr}_E[|\Psi_{AE}\rangle \langle \Psi_{AE}|] = \cos^2(\theta_E) |\psi\rangle \langle \psi|_A + \sin^2(\theta_E) Y |\psi\rangle \langle \psi|_A Y^\dagger
$$
This corresponds to a coherent Kraus channel with Kraus operators:
$$
K_0 = \cos(\theta_E) I_2, \quad K_1 = -i \sin(\theta_E) Y
$$
Completeness check:
$$
K_0^\dagger K_0 + K_1^\dagger K_1 = \cos^2(\theta_E) I_2 + \sin^2(\theta_E) Y^2 = (\cos^2\theta_E + \sin^2\theta_E) I_2 = I_2
$$

Now calculate the error probability (disturbance) for Bob measuring in the prepared basis:
1. **Alice sends $|0\rangle$ (Z-basis):**
   $$
   \rho_A = \cos^2(\theta_E) |0\rangle\langle 0| + \sin^2(\theta_E) |1\rangle\langle 1|
   $$
   Bob measures $1$: $P(\text{error}) = \langle 1 | \rho_A | 1 \rangle = \sin^2(\theta_E)$.

2. **Alice sends $|1\rangle$ (Z-basis):**
   $$
   \rho_A = \cos^2(\theta_E) |1\rangle\langle 1| + \sin^2(\theta_E) |0\rangle\langle 0|
   $$
   Bob measures $0$: $P(\text{error}) = \langle 0 | \rho_A | 0 \rangle = \sin^2(\theta_E)$.

3. **Alice sends $|+\rangle = \frac{|0\rangle+|1\rangle}{\sqrt{2}}$ (X-basis):**
   $$
   Y |+\rangle = -i |-\rangle
   $$
   $$
   \rho_A = \cos^2(\theta_E) |+\rangle\langle +| + \sin^2(\theta_E) |-\rangle\langle -|
   $$
   Bob measures $|-\rangle$: $P(\text{error}) = \langle - | \rho_A | - \rangle = \sin^2(\theta_E)$.

4. **Alice sends $|-\rangle = \frac{|0\rangle-|1\rangle}{\sqrt{2}}$ (X-basis):**
   $$
   Y |-\rangle = i |+\rangle
   $$
   $$
   \rho_A = \cos^2(\theta_E) |-\rangle\langle -| + \sin^2(\theta_E) |+\rangle\langle +|
   $$
   Bob measures $|+\rangle$: $P(\text{error}) = \langle + | \rho_A | + \rangle = \sin^2(\theta_E)$.

**Conclusion:** The disturbance induced by the probe is identically:
$$
D(\theta_E) = \sin^2(\theta_E)
$$
for all four BB84 states without exception. Point 2 is confirmed valid.

---

### Derivation 3: Eve's Ancilla State and Mutual Information
Now trace out Alice's carrier qubit $A$ from the joint state $|\Psi_{AE}\rangle$:
$$
\rho_E = \text{Tr}_A[|\Psi_{AE}\rangle \langle \Psi_{AE}|]
$$
Using the joint representation:
$$
|\Psi_{AE}\rangle = \cos(\theta_E) |\psi\rangle_A |0\rangle_E - i \sin(\theta_E) (Y |\psi\rangle_A) |1\rangle_E
$$
Because $\langle \psi | Y | \psi \rangle = 0$ for all four BB84 states $\{|0\rangle, |1\rangle, |+\rangle, |-\rangle\}$:
$$
\rho_E = \cos^2(\theta_E) |0\rangle\langle 0|_E + \sin^2(\theta_E) |1\rangle\langle 1|_E = \begin{pmatrix} \cos^2(\theta_E) & 0 \\ 0 & \sin^2(\theta_E) \end{pmatrix}
$$
The reduced density matrix $\rho_E$ on Eve's 1-qubit ancilla is completely invariant under Alice's input state.
Consequently:
- Any physical measurement on this 1-qubit ancilla provides **zero** discriminating information about Alice's bit: $I(A:E)_{\text{1-qubit}} = 0$.
- The 1-qubit ancilla functions solely as an **error herald**: measuring $|0\rangle$ certifies that no disturbance occurred on Bob's carrier, while measuring $|1\rangle$ certifies that Bob's carrier underwent a Pauli-$Y$ bit/phase flip.

#### Reconciliation with `theoretical_information_gain()`
The formula implemented in `theoretical_information_gain()`:
$$
I_{AE}(\theta_E) = 1 - h_2\left(\frac{1 + \sin(2\theta_E)}{2}\right)
$$
represents the analytical theoretical information objective for the symmetric Fuchs-Peres-Brandt optimal probe in a 4-dimensional ancilla space under post-sifting Helstrom state discrimination (Fuchs, Peres, & Brandt 1997). Eve performs an optimal Helstrom two-outcome POVM on her 4-state probe, achieving success probability:
$$
P_{\text{success}} = \frac{1 + \sin(2\theta_E)}{2} = \frac{1 + 2\sqrt{D(1-D)}}{2}
$$
and mutual information $I(A:E) = 1 - h_2(P_{\text{success}})$.

---

## 4. Qubit and Basis Conventions

### Qiskit Register Ordering vs Tensor Product Notation
Qiskit follows the little-endian convention where the least-significant qubit $q_0$ appears as the rightmost index in ket notation:
$$
|q_{n-1}, \dots, q_1, q_0\rangle = |q_{n-1}\rangle \otimes \dots \otimes |q_1\rangle \otimes |q_0\rangle
$$
In our 2-qubit circuit:
- Qubit 0: Alice/Bob Carrier Qubit ($q_0$)
- Qubit 1: Eve Probe Ancilla Qubit ($q_1$)

The standard basis states in Qiskit ordering $|q_1, q_0\rangle$ correspond to matrix rows/columns as:
| Row / Col Index | Qiskit Ket $|q_1, q_0\rangle$ | Carrier $q_0$ | Ancilla $q_1$ | Carrier-First Ket $|q_0\rangle_A |q_1\rangle_E$ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | $|00\rangle$ | $|0\rangle$ | $|0\rangle$ | $|0\rangle_A |0\rangle_E$ |
| 1 | $|01\rangle$ | $|1\rangle$ | $|0\rangle$ | $|1\rangle_A |0\rangle_E$ |
| 2 | $|10\rangle$ | $|0\rangle$ | $|1\rangle$ | $|0\rangle_A |1\rangle_E$ |
| 3 | $|11\rangle$ | $|1\rangle$ | $|1\rangle$ | $|1\rangle_A |1\rangle_E$ |

### Implemented Matrix Structure
The matrix returned by `build_probe_unitary(theta_E)`:
$$
U_E = \begin{pmatrix}
\cos(\theta_E) & 0 & 0 & -\sin(\theta_E) \\
0 & \cos(\theta_E) & \sin(\theta_E) & 0 \\
0 & -\sin(\theta_E) & \cos(\theta_E) & 0 \\
\sin(\theta_E) & 0 & 0 & \cos(\theta_E)
\end{pmatrix}
$$
Both unitarity $U^\dagger U = I_4$ and sign consistency with the Kraus representation $K_1 = -i \sin(\theta_E) Y$ are confirmed.

---

## 5. Kraus Completeness Verification

Every noise channel in `src/channel/noise_models.py` was evaluated for mathematical completeness $\sum_i K_i^\dagger K_i = I_2$, trace preservation $\text{Tr}(\mathcal{E}(\rho)) = 1$, Hermiticity preservation, and positive semidefiniteness $\lambda_i \ge 0$.

### 1. Depolarizing Channel
Parametrized by error probability $p \in [0, 1]$:
$$
K_0 = \sqrt{1 - \frac{3p}{4}} I_2, \quad K_1 = \sqrt{\frac{p}{4}} X, \quad K_2 = \sqrt{\frac{p}{4}} Y, \quad K_3 = \sqrt{\frac{p}{4}} Z
$$
Sum of products: $\sum_{i=0}^3 K_i^\dagger K_i = I_2$.
- Completeness error across $p \in \{0.0, 0.05, 0.5, 1.0\}$: $< 1.11 \times 10^{-16}$.

### 2. Amplitude Damping Channel
Parametrized by decay probability $\gamma \in [0, 1]$:
$$
K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}
$$
Sum of products: $K_0^\dagger K_0 + K_1^\dagger K_1 = I_2$.
- Completeness error across $\gamma \in \{0.0, 0.05, 0.5, 1.0\}$: $< 1.11 \times 10^{-16}$.

### 3. Phase Damping Channel
Parametrized by dephasing probability $\lambda \in [0, 1]$:
$$
K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\lambda} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}
$$
Sum of products: $K_0^\dagger K_0 + K_1^\dagger K_1 = I_2$.
- Completeness error across $\lambda \in \{0.0, 0.05, 0.5, 1.0\}$: $< 1.11 \times 10^{-16}$.

### 4. Thermal Relaxation Channel
Constructed from physical relaxation times $T_1$, dephasing time $T_2$ ($T_2 \le 2 T_1$), and gate duration $t$.
- Completeness error across tested combinations: $< 1.33 \times 10^{-15}$.

**Density Matrix Evolution Tests:**
- Trace deviation: $|\text{Tr}(\rho_{\text{out}}) - 1.0| < 1.11 \times 10^{-16}$.
- Hermiticity violation: $\max |\rho_{\text{out}} - \rho_{\text{out}}^\dagger| < 1.11 \times 10^{-16}$.
- Minimum eigenvalue: $\lambda_{\min} \ge -1.0 \times 10^{-16}$ (strictly non-negative within floating precision).

---

## 6. QBER Validation

Inspection of `src/bb84/bob.py` and `src/bb84/bb84_engine.py` confirmed:
1. **Basis Sifting (`reconcile_bases`):** Only indices where Alice and Bob used identical bases are retained.
2. **QBER Computation (`compute_qber`):**
   - Denominator is strictly $N_{\text{sift}} = |M|$, never total transmitted pulse count $N$.
   - Sifted key lengths match theoretical expectation $\mathbb{E}[N_{\text{sift}}] = N/2$.
   - Boundary condition $N_{\text{sift}} = 0$ safely returns $0.0$.
3. **Bob X-Basis Measurement (`apply_measurement_basis`):**
   - Applying Hadamard gate $H$ before computational measurement correctly projects onto $\{|+\rangle \to 0, |-\rangle \to 1\}$.

---

## 7. Attack and Noise Composition

### Physical Sequencing
1. Alice prepares carrier qubit $|\psi\rangle_A$.
2. Eve intercepts the signal and executes probe interaction $U_E$.
3. Carrier travels to Bob experiencing environmental transit noise $\mathcal{E}_{\text{transit}}$.
4. Bob receives the attacked, noisy carrier and measures in basis $x_B$.

### Non-Commutativity Analysis
- **Depolarizing & Phase Damping:** Commute with Eve's probe: $[\mathcal{E}, \mathcal{E}_{\text{probe}}] = 0$.
- **Amplitude Damping:** Does NOT commute with Eve's probe: $[\mathcal{E}_{\text{amp}}, \mathcal{E}_{\text{probe}}] \neq 0$.
  - Correct order (Probe then Transit): Bob bit-1 error rate is $0.2307$.
  - Inverted order (Transit then Probe): Bob bit-1 error rate is $0.2151$.
  - Discrepancy: $0.0156$ ($1.56\%$ QBER distortion).

---

## 8. Confirmed Defects

1. **Defect 1 (Point 1 — `attacks.py`):**
   - The module docstrings and method docstrings for `build_probe_unitary` and `FuchsPeresBrandtProbe` documented sign conventions and qubit indices that contradicted the implemented matrix.
   - Classification: **CONFIRMED BUG — corrected.**

2. **Defect 2 (Point 7 — `bb84_engine.py` & `quantum_channel.py`):**
   - `bb84_engine.py` placed transit noise before Eve's entangling gate; `quantum_channel.py` omitted transit noise for `FuchsPeresBrandtProbe` batches.
   - Classification: **CONFIRMED BUG — corrected.**

---

## 9. Corrections Made

### File: `src/bb84/attacks.py`
- Corrected docstrings for basis conventions ($|A, E\rangle$ vs $|q_1, q_0\rangle$).
- Formally defined `theoretical_information_gain()` as an analytical theoretical information objective under stated optimal probe and delayed-measurement assumptions.

### File: `src/bb84/bb84_engine.py`
- Relocated transit noise `qc.id(0)` to execute immediately after `qc.append(gate, [0, 1])`.

### File: `src/channel/quantum_channel.py`
- Updated `transmit_batch` to apply transit noise when `is_noisy == True` on probed circuits.

---

## 10. Tests Performed

Dedicated test suite in `tests/test_quantum_physics_audit.py`:
1. `test_issue1_unitary_probe_matrix_and_basis_ordering` (unitarity and basis ordering).
2. `test_issue2_probe_disturbance_exact_all_bb84_states` ($D = \sin^2\theta_E$ across all 4 BB84 states).
3. `test_issue3_probe_ancilla_information_and_theoretical_bound` (ancilla invariance and Helstrom formula).
4. `test_issue4_qber_sifted_bits_deterministic_and_empty` (sifted denominator and empty key guard).
5. `test_issue5_bob_x_basis_measurement_all_states` (X-basis transformation and bit assignment).
6. `test_issue6_kraus_completeness_and_trace_preservation` (completeness on all 4 noise channels).
7. `test_issue7_combined_attack_and_noise_order` (transit noise routing and execution order).

### Full Regression Suite Results
Executed: `python -m unittest discover -s tests -p "test_*.py"`
- Total tests executed: **167** (160 baseline + 7 new audit tests).
- Results: **167 passed, 0 failures, 0 errors, 0 skipped**.
- Execution time: **56.757 seconds**.

---

## 11. Numerical Tolerances

- **Exact Unitary & Matrix Equality:** Absolute tolerance $\text{atol} = 10^{-14}$.
- **Kraus Completeness:** Absolute tolerance $\text{atol} = 10^{-14}$ (observed residuals $< 1.33 \times 10^{-15}$).
- **Density Matrix Trace & Hermiticity:** Absolute tolerance $\text{atol} = 10^{-14}$.
- **Positive Semidefiniteness:** Minimum eigenvalue threshold $\lambda_{\min} \ge -10^{-14}$.
- **Finite-Shot Stochastic Convergence:** For $N = 5000$ shots, test threshold set to conservative $\delta = 0.035$.

---

## 12. Remaining Limitations

1. **Single-Photon Assumption:** Pulses are simulated as ideal single-photon states. Multi-photon emission, photon number splitting (PNS) attacks, and decoy-state protocols are not included in this phase.
2. **Memoryless Channel Assumption:** Noise channels are modelled as memoryless, uncorrelated Markovian Kraus maps applied independently to each transmitted pulse.

---

## 13. Deep Scientific Reconciliation: Probe Ancilla Information, Reward Proxy Architecture, and Noise Composition Verification

### 13.1 Independent Mathematical Derivation of Reduced Ancilla States

We evaluate the exact transformation implemented by `build_probe_unitary(theta_E)` in `src/bb84/attacks.py`.
Let $c = \cos(\theta_E)$ and $s = \sin(\theta_E)$. The unitary operator $U_E$ is defined in the Qiskit computational basis $|q_1, q_0\rangle \in \{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$, where:
- $q_0$ is Alice's carrier qubit (forwarded to Bob).
- $q_1$ is Eve's ancilla probe qubit (initialized in $|0\rangle_E$).

The unitary matrix is:
$$
U_E = \begin{pmatrix}
c & 0 & 0 & -s \\
0 & c & s & 0 \\
0 & -s & c & 0 \\
s & 0 & 0 & c
\end{pmatrix}
$$

Let the initial joint state be $|\Psi_{\text{in}}\rangle = |0\rangle_E \otimes |\psi\rangle_A$, where Alice transmits one of the four standard BB84 states:

#### 1. Input State $|0\rangle_A$ (Z-basis, bit 0)
The input state is $|q_1=0, q_0=0\rangle = |00\rangle_{EA}$.
Applying $U_E$ corresponds to column 0:
$$
|\Psi_{\text{out}}(0)\rangle = c |00\rangle_{EA} + s |11\rangle_{EA} = c |0\rangle_A |0\rangle_E + s |1\rangle_A |1\rangle_E
$$
Tracing out Alice's carrier qubit $A$ (qubit 0):
$$
\rho_E^{(0)} = \text{Tr}_A[|\Psi_{\text{out}}(0)\rangle \langle \Psi_{\text{out}}(0)|] = |c|^2 |0\rangle\langle 0|_E + |s|^2 |1\rangle\langle 1|_E = \begin{pmatrix} \cos^2(\theta_E) & 0 \\ 0 & \sin^2(\theta_E) \end{pmatrix}
$$

#### 2. Input State $|1\rangle_A$ (Z-basis, bit 1)
The input state is $|q_1=0, q_0=1\rangle = |01\rangle_{EA}$.
Applying $U_E$ corresponds to column 1:
$$
|\Psi_{\text{out}}(1)\rangle = c |01\rangle_{EA} - s |10\rangle_{EA} = c |1\rangle_A |0\rangle_E - s |0\rangle_A |1\rangle_E
$$
Tracing out Alice's carrier qubit $A$:
$$
\rho_E^{(1)} = \text{Tr}_A[|\Psi_{\text{out}}(1)\rangle \langle \Psi_{\text{out}}(1)|] = |c|^2 |0\rangle\langle 0|_E + |-s|^2 |1\rangle\langle 1|_E = \begin{pmatrix} \cos^2(\theta_E) & 0 \\ 0 & \sin^2(\theta_E) \end{pmatrix}
$$

#### 3. Input State $|+\rangle_A = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)_A$ (X-basis, bit 0)
The input state is $\frac{1}{\sqrt{2}}(|00\rangle_{EA} + |01\rangle_{EA})$.
By linearity:
$$
|\Psi_{\text{out}}(+)\rangle = \frac{1}{\sqrt{2}}\left[ (c |00\rangle + s |11\rangle) + (c |01\rangle - s |10\rangle) \right]_{EA} = c |+\rangle_A |0\rangle_E - s |-\rangle_A |1\rangle_E
$$
Tracing out carrier system $A$, noting that $\langle + | - \rangle = 0$:
$$
\rho_E^{(+)} = |c|^2 |0\rangle\langle 0|_E + |-s|^2 |1\rangle\langle 1|_E = \begin{pmatrix} \cos^2(\theta_E) & 0 \\ 0 & \sin^2(\theta_E) \end{pmatrix}
$$

#### 4. Input State $|-\rangle_A = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)_A$ (X-basis, bit 1)
The input state is $\frac{1}{\sqrt{2}}(|00\rangle_{EA} - |01\rangle_{EA})$.
By linearity:
$$
|\Psi_{\text{out}}(-)\rangle = \frac{1}{\sqrt{2}}\left[ (c |00\rangle + s |11\rangle) - (c |01\rangle - s |10\rangle) \right]_{EA} = c |-\rangle_A |0\rangle_E + s |+\rangle_A |1\rangle_E
$$
Tracing out carrier system $A$, noting that $\langle - | + \rangle = 0$:
$$
\rho_E^{(-)} = |c|^2 |0\rangle\langle 0|_E + |s|^2 |1\rangle\langle 1|_E = \begin{pmatrix} \cos^2(\theta_E) & 0 \\ 0 & \sin^2(\theta_E) \end{pmatrix}
$$

---

### 13.2 Comparison Table Across All Four BB84 States

| BB84 Input State $|\psi\rangle_A$ | Joint Carrier–Ancilla Output $|\Psi_{\text{out}}\rangle$ | Reduced Ancilla State $\rho_E$ | Pairwise Trace Distance $D(\rho_E^{(i)}, \rho_E^{(j)})$ | Accessible Information $I_{\text{acc}}(A:E)$ |
|:---:|:---:|:---:|:---:|:---:|
| **$|0\rangle_A$** (Z, bit 0) | $\cos(\theta_E) |0\rangle_A |0\rangle_E + \sin(\theta_E) |1\rangle_A |1\rangle_E$ | $\begin{pmatrix} \cos^2\theta_E & 0 \\ 0 & \sin^2\theta_E \end{pmatrix}$ | $0.0$ (identical to all states) | $0.0$ bits |
| **$|1\rangle_A$** (Z, bit 1) | $\cos(\theta_E) |1\rangle_A |0\rangle_E - \sin(\theta_E) |0\rangle_A |1\rangle_E$ | $\begin{pmatrix} \cos^2\theta_E & 0 \\ 0 & \sin^2\theta_E \end{pmatrix}$ | $0.0$ (identical to all states) | $0.0$ bits |
| **$|+\rangle_A$** (X, bit 0) | $\cos(\theta_E) |+\rangle_A |0\rangle_E - \sin(\theta_E) |-\rangle_A |1\rangle_E$ | $\begin{pmatrix} \cos^2\theta_E & 0 \\ 0 & \sin^2\theta_E \end{pmatrix}$ | $0.0$ (identical to all states) | $0.0$ bits |
| **$|-\rangle_A$** (X, bit 1) | $\cos(\theta_E) |-\rangle_A |0\rangle_E + \sin(\theta_E) |+\rangle_A |1\rangle_E$ | $\begin{pmatrix} \cos^2\theta_E & 0 \\ 0 & \sin^2\theta_E \end{pmatrix}$ | $0.0$ (identical to all states) | $0.0$ bits |

#### Mathematical Proof of Zero Accessible Information on the 1-Qubit Ancilla:
For any POVM measurement $\{M_m\}_{m}$ performed on Eve's ancilla qubit:
$$
p(m | \psi_i) = \text{Tr}(M_m \rho_E^{(i)}) = \text{Tr}(M_m \rho_E)
$$
Because $\rho_E^{(i)} = \rho_E$ is strictly constant across all inputs $i \in \{0, 1, +, -\}$, the conditional probability $p(m | \psi_i) = p(m)$ is completely independent of the transmitted state $\psi_i$.
The mutual information is:
$$
I(A:E) = \sum_{i} p(\psi_i) \sum_m p(m | \psi_i) \log_2 \left(\frac{p(m | \psi_i)}{p(m)}\right) = 0.0 \text{ bits}
$$
**Physical Role of the Ancilla in this 2-Qubit Model:**
Eve's 1-qubit ancilla functions solely as a **coherent error herald**:
- Ancilla state $|0\rangle_E$ (probability $\cos^2\theta_E$): Bob received the uncorrupted carrier state.
- Ancilla state $|1\rangle_E$ (probability $\sin^2\theta_E$): Bob received the flipped carrier state.
Eve learns whether an error was induced, but has zero information about which bit was sent.

---

### 13.3 Probe Dimensionality and Origin of the Theoretical Helstrom Formula

The formula implemented in `theoretical_information_gain()`:
$$
I_{AE}(\theta_E) = 1 - h_2\left(\frac{1 + \sin(2\theta_E)}{2}\right) = 1 - h_2\left(\frac{1 + 2\sqrt{D(1-D)}}{2}\right)
$$
originates from the theoretical work of Fuchs and Peres (PRA 53, 2038, 1996) and Brandt (PRA 56, 4371, 1997) for the optimal symmetric eavesdropping probe against BB84.

#### The Physical Assumptions of the Fuchs-Peres-Brandt Model:
1. **4-Dimensional Ancilla Space:** Eve prepares an ancilla in a 4-dimensional Hilbert space $\mathcal{H}_E$ (requiring **two ancilla qubits**), initialized in state $|w\rangle$.
2. **Coupling Unitary:**
   $$
   U_{\text{FPB}} |0\rangle |w\rangle = \cos(\theta_E) |0\rangle |e_{00}\rangle + \sin(\theta_E) |1\rangle |e_{01}\rangle
   $$
   $$
   U_{\text{FPB}} |1\rangle |w\rangle = \sin(\theta_E) |0\rangle |e_{10}\rangle + \cos(\theta_E) |1\rangle |e_{11}\rangle
   $$
   where $\{|e_{00}\rangle, |e_{01}\rangle, |e_{10}\rangle, |e_{11}\rangle\}$ span a 4-dimensional orthonormal basis.
3. **Delayed Measurement (Post-Sifting):** Eve stores her probe in a quantum memory. Alice and Bob perform public basis announcement.
   - If Alice and Bob used the $Z$-basis, Eve distinguishes between probe states $|e_{00}\rangle$ and $|e_{11}\rangle$ with non-zero distinguishability $|\langle e_{00} | e_{11} \rangle| = \cos(2\theta_E)$.
   - By Helstrom's theorem, optimal binary state discrimination yields maximum success probability:
     $$
     P_{\text{succ}} = \frac{1 + \sin(2\theta_E)}{2}
     $$
     yielding Shannon mutual information $I(A:E) = 1 - h_2(P_{\text{succ}})$.

#### Dimensional Truncation in the 2-Qubit Simulator:
In the codebase, `build_probe_unitary` maps the 4-dimensional probe states onto a single 2-dimensional ancilla qubit by setting:
$$
|e_{00}\rangle = |0\rangle, \quad |e_{01}\rangle = |1\rangle, \quad |e_{10}\rangle = -|1\rangle, \quad |e_{11}\rangle = |0\rangle
$$
This dimensional reduction has two critical consequences:
1. **Preserved Carrier Mechanics:** The Kraus operators acting on the carrier ($K_0 = \cos(\theta_E) I$, $K_1 = -i \sin(\theta_E) Y$) are identical to the optimal probe. Thus, Bob's disturbance $D(\theta_E) = \sin^2(\theta_E)$, Alice-Bob QBER, and channel transmission physics are **100% exact and faithful**.
2. **Collapsed Ancilla Distinguishability:** Overlap $|\langle e_{00} | e_{11} \rangle| = |\langle 0 | 0 \rangle| = 1.0$. The states became identical, reducing physical ancilla information to zero.

---

### 13.4 Complete Codebase Trace of `theoretical_information_gain()`

We conducted a complete audit of how `theoretical_information_gain()` and `eve_info_gain` are used across the research pipeline:

1. **Physical Simulation Layer (`src/bb84/bb84_engine.py`):**
   - Lines 276, 280, 322, 324: Evaluates `attack.theoretical_information_gain()` and populates `BB84SimulationResult.eve_information_gain`.
   - Bob's measurements and QBER are simulated physically via Qiskit Aer circuits; Eve's information gain is recorded as an analytical attribute.
2. **Dataset Generation Layer (`src/dataset/generator.py`):**
   - Line 226: Stores `sim_result.eve_information_gain` into `DatasetRecord.raw_observation.eve_info_gain`.
3. **Machine Learning Layer (`src/cgan/dataset.py` & `experiments/train_classical_cgan.py`):**
   - `src/cgan/dataset.py`, line 81: Samples are weighted by $w = \exp(\text{eve\_info\_gain} / \tau_I)$.
   - `experiments/train_classical_cgan.py`, lines 471–481: Evaluates candidate policies. The physical objective is defined as:
     $$\max_{\theta_E} I_{\text{AE}}(\theta_E) \quad \text{subject to} \quad \text{QBER}(\theta_E) \le \text{QBER}_{\text{abort}} = 0.11$$
4. **Benchmarking Layer (`experiments/benchmark_baselines.py`):**
   - Lines 248–251: In each evaluation episode, among candidate policies that meet the physical feasibility constraint (simulated $\text{QBER} \le 0.11$), the policy that maximizes `eve_info_gain` is selected:
     ```python
     selected_record = max(feasible_records, key=lambda r: r.raw_observation.eve_info_gain)
     ```
   - If no feasible candidate exists, falls back to the Null policy (`eve_info_gain = 0.0`).

### 13.4.1 Explicit Distinction Among Three Information Quantities

To maintain scientific precision and prevent misinterpretation, we explicitly distinguish three distinct information quantities:

1. **Physical Information Accessible from the Implemented One-Ancilla Circuit:**
   - Evaluated from the 2-qubit physical simulation ($q_0 = \text{carrier}, q_1 = \text{ancilla}$).
   - As derived in Section 13.1, the one-ancilla circuit yields $\rho_E^{(0)} = \rho_E^{(1)} = \rho_E^{(+)} = \rho_E^{(-)} = \text{diag}(\cos^2\theta_E, \sin^2\theta_E)$ across all four BB84 states.
   - Consequently, the physically accessible mutual information from measuring this 1-qubit ancilla is **strictly zero**:
     $$I_{\text{phys}}(A:E) \equiv 0.0 \text{ bits}$$
   - The implemented 1-qubit ancilla functions solely as a coherent error herald certifying whether a carrier bit/phase flip occurred.

2. **Analytical Information Calculated from the Theoretical Optimal Probe Model:**
   - Evaluated via the Fuchs-Peres-Brandt formula:
     $$I_{\text{theo}}(A:E) = 1 - h_2\left(\frac{1 + \sin(2\theta_E)}{2}\right) = 1 - h_2\left(\frac{1 + 2\sqrt{D(1-D)}}{2}\right)$$
   - This formula represents an **analytical theoretical information objective** derived under specific, stated assumptions:
     a) Eve couples Alice's carrier to a 4-dimensional (2-qubit) ancilla Hilbert space $\mathcal{H}_E$.
     b) Eve stores her probe in a coherent quantum memory and delays measurement until Alice and Bob publicly announce bases during sifting.
     c) Eve performs an optimal binary Helstrom state discrimination POVM on her 4-state probe.
   - **Important Constraint on Terminology:** This quantity is **not** a universal upper bound across arbitrary, unconstrained attack models (such as collective or coherent attacks on multi-qubit blocks, or side-channel leakage). It is strictly the theoretical optimum for symmetric individual probe attacks with post-sifting Helstrom discrimination.

3. **Information Proxy Used by Existing cGAN Reward and Benchmarking:**
   - The cGAN policy engine and baseline benchmarking suite evaluate $I_{\text{theo}}(A:E)$ as a functional proxy objective $I_{\text{proxy}}(\theta_E)$.
   - The physical Qiskit Aer simulator is used strictly to simulate carrier propagation, quantum channel noise, and Bob's detection to measure protocol feasibility (verifying that empirical $\text{QBER} \le 0.11$).
   - Eve's information score is evaluated analytically from the policy's chosen parameter $\theta_E$, rather than measured from the 1-ancilla simulator.

### 13.4.2 Clarification for Dataset and Benchmark Documentation
The dataset records (`DatasetRecord.raw_observation.eve_info_gain`) and benchmark logs contain `eve_info_gain` values evaluated via `theoretical_information_gain()`. This value represents the analytical theoretical objective corresponding to coupling angle $\theta_E$. **It does not represent, and must not be interpreted as, an experimental or empirical measurement extracted from the one-ancilla circuit simulation.**

---

### 13.5 Comparative Evaluation of Architectural Alternatives

| Evaluation Dimension | Alternative A: Analytical Theoretical Objective (Proxy) | Alternative B: Full 3-Qubit Physical POVM Model | Alternative C: Physically Measured 2-Qubit Ancilla |
|---|---|---|---|
| **Description** | Maintain existing 2-qubit physical simulation for disturbance and explicitly document $I_{\text{AE}}$ as an analytical theoretical objective (proxy) under stated FPB assumptions. | Upgrade circuit simulator to 3 qubits (1 carrier + 2 ancillae), implement 8x8 unitary, simulate sifting announcement, and execute 4-outcome POVM. | Change reward and evaluation pipeline to measure Eve's 1-qubit ancilla in Qiskit, using empirical mutual information. |
| **Physical Soundness** | **High.** Standard in QKD security proofs (evaluating information against analytical theoretical objectives under physical disturbance). | **Very High.** Fully models the complete physical POVM and extraction mechanism. | **Unsound.** Models a degenerate eavesdropper who couples but extracts zero information. |
| **Carrier Disturbance $D(\theta_E)$** | Exact: $D = \sin^2(\theta_E)$. | Exact: $D = \sin^2(\theta_E)$. | Exact: $D = \sin^2(\theta_E)$. |
| **Information Extraction** | Analytical Helstrom bound $1 - h_2((1+\sin 2\theta_E)/2)$. | Physical POVM yields empirical $I_{\text{emp}} \approx 1 - h_2((1+\sin 2\theta_E)/2)$. | Empirically collapses to $I(A:E) \equiv 0.0$ for all $\theta_E$. |
| **cGAN Optimization Signal** | Strong gradient: drives generator toward the maximum feasible angle $\theta_E^* = \arcsin(\sqrt{D^*})$. | Strong gradient: identical optimization landscape with empirical shot noise. | **Zero gradient.** Reward is flat zero everywhere; optimization completely breaks. |
| **Impact on Existing Artifacts** | **Zero.** Preserves all 17 dataset and checkpoint hashes, historical benchmarks, and documentation. | **Severe.** Invalidates existing checkpoints, datasets, and benchmark logs; requires complete retraining. | **Destructive.** Renders all trained models and benchmarks meaningless. |
| **Recommended Scope** | **Immediate resolution for Phase 1.** Clean, mathematically honest, and preserves research integrity. | **Recommended for future milestone (Phase 3 / Milestone 5).** | **Rejected.** Scientifically invalid for eavesdropping optimization. |

---

### 13.6 Rigorous Verification of Attack and Transit Noise Sequencing

We conducted extensive analytical and numerical verification of the diff in `src/bb84/bb84_engine.py` and `src/channel/quantum_channel.py`.

#### 1. Transit Noise Localization
- In `bb84_engine.py`:
  ```python
  qc.append(gate, [0, 1])  # Eve entangles carrier q0 with ancilla q1
  if is_noisy:
      qc.id(0)             # Transit noise is appended strictly to carrier q0
  ```
- Carrier qubit $q_0$ traverses the physical channel to Bob. Eve's ancilla $q_1$ is retained in Eve's laboratory and experiences zero transit noise. Verified that `id` is applied strictly to qubit 0.

#### 2. Singularity of Application
- Verified that `qc.id(0)` is executed exactly once per transmitted pulse.
- In `quantum_channel.py`, `self.apply_channel_transit_noise` is called exactly once when `is_noisy == True`, appending a single `id` operation to each probed circuit.

#### 3. Noisy vs Noiseless Branch Consistency
- In `BB84Engine`: when `is_noisy == False`, no `id` gate is appended; Bob receives the undisturbed probed state.
- In `QuantumChannel`: when `is_noisy == False`, `attack_res.probed_circuits` is returned directly without transit noise.

#### 4. Agreement with Independent Density-Matrix Reference Calculation
An independent script compared Qiskit Aer circuit simulation ($100,000$ shots) against an exact numerical Kraus operator density-matrix formulation:
$$\rho_{\text{ref}} = \sum_{j} K_{\text{noise}, j} \left( \sum_{i} K_{\text{probe}, i} \rho_0 K_{\text{probe}, i}^\dagger \right) K_{\text{noise}, j}^\dagger$$
Under Amplitude Damping noise ($\gamma = 0.20$, $\theta_E = \pi/8$):
- **Exact Density-Matrix Reference Error:** $0.117157$
- **Qiskit Aer Circuit Simulation Error (100k shots):** $0.115050$
- **Residual Difference:** $0.002107$ (well within $3\sigma = 0.00305$ binomial sampling tolerance).

#### 5. Documentation of Amplitude Damping Non-Commutativity
Evaluating the inverted operator order (Transit noise before Eve attack):
$$\rho_{\text{inverted}} = \sum_{i} K_{\text{probe}, i} \left( \sum_{j} K_{\text{noise}, j} \rho_0 K_{\text{noise}, j}^\dagger \right) K_{\text{probe}, i}^\dagger$$
- **Inverted Order Bob Error Rate:** $0.146447$
- **Discrepancy due to Non-Commutativity:** $|0.117157 - 0.146447| = 0.029289$ (nearly $3.0\%$ QBER distortion).
- **Commutativity Analysis for Other Channels:**
  - Depolarizing noise: $[\mathcal{E}_{\text{depol}}, \mathcal{E}_{\text{probe}}] = 0$ (density matrix discrepancy $= 0.0$).
  - Phase damping noise: $[\mathcal{E}_{\text{phase}}, \mathcal{E}_{\text{probe}}] = 0$ (density matrix discrepancy $= 0.0$).
  - Amplitude damping noise: $[\mathcal{E}_{\text{amp}}, \mathcal{E}_{\text{probe}}] \neq 0$. Applying the probe first reflects the physical setup where Eve taps the optical fibre before channel loss/attenuation completes.

---

### 13.7 Final Scientific Recommendations

1. **Adopt Alternative A for Phase 1 Closure:**
   Maintain the verified 2-qubit physical simulation for carrier disturbance and noise evolution. Formally define `theoretical_information_gain()` as an analytical theoretical objective (proxy) under stated optimal probe and delayed-measurement assumptions in all documentation and module docstrings.
2. **Preserve Training Pipeline and Checkpoints:**
   Do not alter existing datasets, checkpoints, or benchmark scripts. The optimization objective $\max_{\theta_E} I_{\text{AE}}(\theta_E)$ subject to $\text{QBER}(\theta_E) \le 0.11$ is mathematically and physically consistent under Alternative A.
3. **Plan Alternative B for a Future Research Phase:**
   When full physical POVM simulation is required, implement a dedicated 3-qubit `FullFuchsPeresBrandtProbe` module in an isolated branch for future milestones without disrupting the current milestone baselines.
