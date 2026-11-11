# Variational Quantum Generator Architecture (Milestone 4B Proposal)

## 1. Research Motivation

The classical cGAN benchmark results demonstrated that while an MLP generator achieves high feasibility yield ($81.25\%$), its continuous policy exploration struggles against analytical heuristics when budget constraints are relaxed. A Parameterized Quantum Circuit (PQC) generator offers a natural, quantum-native inductive bias for representing non-classical correlations and exploring complex multi-objective Pareto frontiers under NISQ constraints.

---

## 2. Circuit Ansatz Specification

### 1. Qubit Allocation & Condition Encoding
- **Qubit Register:** 4 to 8 qubits.
- **Condition Embedding:** Angle embedding using $R_Y(\alpha_i)$ rotations:
  $$|\psi_0(\vec{c})\rangle = \bigotimes_{i=1}^{d_c} R_Y(\arctan(c_i)) |0\rangle$$

### 2. Strongly Entangling Variational Layers
- **Parameterized Rotations:** Each layer applies single-qubit rotations $R(\theta) = R_Z(\theta_3) R_Y(\theta_2) R_Z(\theta_1)$ on every qubit.
- **Entangling Gates:** Circular ring of two-qubit CNOT gates:
  $$\text{CNOT}_{j, (j+1) \pmod N}$$
- **Layer Depth:** $L \in \{2, 4, 6\}$ layers.

### 3. Measurement & Policy Decoding
- Expectation values of Pauli observables:
  $$\langle Z_1 \rangle \implies p_{\text{intercept}}, \quad \langle Z_2 \rangle \implies \theta_E, \quad \langle Z_3 \rangle \implies \text{family}$$
- Differentiable training via parameter-shift rules.
