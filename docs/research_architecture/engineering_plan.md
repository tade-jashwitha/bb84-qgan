# Engineering & Research Plan: Adaptive QGAN-Based Eavesdropping in BB84 QKD under NISQ Noise

**Project:** Adaptive QGAN-Based Eavesdropping in BB84 QKD: Resource-Constrained Conditional Quantum Generative Approach under NISQ Noise  
**Role:** Senior Quantum Computing Research Engineer & Python Mentor  
**Environment Target:** Python 3.11.9, Qiskit 2.6.0, Qiskit-Aer 0.18.0, PyTorch 2.5.1 (CUDA enabled), NumPy 2.4.6, SciPy 1.17.1  

---

## 1. System Architecture & Data Flow

### 1.1 High-Level Architecture Overview
The system models a full end-to-end simulation of the BB84 Quantum Key Distribution (QKD) protocol operating over a realistic noisy quantum channel (depolarizing, amplitude damping, phase damping, and Pauli channel models). An adaptive eavesdropper (Eve) leverages either:
1. **Classical Baseline (cGAN):** A classical Conditional GAN (MLP-based generator and discriminator in PyTorch).
2. **Quantum Generative Model (cQGAN):** A Parameterized Quantum Circuit (PQC) generator acting as a quantum neural network with variational angles conditioned on observed channel statistics, coupled with a hybrid classical/quantum discriminator.

```mermaid
flowchart TD
    subgraph BB84_Protocol [1. BB84 QKD Protocol Layer]
        AliceState[Alice: State Prep |0>, |1>, |+>, |->] --> ChannelIn[Quantum Channel Input]
        ChannelOut[Quantum Channel Output] --> BobMeas[Bob: Random Basis Measurement]
        BobMeas --> Sift[Sifting & QBER Estimation]
    end

    subgraph Channel_Noise [2. NISQ Noise Layer]
        ChannelIn --> NoiseEngine[Qiskit-Aer Noise Models: Depolarizing, Dephasing, Damping]
        NoiseEngine --> ChannelOut
    end

    subgraph Eavesdropper_Module [3. Adaptive Eavesdropper & Attack Engine]
        ChannelIn -.-> EveIntercept[Eve Probe / Intercept]
        EveIntercept --> EvePhysicalAction[Physically-defined Attack Action]
        EvePhysicalAction --> ChannelOut
        EveObs[Channel Observation: Estimated Error & Detection Budget] --> ConditionVector[Condition Vector c]
    end

    subgraph Generative_Core [4. Generative Models: cQGAN vs cGAN]
        ConditionVector --> GeneratorChoice{Model Selector}
        GeneratorChoice -->|cQGAN| PQC_Gen[Variational Quantum Circuit PQC]
        GeneratorChoice -->|Baseline| MLP_Gen[Classical PyTorch cGAN]
        PQC_Gen --> AttackParams[Attack Settings: θ probe, basis bias, probe ratio]
        MLP_Gen --> AttackParams
        AttackParams --> EvePhysicalAction
    end

    subgraph Evaluation_Bench [5. Evaluation & Benchmarking]
        Sift --> Metrics[Metrics Engine: QBER, I_AE, Detectability, Cost]
        EvePhysicalAction --> Metrics
    end
```

### 1.2 End-to-End Data Flow Pipeline
1. **Alice Stage:** Alice randomly samples a bit sequence $b_A \in \{0, 1\}^N$ and basis sequence $x_A \in \{Z, X\}^N$. She prepares quantum states $|\psi(b_A, x_A)\rangle \in \{|0\rangle, |1\rangle, |+\rangle, |-\rangle\}$.
2. **Eve Observation & Conditioning:** Eve samples channel quality or intercept history, forming a condition vector $c = (\text{QBER}_{\text{budget}}, \gamma_{\text{noise}})$, representing Eve's risk tolerance (detection threshold) and perceived channel noise.
3. **Attack Parameter Generation:**
   - **cQGAN:** Latent vector $z \sim \mathcal{N}(0, I)$ and condition $c$ are encoded into quantum state $|\Phi(z, c)\rangle$ via feature map. A Parameterized Quantum Circuit $U(\theta)$ transforms this state. Expectation values $\langle M_i \rangle$ or measurement samples parameterize the physical attack strategy.
   - **Classical cGAN Baseline:** Latent vector $z$ and $c$ are fed into an MLP $G_\phi(z, c)$ outputting attack parameters.
4. **Physical Attack Execution:** Eve applies a physically grounded interaction:
   - *Option A: Partial/Probabilistic Intercept-Resend:* Eve intercepts a fraction $p_{\text{attack}} \in [0, 1]$ of qubits in basis $x_E$ (or weighted random basis).
   - *Option B: Unitary Probe / Asymmetric Cloning:* Eve couples the carrier qubit with an ancillary probe qubit via a controlled rotation $U_{\text{probe}}(\theta_E)$, then measures the ancilla.
5. **Noise Transmission:** Qubits traverse the NISQ channel simulated via Qiskit Aer `NoiseModel` with parameterized noise rates (e.g., $p_{\text{depol}}$, $T_1$, $T_2$).
6. **Bob Measurement & Classical Exchange:** Bob measures in random basis $x_B \in \{Z, X\}^N$ and records bit sequence $b_B$. Alice and Bob perform sifting over a classical channel, revealing matching bases ($x_A = x_B$) and computing the sample Quantum Bit Error Rate (QBER).
7. **Adversarial Loss & Updates:**
   - Discriminator $D$ evaluates whether generated parameter/attack outcome tuples are "optimal eavesdropping configurations" under budget $c$ versus reference Pareto-optimal strategies.
   - Generator $G$ maximizes Eve's mutual information $I_{AE}$ while keeping induced QBER strictly below the abort threshold $Q_{\text{abort}}$ (typically $11\%$ in standard BB84).

---

## 2. Mathematical Formulations & Assumptions

### 2.1 BB84 States and Sifting
- $Z$-basis ($\sigma_z$): $|0\rangle$, $|1\rangle$.
- $X$-basis ($\sigma_x$): $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$, $|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$.
- Sifting condition: $S = \{i \in \{1, \dots, N\} \mid x_{A,i} = x_{B,i}\}$.
- Error calculation:
  $$\text{QBER} = \frac{1}{|S_{\text{test}}|} \sum_{i \in S_{\text{test}}} (b_{A,i} \oplus b_{B,i})$$

### 2.2 Channel & NISQ Noise Modeling
- **Depolarizing Channel:** $\mathcal{E}_{\text{depol}}(\rho) = (1 - p)\rho + \frac{p}{3}(\sigma_x \rho \sigma_x + \sigma_y \rho \sigma_y + \sigma_z \rho \sigma_z)$.
- **Amplitude Damping:** $\mathcal{E}_{\text{amp}}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger$ with $E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$, $E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$.
- **Phase Damping (Pure Dephasing):** $\mathcal{E}_{\text{phase}}(\rho) = M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger$ with $M_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\lambda} \end{pmatrix}$, $M_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}$.
- Natural channel noise induces a baseline error rate $\text{QBER}_{\text{env}}$. Eve's intervention induces additional error $\Delta \text{QBER}_{\text{Eve}}$. Total observed QBER:
  $$\text{QBER}_{\text{total}} \approx \text{QBER}_{\text{env}} + \Delta \text{QBER}_{\text{Eve}} - 2 \cdot \text{QBER}_{\text{env}} \cdot \Delta \text{QBER}_{\text{Eve}}$$

### 2.3 Physical Attack Parameterization
To prevent unphysical "magical" generative outputs, Eve's actions are restricted to real quantum operations:
1. **Probe Interaction Angle $\theta_E \in [0, \pi/2]$:**
   - Eve's probe couples with carrier: $U_E(\theta_E) |0\rangle_A |0\rangle_E = \cos\theta_E |0\rangle_A |0\rangle_E + \sin\theta_E |1\rangle_A |1\rangle_E$.
   - Induced QBER contribution: $D(\theta_E) = \frac{1}{2}\sin^2\theta_E$.
   - Eve's Mutual Information: $I_{AE}(\theta_E) = 1 - h_2\left(\frac{1 + \sin 2\theta_E}{2}\right)$, where $h_2(p) = -p\log_2 p - (1-p)\log_2(1-p)$.
2. **Selective Interception Ratio $\alpha \in [0, 1]$:** Fraction of qubits measured via projectives.
3. **Basis Bias $\beta \in [0, 1]$:** Eve's probability of choosing $Z$ versus $X$.

### 2.4 Explicit Differentiation: cQGAN vs. Classical cGAN

| Dimension | Classical cGAN Baseline | Genuine Quantum-Circuit cQGAN |
| :--- | :--- | :--- |
| **Generator Architecture** | PyTorch Multi-Layer Perceptron (Linear $\to$ LeakyReLU $\to$ Tanh/Sigmoid) | Parameterized Quantum Circuit (PQC) via Qiskit / PyTorch Hybrid (using ParameterVector & SamplerV2 / EstimatorV2) |
| **Input Representation** | Classical vector $[z, c] \in \mathbb{R}^{d_z + d_c}$ | Quantum State Embedding: Angle/Amplitude Embedding $S(z, c) |0\rangle^{\otimes n}$ |
| **Parameter Space** | Real-valued neural network weights $W, b$ | Continuous rotation gate angles $\vec{\theta} \in [0, 2\pi]^m$ (e.g. $RY, RZ, CNOT$ entangling layers) |
| **Nonlinearities** | Classical activation functions (ReLU, Sigmoid) | Quantum measurement expectation values & non-linear post-processing (or measurement projection) |
| **Gradient Estimation** | Standard backpropagation | Parameter-shift rule ($g_i = \frac{f(\theta + \frac{\pi}{2}) - f(\theta - \frac{\pi}{2})}{2}$) or PyTorch autograd bridge |
| **Discriminator** | Classical MLP $D_\psi(x, c)$ | Classical MLP or Hybrid PQC Discriminator |

*Strict Policy:* The project will implement both distinctly and directly compare their training convergence, parameter efficiency, and generative fidelity.

### 2.5 Assumptions & Operational Scope
- Alice and Bob follow standard BB84 with single-photon sources (or attenuated coherent pulses without decoy states unless specified).
- Alice and Bob have an authenticated classical channel (no man-in-the-middle impersonation on the classical channel).
- Detection threshold: $Q_{\text{abort}} = 0.11$ (11%). If $\text{QBER} \ge 0.11$, the key is discarded.
- Eve operates under NISQ constraints: limited circuit depth ($D \le 6$ layers in PQC), finite shots ($S \in [1000, 4000]$), and local gate errors on her own probe circuit.

---

## 3. Module & File Structure

The project will reside cleanly within the workspace:

```text
adaptive-qgan-bb84/
├── configs/
│   ├── default_bb84.yaml             # BB84 protocol parameters (num_qubits, sifting)
│   ├── noise_profiles.yaml           # NISQ channel noise specs (depol, damping rates)
│   ├── cqgan_config.yaml             # PQC depth, learning rate, condition dimensions
│   └── cgan_baseline_config.yaml     # PyTorch MLP baseline hyperparams
├── src/
│   ├── __init__.py
│   ├── protocol/
│   │   ├── __init__.py
│   │   ├── alice.py                  # State preparation & random basis choice
│   │   ├── bob.py                    # Measurement & classical sifting
│   │   └── bb84_engine.py            # End-to-end execution coordinator
│   ├── channel/
│   │   ├── __init__.py
│   │   ├── noise_models.py           # Qiskit Aer NoiseModel builders
│   │   └── quantum_channel.py        # Channel execution with noise injection
│   ├── eavesdropper/
│   │   ├── __init__.py
│   │   ├── attacks.py                # Physically parameterized attack implementations
│   │   └── adaptive_agent.py         # Eve runtime controller combining generator & attack
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cqgan.py                  # Genuine Quantum PQC Generator & Qiskit interface
│   │   ├── cgan_baseline.py          # Classical PyTorch cGAN baseline
│   │   ├── discriminator.py          # Condition-aware Discriminator architecture
│   │   └── hybrid_layers.py          # Torch-Qiskit autograd wrapper via Parameter-Shift
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # QBER, Shannon mutual info I_AE, detectability
│   │   ├── resource_counter.py       # Gate counts, circuit depth, shot cost, runtime
│   │   └── benchmark.py              # Comparative experiment harness
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py          # Pareto fronts, QBER vs I_AE plots, circuit draws
│       └── logger.py                 # Structured logging & run artifacts
├── tests/
│   ├── test_bb84_noiseless.py        # Validates theoretical 0% QBER on ideal channel
│   ├── test_noise_models.py          # Validates error rate scaling with noise params
│   ├── test_physical_attacks.py      # Checks theoretical vs simulated QBER for probe angles
│   ├── test_cqgan_circuit.py         # Validates PQC gradients, output shapes, and unitary validity
│   ├── test_cgan_baseline.py         # Validates PyTorch baseline forward & backward passes
│   └── test_metrics.py               # Unit tests on QBER & mutual information calculations
├── experiments/
│   ├── 01_baseline_attacks.py        # Static Intercept-Resend vs Unitary Probe
│   ├── 02_cgan_vs_cqgan_train.py     # Training convergence and parameter efficiency
│   ├── 03_detectability_tradeoff.py  # QBER vs I_AE Pareto frontier under noise
│   └── 04_cross_noise_generalize.py  # Train on Depolarizing, test on Damping/Phase noise
├── requirements.txt
└── README.md
```

---

## 4. Dependencies & Compatibility Requirements

Based on verification of the local runtime:
- **Python:** 3.11.9 (Compatible)
- **Qiskit Core:** 2.6.0 (Uses modern V2 primitives `SamplerV2`, `EstimatorV2`, `QuantumCircuit`)
- **Qiskit Aer:** 0.18.0 (Compatible for statevector and noisy density-matrix simulations)
- **PyTorch:** 2.5.1+cu121 (CUDA accelerated for neural network operations and hybrid backprop)
- **Scientific Computing:** NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2

*Note on Qiskit 2.x migration:* Qiskit 2.0+ deprecated legacy `qiskit.execute` and V1 primitives. All circuit executions will use `qiskit_aer.AerSimulator` with explicit `PassManager` transpilation and modern `SamplerV2` workflows to guarantee future-proof, deprecation-free code.

---

## 5. Experiments & Evaluation Metrics

### 5.1 Primary Metrics
1. **Quantum Bit Error Rate (QBER):**
   $$\text{QBER} = \frac{N_{\text{error}}}{N_{\text{sifted}}}$$
2. **Eve Information Gain ($I_{AE}$):**
   Mutual information between Alice's key bits and Eve's inferred bits on sifted positions:
   $$I_{AE} = 1 - H(A | E) = 1 + \sum_{a, e} P(a, e) \log_2 P(a | e)$$
3. **Attack Detectability Rate ($P_{\text{detect}}$):**
   Probability that observed QBER exceeds protocol threshold $Q_{\text{abort}} = 0.11$:
   $$P_{\text{detect}} = \Pr(\text{QBER} \ge 0.11)$$
4. **Adversarial Utility (Eve Objective):**
   Trade-off objective: $U_{\text{Eve}} = I_{AE} - \lambda \cdot \max(0, \text{QBER} - Q_{\text{target}})^2$.
5. **Computational & Quantum Resource Cost:**
   - Parameter count (Classical vs PQC).
   - Quantum circuit depth, 2-qubit (CNOT/CZ) gate count, and single-qubit rotation count.
   - Shot overhead per gradient update step (evaluating parameter-shift evaluation cost).

### 5.2 Planned Experimental Regimes
- **Experiment 1 (Validation of Physical Soundness):** Verify analytical formulas for intercept-resend ($\text{QBER} = 25\%$, $I_{AE} = 0.5$) and unitary probe against simulator outputs.
- **Experiment 2 (Model Comparison):** Train genuine cQGAN vs classical cGAN on depolarizing noise channel under identical condition distributions. Compare:
  - Convergence speed (epochs).
  - Parameter economy (e.g., ~20 quantum circuit parameters vs ~5,000 classical weights).
  - Stability of generated parameters (variance of predicted angles).
- **Experiment 3 (Pareto-Optimality under NISQ Noise):** Map out the $(\text{QBER}, I_{AE})$ frontier across varied noise levels $\gamma \in [0.01, 0.08]$. Determine whether Eve can hide within the ambient channel noise.
- **Experiment 4 (Cross-Noise Generalization):** Train models strictly on Depolarizing noise; evaluate zero-shot adaptation on Amplitude Damping and Phase Damping channels to assess robustness.

---

## 6. Unit Tests & Acceptance Criteria

| Component | Test Case | Acceptance Criteria |
| :--- | :--- | :--- |
| **Alice-Bob Ideal** | 4,000 qubits over ideal channel with Eve absent. | $\text{QBER} \equiv 0.0\%$, sifting efficiency $\approx 50\% \pm 2\%$. |
| **Standard Intercept-Resend** | Full interception with random basis. | $\text{QBER} = 25.0\% \pm 1.5\%$, $I_{AE} = 0.50 \pm 0.03$. |
| **Noise Model Scaling** | Channel with $p_{\text{depol}} = 0.06$ (no Eve). | Measured $\text{QBER} = 2.0\% \pm 0.3\%$ ($p/2$ bit flip on matching bases). |
| **PQC Circuit Validity** | cQGAN generator outputs. | Angles bounded in $[0, \pi/2]$, parameter-shift gradients finite and non-NaN. |
| **Classical cGAN** | PyTorch generator output bounds. | Generated parameters strictly within physical bounds via Sigmoid/Tanh scaling. |
| **Detectability Metric** | High-intensity attack vs low-intensity attack. | $P_{\text{detect}} \to 1.0$ when $\text{QBER} > 0.11$, $P_{\text{detect}} \to 0.0$ when $\text{QBER} < 0.05$. |

---

## 7. Hardware Limitations & Estimated Computational Cost

- **Simulation Mode:** Qiskit Aer `AerSimulator(method='density_matrix')` for noise modeling, or `method='statevector'` with stochastic quantum error trajectories.
- **Qubit Allocation:**
  - BB84 transmission: 1 qubit per pulse (or batch simulated sequentially).
  - Eve's probe: 1 carrier qubit + 1 ancilla qubit (2-qubit circuit).
  - cQGAN Generator: 3 to 5 variational qubits, depth 3–6 layers.
- **Memory Footprint:** Well within local RAM limits ($< 500$ MB for $N \le 6$ qubits).
- **Runtime Estimates (Local Intel/NVIDIA Machine):**
  - BB84 sample run (4,000 pulses): $\approx 1.5 - 3.5$ seconds.
  - cGAN (classical) training (100 epochs, batch size 64): $\approx 15 - 30$ seconds on GPU.
  - cQGAN training (parameter-shift rule, 50 epochs): $\approx 2 - 5$ minutes on CPU/GPU depending on shot count (e.g. 1024 shots per expectation).
- **Mitigation for Quantum Training Cost:** We will use exact statevector expectation computation during initial debugging and development, switching to shot-based Aer sampling for noisy realism.

---

## 8. Staged Execution Milestones

- **Milestone 1:** BB84 Core + Analytical Physical Attacks + Unit Tests.
- **Milestone 2:** NISQ Noise Channels (Depolarizing, Thermal/Damping) + Validation.
- **Milestone 3:** Classical cGAN Baseline (PyTorch) + Conditioning Engine.
- **Milestone 4:** Genuine Quantum-Circuit cQGAN (Qiskit PQC + Hybrid PyTorch Layer).
- **Milestone 5:** Comparative Benchmarking, Pareto Front Analysis, Cross-Noise Experiments, and Plotting.
