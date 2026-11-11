# Comparative Benchmark Baseline Implementations

## 1. Evaluated Engines

Four distinct eavesdropping policy engines are implemented and evaluated under identical conditions in `experiments/benchmark_baselines.py`:

### 1. Conditional Generative Adversarial Network (cGAN)
- **Architecture:** MLP Generator $G: \mathbb{R}^{16} \times \mathbb{R}^2 \to \mathbb{R}^3$ (latent dim 16, hidden layers [128, 128], LeakyReLU activations).
- **Execution:** Generates $K=16$ candidate policy vectors per episode. Validates feasibility and returns the candidate maximizing predicted mutual information.

### 2. Uniform Random Policy Search
- Samples $K=16$ candidates uniformly at random from the physical parameter bounding box:
  $$p_{\text{int}} \in [0, 0.40], \quad \theta_E \in [0, \pi/4], \quad \text{family} \in \{\text{Probe, PIR}\}$$

### 3. Deterministic Grid Search
- Evaluates a fixed $4 \times 4$ regular lattice ($K=16$) covering the operational parameter space.

### 4. Analytical Probe Heuristic
- Implemented in `src/cgan/policy_engine.py` (`AnalyticalProbePolicyEngine`).
- Directly calculates the allowable probe disturbance from calibrated channel noise:
  $$D^* = \max\left(0, \frac{QBER_{\text{tol}} - E_{\text{noise}}}{1 - 2 E_{\text{noise}}}\right)$$
- Maps to optimal probe rotation angle using the exact inverse physical model:
  $$\theta_E^* = \arcsin(\sqrt{D^*})$$
