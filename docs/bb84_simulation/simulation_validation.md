# BB84 Simulation Engine Validation

## 1. Experimental Protocol Summary

The core quantum simulation suite was verified through five rigorous physical experiment suites (`experiments/run_bb84_simulation.py`) and recorded in `results/bb84_simulation_results.json`:
- **Experiment 1:** Baseline Ideal Channel ($N=10,000$ pulses).
- **Experiment 2:** Full Intercept-Resend Attack ($p_{\text{int}} = 1.0$).
- **Experiment 3:** Partial Intercept-Resend Sweep ($p_{\text{int}} \in [0.0, 1.0]$, step 0.1).
- **Experiment 4:** Asymmetric Basis Bias Analysis ($p_Z \in [0.1, 0.9]$).
- **Experiment 5:** Unitary Entangling Probe Sweep ($\theta_E \in [0, \pi/4]$).

---

## 2. Empirical Validation Results

| Test Regime | Theoretical Prediction | Empirical Value | Status |
|:---|:---:|:---:|:---:|
| **Ideal Channel QBER** | $0.0000$ | $0.0000$ | PASSED |
| **Ideal Sifted Key Rate** | $0.5000$ | $0.4994 \pm 0.005$ | PASSED |
| **Full PIR Induced QBER** | $0.2500$ | $0.2482 \pm 0.006$ | PASSED |
| **Full PIR Eve Info $I(A:E)$** | $0.5000$ | $0.4985$ | PASSED |
| **PIR Slope $\Delta E / \Delta p$** | $0.2500$ | $0.2491 \pm 0.003$ | PASSED |
| **Unitary Probe Max Disturbance** | $0.5000$ (at $\pi/4$) | $0.5012 \pm 0.005$ | PASSED |

All 160 unit tests in `tests/test_bb84_engine.py`, `tests/test_alice.py`, `tests/test_bob.py`, and `tests/test_attacks.py` pass without error.
