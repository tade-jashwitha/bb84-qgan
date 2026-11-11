# Environmental Condition Grid & Dataset Architecture

## 1. Environmental Matrix (C01–C24)

The offline dataset spans 24 environmental conditions representing realistic optical fiber and free-space quantum channels:

| Condition ID | Depolarizing Noise $p$ | Dephasing Noise $\lambda$ | SPAM Error $\epsilon$ | Abort Threshold $QBER_{\text{tol}}$ | Partition |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **C01–C16** | $0.00$ to $0.07$ | $0.00$ to $0.07$ | $0.00$ to $0.03$ | $0.08$ to $0.15$ | **Training** (16 conditions) |
| **C17–C20** | $0.02$ to $0.06$ | $0.02$ to $0.06$ | $0.01$ to $0.02$ | $0.09$ to $0.14$ | **Validation** (4 conditions) |
| **C21–C24** | $0.01$ to $0.07$ | $0.01$ to $0.07$ | $0.01$ to $0.03$ | $0.08$ to $0.13$ | **Held-Out Test** (4 conditions) |

---

## 2. Policy Action Lattice

For each environmental condition, a structured 16-point policy lattice covers:
- Null policy (zero disturbance, baseline channel noise).
- Probe attacks with fine-grained angles $\theta_E \in [0.05, 0.45]$ rad.
- Partial Intercept-Resend attacks with intercept probabilities $p_{\text{int}} \in [0.05, 0.40]$.
- Hybrid composite attacks.

This yields $24 \times 16 = 384$ distinct condition-policy pairs, each replicated across calibration seeds to construct the 24,000-sample authoritative dataset in `results/full_training_dataset.json`.
