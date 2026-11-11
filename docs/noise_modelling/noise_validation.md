# Quantum Noise Models & Empirical Validation

## 1. Channel Noise Formulations

Implemented in `src/channel/noise_models.py` and `src/channel/quantum_channel.py`.

### 1. Depolarizing Channel
Isotropic contraction of the Bloch sphere:
$$\mathcal{E}_{\text{depol}}(\rho) = (1 - p)\rho + \frac{p}{3}\left(X\rho X + Y\rho Y + Z\rho Z\right)$$
Induced QBER in BB84:
$$E_{\text{depol}}(p) = \frac{p}{2}$$

### 2. Amplitude Damping Channel
Models energy dissipation and photon loss into a zero-temperature bath:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

### 3. Phase Damping Channel
Models pure environmental dephasing without energy loss:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \lambda} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}$$
Asymmetric effect: leaves $\mathcal{Z}$-basis invariant while inducing bit flips in $\mathcal{X}$-basis:
$$E_{\mathcal{X}}(\lambda) = \frac{1 - \sqrt{1 - \lambda}}{2}$$

### 4. Thermal Relaxation Channel
Full Lindblad dynamics parameterised by $T_1$ (relaxation), $T_2$ (dephasing), and gate duration $t_{\text{gate}}$:
$$p_{\text{reset}} = 1 - e^{-t / T_1}, \quad p_{\text{phase}} = 1 - e^{-t / T_2^*}$$

---

## 2. Experimental Verification

Validation sweeps executed in `experiments/run_noise_experiments.py` verified the theoretical curves across all parameter ranges, serialized in `results/quantum_noise_results.json`.
