# Eavesdropping Attack Formulations

## 1. Probabilistic / Partial Intercept-Resend (PIR)

Implemented in `src/bb84/attacks.py` (`PartialInterceptResendAttack`).

### Theoretical Mechanics
For each transmitted photon:
1. With probability $p_{\text{int}} \in [0, 1]$, Eve intercepts the photon.
2. Eve selects a random measurement basis $\beta_E \in \{\mathcal{Z}, \mathcal{X}\}$ with equal probability.
3. Eve measures the photon, records her outcome, prepares a fresh state corresponding to her outcome in basis $\beta_E$, and forwards it to Bob.
4. With probability $1 - p_{\text{int}}$, Eve leaves the photon undisturbed.

### Disturbance and Mutual Information
When Alice and Bob share the same basis:
- If Eve chose the matching basis (probability $0.5$), zero error is induced.
- If Eve chose the orthogonal basis (probability $0.5$), Bob measures the incorrect bit with probability $0.5$.

Induced QBER:
$$E_{\text{PIR}}(p_{\text{int}}) = \frac{p_{\text{int}}}{4}$$

Eve Mutual Information with Alice:
$$I(A:E)_{\text{PIR}} = \frac{p_{\text{int}}}{2}$$

---

## 2. Unitary Entangling Probe (Fuchs-Peres-Brandt Model)

Implemented in `src/bb84/attacks.py` (`UnitaryProbeAttack`).

### Theoretical Mechanics
Eve prepares an ancillary probe in initial state $|w\rangle \in \mathcal{H}_E$. When Alice transmits signal state $|\psi\rangle \in \mathcal{H}_S$, Eve applies a coherent entangling unitary operation $U_{BE}$ on $\mathcal{H}_S \otimes \mathcal{H}_E$:

$$U_{BE}|0\rangle|w\rangle = \cos(\theta_E)|0\rangle|e_{00}\rangle + \sin(\theta_E)|1\rangle|e_{01}\rangle$$
$$U_{BE}|1\rangle|w\rangle = \sin(\theta_E)|0\rangle|e_{10}\rangle + \cos(\theta_E)|1\rangle|e_{11}\rangle$$

where $\theta_E \in [0, \pi/4]$ is Eve's probe rotation angle, and $\{|e_{ij}\rangle\}$ are orthonormal probe states.

### Physical Observable Relationships
In the symmetric Fuchs-Peres-Brandt entangling configuration:
1. **Signal Disturbance:**
   $$D(\theta_E) = \sin^2(\theta_E)$$
2. **Induced Channel Error Rate (Ideal Channel):**
   $$E = D(\theta_E) = \sin^2(\theta_E)$$
3. **Exact Inverse Mapping:**
   $$\theta_E^* = \arcsin(\sqrt{D^*})$$
4. **Eve Mutual Information:**
   $$I(A:E) = 1 - H(p_e) = 1 + p_e \log_2(p_e) + (1 - p_e) \log_2(1 - p_e)$$
   where $p_e = \frac{1 - \sin(2\theta_E)}{2}$.
