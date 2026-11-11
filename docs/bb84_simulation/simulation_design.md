# Physical BB84 Simulation Design & Sifting Protocol

## 1. State Space & Quantum Encoding

Alice encodes classical random bits $b_A \in \{0, 1\}$ into single-photon polarization quantum states chosen from two mutually unbiased bases (MUBs):
- **Rectilinear Basis ($\mathcal{Z}$):**
  $$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$
- **Diagonal Basis ($\mathcal{X}$):**
  $$|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

State preparation is implemented in `src/bb84/alice.py` (`Alice` class).

---

## 2. Quantum Transmission & Bob Measurement

1. **State Transmission:** Alice transmits quantum state $|\psi\rangle$ through the quantum channel (`src/bb84/channel.py`).
2. **Bob Basis Selection:** Bob independently and randomly selects measurement basis $b_B \in \{\mathcal{Z}, \mathcal{X}\}$ with equal probability ($p=0.5$).
3. **Projective Measurement:**
   - In $\mathcal{Z}$ basis: projectors $\Pi_0 = |0\rangle\langle 0|$ and $\Pi_1 = |1\rangle\langle 1|$.
   - In $\mathcal{X}$ basis: projectors $\Pi_+ = |+\rangle\langle +|$ and $\Pi_- = |-\rangle\langle -|$.
   - Outcome probability: $P(m) = \langle\psi|\Pi_m|\psi\rangle$.

Measurement is implemented in `src/bb84/bob.py` (`Bob` class).

---

## 3. Classical Sifting & QBER Estimation

Implemented in `src/bb84/bb84_engine.py` (`BB84Engine`):
1. **Basis Reconciliation:** Alice and Bob publicly announce their chosen bases over an authenticated classical channel.
2. **Sifting:** Instances where bases coincide ($b_A = b_B$) are retained; mismatched instances are discarded. Expected sifted key yield: $N_{\text{sift}} \approx 0.5 N$.
3. **Error Sampling:** A randomly chosen subset of sifted bits ($N_{\text{sample}}$) is disclosed to compute the sample Quantum Bit Error Rate:
   $$E = \frac{N_{\text{err}}}{N_{\text{sample}}}$$
4. **Abort Logic:**
   $$\text{Protocol Action} = \begin{cases} \text{PROCEED to Privacy Amplification}, & \text{if } E < QBER_{\text{tol}} \\ \text{ABORT (Eavesdropping Detected)}, & \text{if } E \ge QBER_{\text{tol}} \end{cases}$$
