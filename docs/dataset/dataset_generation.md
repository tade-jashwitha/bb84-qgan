# Dataset Generation Methodology & Common Random Numbers

## 1. Generation Pipeline

Dataset generation is managed by `experiments/generate_full_dataset.py` using `src/dataset/generator.py`:
1. **Condition Iteration:** For each condition $c \in \{C01, \dots, C24\}$, the channel parameters are locked.
2. **Policy Iteration:** Each candidate policy in the 16-element lattice is applied.
3. **Simulation Pulse:** A physical transmission of $N=2,000$ quantum pulses is simulated.
4. **Observable Calculation:**
   - Sample QBER $E$.
   - Eve Mutual Information $I(A:E)$.
   - Protocol Abort Flag: $\mathbb{I}(E \ge QBER_{\text{tol}})$.
   - Wilson Score 95% Confidence Upper Bound $\text{UCB}_{95\%}(E)$.

---

## 2. Common Random Numbers (CRN) Design

To ensure that differences in observed error rates between competing eavesdropping policies reflect genuine physical differences rather than Monte Carlo seed variance, a strict Common Random Numbers (CRN) protocol is enforced:
- Alice's bit sequence and basis sequence are frozen per pulse run using a deterministic base seed hierarchy.
- Channel noise realizations utilize coupled pseudo-random streams.
- The protocol guarantees variance-reduced pairwise policy comparisons across all conditions.
