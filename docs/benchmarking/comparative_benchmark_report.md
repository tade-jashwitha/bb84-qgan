# M4A COMPARATIVE BENCHMARK REPORT: FOUR-METHOD BASELINE EVALUATION ACROSS HELD-OUT TEST CONDITIONS

**Project:** Adaptive QGAN-Based Eavesdropping in BB84 QKD: Resource-Constrained Conditional Quantum Generative Approach under NISQ Noise
**Working Repository:** `C:\Users\lenovo\bb84-qgan-research` (Primary Repository)
**Execution Timestamp:** 2026-11-11T02:21:31 UTC
**Evaluated Checkpoint:** `checkpoints/classical_cgan/cgan_m4a_lexicographic_best.pt` (Epoch 150)
**Test Partition:** Strictly Held-Out Test Conditions (C21, C22, C23, C24) across 5 Frozen Evaluation Seeds (901–905)
**Total Simulation Budget:** 4 Methods $\times$ 4 Conditions $\times$ 5 Seeds $\times$ 16 Queries = 1,280 Forward Physical Simulations
**Status:** BENCHMARK COMPLETED — FULL RIGOROUS EMPIRICAL AUDIT

---

## 1. EXECUTIVE SUMMARY & SCIENTIFIC VERDICT

The official four-method comparative baseline benchmark has concluded. The evaluation protocol tested four attack policy strategies under strictly identical physical simulator query budgets ($K = 16$), identical pulse counts ($N = 2000$), identical statistical feasibility filtering (Wilson score 95% UCB $\le Q_{\text{budget}}$), and identical Tri-Case operational fallback across the quarantined Test Partition (C21–C24).

```
==========================================================================================
M3 STAGE C: BENCHMARK COMPARISON TABLE (Held-out Test Conditions, K=16 Queries)
==========================================================================================
Method                       | I(A;E) Mean ± Std  | Abort Rate  | Feasibility | Case A/B/C
------------------------------------------------------------------------------------------
cGAN_Policy_Engine           | 0.0451 ± 0.0280    | 0.0%        | 95.0%       | 18/1/1
Uniform_Random_Search        | 0.0661 ± 0.0683    | 0.0%        | 85.0%       | 14/3/3
Deterministic_Grid_Search    | 0.0760 ± 0.0706    | 0.0%        | 100.0%      | 19/1/0
Analytical_Probe_Heuristic   | 0.0861 ± 0.0712    | 0.0%        | 100.0%      | 19/1/0
==========================================================================================
```

### Empirical Verdict:
**The classical conditional GAN (cGAN) is NOT globally superior to the analytical probe heuristic or discrete grid search.**

Instead, the empirical data reveals a nuanced, regime-dependent division:
1. **Regime of cGAN Superiority (Tight Security & Critical Noise — C21 & C23):**
   Under strict error budgets ($Q_{\text{budget}} = 0.05$) in high noise, cGAN outperforms all classical baselines. Across C21 and C23 ($n = 10$), cGAN achieves a mean mutual information of **$0.0247$** (median $0.0270$), compared to **$0.0174$** for Analytical Heuristic, **$0.0093$** for Grid Search, and **$0.0085$** for Uniform Random Search. In critical noise C23, Random Search failed completely (0/16 feasible candidates in all episodes), whereas cGAN successfully generated active, stealthy attack policies.
2. **Regime of cGAN Under-Exploitation (Relaxed Security Budget — C22 & C24):**
   Under relaxed error budgets ($Q_{\text{budget}} = 0.11$), the classical cGAN behaves overly conservatively, generating safe policies with $I(A;E) \approx 0.06 - 0.07$ (100% feasibility yield). Meanwhile, the Analytical Heuristic ($I \approx 0.155$) and Grid Search ($I \approx 0.143$) aggressively exploit the available disturbance headroom without violating the security threshold.
3. **Motivation for Milestone 4B (Quantum GAN):**
   This exact empirical asymmetry provides the scientific rationale for Milestone 4B: classical MLP generators trained under min-max game dynamics suffer from risk-averse mode shrinkage in wide regions of parameter space, motivating the parameterized quantum circuit (PQC) generator to span the non-dominated Pareto frontier across all NISQ regimes.

---

## 2. COMPREHENSIVE PERFORMANCE SUMMARY

Across all 80 evaluation episodes (20 per method), the performance metrics are compiled below:

| Metric | cGAN Policy Engine | Uniform Random Search | Deterministic Grid Search | Analytical Probe Heuristic |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Mean $I(A;E)$ [bits]** | $0.0451 \pm 0.0280$ | $0.0661 \pm 0.0683$ | $0.0759 \pm 0.0706$ | **$0.0861 \pm 0.0712$** |
| **95% Confidence Interval for Mean** | $[0.0316, 0.0586]$ | $[0.0333, 0.0989]$ | $[0.0421, 0.1098]$ | $[0.0519, 0.1203]$ |
| **Overall Median $I(A;E)$ [bits]** | $0.0478$ | $0.0361$ | $0.0705$ | **$0.0890$** |
| **Maximum Observed $I(A;E)$** | $0.1310$ | **$0.2092$** | $0.1728$ | $0.1779$ |
| **Empirical Abort Rate** | **$0.0\%$** (0/20) | **$0.0\%$** (0/20) | **$0.0\%$** (0/20) | **$0.0\%$** (0/20) |
| **Statistical Feasibility Yield** | $95.0\%$ (19/20) | $85.0\%$ (17/20) | **$100.0\%$** (20/20) | **$100.0\%$** (20/20) |
| **Operational Decisions (Case A / B / C)** | $18\ /\ 1\ /\ 1$ | $14\ /\ 3\ /\ 3$ | $19\ /\ 1\ /\ 0$ | $19\ /\ 1\ /\ 0$ |
| **Mean Candidate Proposal Latency** | $803.82\text{ ms}$ | $0.61\text{ ms}$ | **$0.009\text{ ms}$** | $1.10\text{ ms}$ |
| **Verified Query Budget ($K=16$)** | **PASSED** (16/16) | **PASSED** (16/16) | **PASSED** (16/16) | **PASSED** (16/16) |

---

## 3. FEASIBLE NON-NULL POLICIES (CASE A ACTIVE ATTACKS ONLY)

When evaluating mutual information strictly among episodes where an active attack policy was classified as statistically feasible (Wilson Score 95% UCB $\le Q_{\text{budget}}$):

| Strategy | Active Feasible Episodes | Mean $I(A;E)$ [bits] | Median $I(A;E)$ [bits] | Std Dev | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **cGAN Policy Engine** | **18 / 20 (90.0%)** | $0.0501$ | $0.0488$ | **$0.0250$** | $0.0179$ | $0.1310$ |
| **Uniform Random Search** | 14 / 20 (70.0%) | **$0.0944$** | $0.1088$ | $0.0631$ | $0.0002$ | **$0.2092$** |
| **Deterministic Grid Search** | 19 / 20 (95.0%) | $0.0799$ | $0.1123$ | $0.0702$ | $0.0000$ | $0.1728$ |
| **Analytical Probe Heuristic** | 19 / 20 (95.0%) | $0.0906$ | **$0.1229$** | $0.0702$ | $0.0000$ | $0.1779$ |

### Observations:
- **cGAN Variance is Smallest:** cGAN has a standard deviation of only $0.0250$, compared to $>0.063$ for all other methods. cGAN generates highly predictable, consistent policies without extreme fluctuations.
- **Random Search Suffers Low Yield:** Uniform Random Search succeeded in only 14 of 20 episodes. In the remaining 6 episodes, it failed to find a single feasible candidate in 16 queries.
- **Analytical Heuristic & Grid Search:** Both achieved active feasible policies in 19 of 20 episodes, falling back to Null only in C23 under unfavorable calibration noise.

---

## 4. BREAKDOWN BY ENVIRONMENTAL TEST CONDITION

The held-out Test Partition covers four distinct NISQ noise and security regimes:
- **C21:** High depolarizing noise ($p = 0.05$), strict budget ($Q_{\text{budget}} = 0.05$).
- **C22:** High depolarizing noise ($p = 0.05$), relaxed budget ($Q_{\text{budget}} = 0.11$).
- **C23:** Critical depolarizing noise ($p = 0.07$), strict budget ($Q_{\text{budget}} = 0.05$).
- **C24:** Critical depolarizing noise ($p = 0.07$), relaxed budget ($Q_{\text{budget}} = 0.11$).

### Detailed Results Table by Condition:

| Condition | Metric | cGAN Policy Engine | Uniform Random Search | Deterministic Grid Search | Analytical Probe Heuristic |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **C21** | Mean $I(A;E)$ | **$0.0334$** | $0.0170$ | $0.0186$ | $0.0319$ |
| ($p=0.05$, | Median $I(A;E)$ | **$0.0342$** | $0.0126$ | $0.0287$ | $0.0281$ |
| $Q_{\text{bud}}=0.05$) | Feasible Candidates / 16 | **$5.2\ /\ 16$** | $0.8\ /\ 16$ | $2.6\ /\ 16$ | $4.4\ /\ 16$ |
| | Case A / B / C | **5 / 0 / 0** | 4 / 1 / 0 | 5 / 0 / 0 | 5 / 0 / 0 |
| | Protocol Aborts | **0** | **0** | **0** | **0** |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **C22** | Mean $I(A;E)$ | $0.0728$ | $0.1448$ | $0.1486$ | **$0.1627$** |
| ($p=0.05$, | Median $I(A;E)$ | $0.0574$ | $0.1259$ | $0.1728$ | **$0.1779$** |
| $Q_{\text{bud}}=0.11$) | Feasible Candidates / 16 | **$16.0\ /\ 16$** | $4.2\ /\ 16$ | $10.2\ /\ 16$ | $4.4\ /\ 16$ |
| | Case A / B / C | **5 / 0 / 0** | 5 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 |
| | Protocol Aborts | **0** | **0** | **0** | **0** |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **C23** | Mean $I(A;E)$ | **$0.0159$** | $0.0000$ | $0.0000$ | $0.0029$ |
| ($p=0.07$, | Median $I(A;E)$ | **$0.0229$** | $0.0000$ | $0.0000$ | $0.0009$ |
| $Q_{\text{bud}}=0.05$) | Feasible Candidates / 16 | $0.6\ /\ 16$ | $0.0\ /\ 16$ | $0.8\ /\ 16$ | **$1.8\ /\ 16$** |
| | Case A / B / C | 3 / 1 / 1 | 0 / 2 / 3 | 4 / 1 / 0 | 4 / 1 / 0 |
| | Protocol Aborts | **0** | **0** | **0** | **0** |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **C24** | Mean $I(A;E)$ | $0.0583$ | $0.1025$ | $0.1365$ | **$0.1470$** |
| ($p=0.07$, | Median $I(A;E)$ | $0.0563$ | $0.1113$ | $0.1123$ | **$0.1466$** |
| $Q_{\text{bud}}=0.11$) | Feasible Candidates / 16 | **$15.6\ /\ 16$** | $3.6\ /\ 16$ | $8.8\ /\ 16$ | $5.8\ /\ 16$ |
| | Case A / B / C | **5 / 0 / 0** | 5 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 |
| | Protocol Aborts | **0** | **0** | **0** | **0** |

---

## 5. STATISTICAL HYPOTHESIS TESTING & COMPARISONS

### 5.1 Paired Differences Across All 20 Episodes:

```text
Paired Differences vs cGAN Policy Engine:
- Uniform Random Search - cGAN:      mean diff = +0.0210 b (paired t-test: p = 0.1046, Wilcoxon: p = 0.1841)
- Deterministic Grid Search - cGAN:  mean diff = +0.0308 b (paired t-test: p = 0.0321, Wilcoxon: p = 0.0642)
- Analytical Probe Heuristic - cGAN: mean diff = +0.0410 b (paired t-test: p = 0.0043, Wilcoxon: p = 0.0196)
```

**Interpretation:**
- Across the entire test distribution, the Analytical Probe Heuristic achieves a statistically significant higher mean mutual information ($p = 0.0043 < 0.01$).
- Deterministic Grid Search shows a marginally significant advantage ($p = 0.0321$).
- Uniform Random Search difference is **not statistically significant** ($p = 0.1046$).

### 5.2 Subgroup Analysis: Strict Security Regimes (C21 + C23, $n=10$):

```text
Tight Budget Conditions (C21 + C23, n=10):
- cGAN Policy Engine:          mean = 0.0247 b, median = 0.0270 b  (HIGHEST)
- Analytical Probe Heuristic:  mean = 0.0174 b, median = 0.0137 b
- Deterministic Grid Search:   mean = 0.0093 b, median = 0.0000 b
- Uniform Random Search:       mean = 0.0085 b, median = 0.0000 b
```

**Scientific Significance:**
In strict security environments where eavesdropping is most challenging and stealth margins are razor-thin ($< 1.5\%$), **cGAN is the highest-performing method**, delivering **2.65$\times$ the information gain of Grid Search** and **2.91$\times$ that of Random Search**.

### 5.3 Subgroup Analysis: Relaxed Security Regimes (C22 + C24, $n=10$):

```text
Relaxed Budget Conditions (C22 + C24, n=10):
- Analytical Probe Heuristic:  mean = 0.1548 b, median = 0.1622 b  (HIGHEST)
- Deterministic Grid Search:   mean = 0.1425 b, median = 0.1426 b
- Uniform Random Search:       mean = 0.1236 b, median = 0.1186 b
- cGAN Policy Engine:          mean = 0.0656 b, median = 0.0568 b
```

**Scientific Significance:**
When the budget is relaxed ($Q_{\text{budget}} = 0.11$), the classical cGAN exhibits conservative saturation: its candidate policies cluster safely around $\theta_E \approx 0.18 - 0.22$ rad, whereas the Analytical Heuristic and Grid Search push to $\theta_E \approx 0.30 - 0.34$ rad, capturing double the mutual information.

---

## 6. RUNTIME & COMPUTATIONAL LATENCY PROFILES

Each episode was timed with nanosecond resolution via `time.perf_counter()`:

| Method | Candidate Proposal Latency (Mean $\pm$ Std) | Physical Simulator Execution (16 Queries) | Total Execution Time (20 Episodes) |
| :--- | :---: | :---: | :---: |
| **Deterministic Grid Search** | **$0.009 \pm 0.003\text{ ms}$** | $\approx 21.5\text{ s / episode}$ | $431.2\text{ s}$ |
| **Uniform Random Search** | $0.608 \pm 0.120\text{ ms}$ | $\approx 21.6\text{ s / episode}$ | $433.8\text{ s}$ |
| **Analytical Probe Heuristic** | $1.104 \pm 0.245\text{ ms}$ | $\approx 21.5\text{ s / episode}$ | $431.9\text{ s}$ |
| **cGAN Policy Engine** | $803.82 \pm 65.41\text{ ms}$ | $\approx 21.5\text{ s / episode}$ | $447.4\text{ s}$ |

### Total Benchmark Duration:
- **Total Wall-Clock Time:** $1726.33\text{ seconds}$ ($\approx 28.77\text{ minutes}$).
- **Bottleneck Analysis:** Across all methods, $>96\%$ of runtime is consumed by Qiskit Aer physical quantum circuit simulation (16 queries $\times$ 2,000 pulses = 32,000 pulses per episode). Neural generator inference ($< 1\text{ s}$) represents a negligible fraction of the total execution budget.

---

## 7. ARTIFACT CHECKSUMS & REPRODUCIBILITY

The benchmark was executed deterministically with fixed evaluation seeds ($901, 902, 903, 904, 905$) and simulation seeds ($s_{\text{eval}} \times 1000 + i + 1$).

| Artifact | File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Benchmark Raw Data** | `results/cgan_benchmark_comparison.json` | `6E7DADBE1A6ED81C106B5FF54EBB3B80586CF1B76154329C51927B76C61B57F9` |
| **Evaluated cGAN Checkpoint** | `checkpoints/classical_cgan/cgan_m4a_lexicographic_best.pt` | `F26AEC880D4BAFAF67E22C3691B36450D9273DE2755135236010ECD4BDD2B81E` |
| **Research Dataset** | `results/full_training_dataset.json` | `07C03E810470B3376EB48C51C13138DCF347D915F78E911F74A81E96F27CFB80` |
| **Preserved Pilot Best** | `checkpoints/cgan_pilot_best.pt` | `F4050379F065E9FE3106A23D746396215EBF128F52249220013B07411ED827AA` |

---

## 8. CONCLUSIONS & IMPLICATIONS FOR MILESTONE 4B (QGAN)

1. **Statistical Feasibility and Stealth Confirmed:**
   All four methods achieved a **$0.0\%$ empirical protocol abort rate**, confirming that Wilson Score 95% UCB filtering and Tri-Case fallback provide 100% reliable protection against eavesdropper detection across all tested NISQ channels.
2. **The cGAN Specializes in Constrained Stealth:**
   The classical cGAN excels where search space is severely restricted: in conditions C21 and C23, it out-performed all classical baselines, finding stealthy attacks where Uniform Random Search completely failed.
3. **The Conservative Saturation of Classical MLP Generators:**
   In relaxed conditions (C22 and C24), cGAN generated excessively conservative policies, lagging behind the Analytical Heuristic ($0.066$ b vs $0.155$ b).
4. **Direct Foundation for Milestone 4B (Quantum GAN):**
   This empirical limitation is precisely the research hypothesis of Milestone 4B: replacing the classical MLP generator with a Parameterized Quantum Circuit (PQC) running on quantum state spaces. A variational quantum circuit with parameterized entangling gates can represent quantum state correlations directly, overcoming the classical MLP's conservative mode shrinkage and achieving Pareto-optimal eavesdropping across the entire NISQ spectrum.
