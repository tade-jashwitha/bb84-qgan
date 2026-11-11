# M4A CUDA GENERATOR DEVICE FIX REPORT
**Project:** Adaptive QGAN-Based Eavesdropping in BB84 QKD: Resource-Constrained Conditional Quantum Generative Approach under NISQ Noise  
**Repository:** `C:\Users\lenovo\bb84-qgan-research`  
**Date:** November 10, 2026  
**Status:** RESOLVED — DEVICE MISMATCH FIXED, SMOKE TEST & 150/150 REGRESSION TESTS PASSING  

---

## 1. DIAGNOSTIC ROOT-CAUSE ANALYSIS

### 1.1 The Failure
During initialization of `Milestone4ATrainer` in `experiments/train_classical_cgan.py`, execution failed with:
```text
RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
```

### 1.2 Inspection of Random Generator Calls
A systematic audit across all files in `experiments/`, `src/cgan/`, and `src/bb84/` was conducted:
1. **Lines 128 & 138 of `experiments/train_classical_cgan.py`:**
   - Previous code:
     ```python
     latent_gen = torch.Generator().manual_seed(seed + 999)
     self.fixed_val_latents = torch.randn(
         num_candidates_eval,
         self.generator.latent_dim,
         generator=latent_gen,
         device=self.device,
         dtype=torch.float32,
     )

     cf_gen = torch.Generator().manual_seed(seed + 1234)
     self.fixed_cf_latents = torch.randn(
         num_candidates_eval,
         self.generator.latent_dim,
         generator=cf_gen,
         device=self.device,
         dtype=torch.float32,
     )
     ```
   - **Root Cause:** In PyTorch, `torch.Generator()` without arguments initializes on `device='cpu'`. When `device=self.device` (where `self.device` is `cuda:0` on the NVIDIA RTX 3050 Laptop GPU) was passed to `torch.randn(...)`, PyTorch strictly checked that `generator.device == tensor.device` and raised a `RuntimeError`.
2. **Other `torch.randn` calls in `experiments/train_classical_cgan.py`:**
   - Lines 251, 274, 334: `z = torch.randn(b_size, self.generator.latent_dim, device=self.device, dtype=torch.float32)`
   - These calls use PyTorch's default generator for `self.device` (which was seeded on CUDA via `torch.cuda.manual_seed_all(seed)` at line 124). No explicit generator mismatch exists.
3. **`src/cgan/models.py` & `src/cgan/trainer.py`:**
   - All `torch.randn` calls in generator and discriminator forward passes use `device=c.device` or `device=self.device` without explicit mismatched generators.

---

## 2. APPLIED FIX

In accordance with the approved CUDA specification, `experiments/train_classical_cgan.py` was updated at lines 128 and 138 to explicitly bind the random generators to `self.device`:

```python
        # Fixed validation latents for exact reproducibility across epochs
        latent_gen = torch.Generator(device=self.device).manual_seed(seed + 999)
        self.fixed_val_latents = torch.randn(
            num_candidates_eval,
            self.generator.latent_dim,
            generator=latent_gen,
            device=self.device,
            dtype=torch.float32,
        )

        # Fixed counterfactual latents for controlled sensitivity evaluation
        cf_gen = torch.Generator(device=self.device).manual_seed(seed + 1234)
        self.fixed_cf_latents = torch.randn(
            num_candidates_eval,
            self.generator.latent_dim,
            generator=cf_gen,
            device=self.device,
            dtype=torch.float32,
        )
```

### Key Properties of this Fix:
- **Smallest Correct Change:** Exactly two lines modified (`torch.Generator()` $\to$ `torch.Generator(device=self.device)`).
- **Dual-Device Compatibility:** When `self.device` is `cuda`, generators are instantiated as CUDA generators (`torch.Generator(device='cuda')`). When falling back to `cpu`, generators are instantiated as CPU generators (`torch.Generator(device='cpu')`).
- **Stream Isolation:** `latent_gen` and `cf_gen` remain dedicated, isolated instances seeded with `seed + 999` and `seed + 1234`. They never advance or alter the default random stream used by the training loop.
- **Architectural Preservation:** Zero changes to neural network layers, loss formulations, training epochs, learning rates, or Gumbel schedules.

---

## 3. REPRODUCIBILITY & SMOKE TEST VERIFICATION

A standalone initialization smoke test was executed directly on the RTX 3050 GPU:
1. **CUDA Tensor Generation:** `fixed_val_latents` and `fixed_cf_latents` generated directly on `cuda:0` without errors. Shape: `(16, 16)`, dtype: `torch.float32`.
2. **Determinism (Same Seed):** Two `Milestone4ATrainer` instances initialized with `seed=42` produced **100% bit-exact** validation and counterfactual latents (`torch.equal(t1, t2) == True`).
3. **Sensitivity (Different Seed):** Initializing with `seed=12345` produced distinct latent representations (`torch.equal(t1, t3) == False`).
4. **CPU Fallback:** Verified that mocking `torch.cuda.is_available() -> False` initializes smoothly on CPU with device `cpu`.
5. **No Training Loop Triggered:** Smoke test exited immediately after latent verification without executing any training steps.

---

## 4. REGRESSION SUITE RESULTS

- **cGAN Pipeline Unit Tests (`tests/test_cgan_pipeline.py`):** **25 / 25 PASSED** (including new `TestGeneratorDeviceCompatibility`).
- **Full Repository Regression Suite (`tests/test_*.py`):** **150 / 150 PASSED** in 60.81 seconds.

---

## 5. REPOSITORY INTEGRITY & ARTIFACT HASHES

| File | Status | SHA-256 Hash |
| :--- | :--- | :--- |
| `results/full_training_dataset.json` | UNMODIFIED | `07C03E810470B3376EB48C51C13138DCF347D915F78E911F74A81E96F27CFB80` |
| `checkpoints/cgan_pilot_best.pt` | UNMODIFIED | `F4050379F065E9FE3106A23D746396215EBF128F52249220013B07411ED827AA` |
| `checkpoints/cgan_pilot_latest.pt` | UNMODIFIED | `0BAA0DBD0BE0FB4D75E752AFD2EB1E86045CBCDFB4D1468DA27B0A2BF618E1D5` |
| `experiments/train_classical_cgan.py` | MODIFIED (fix) | `D8DCECEB31F9F80A94A57C886A35BF0348632E3E2693502E7E404AA2E517F8DE` |
| `tests/test_cgan_pipeline.py` | MODIFIED (test) | `7698E968FE842BB088222C04BE7C67FA4362208A19F85DA98C061783CCEC1CA6` |
| **Git Baseline Commit HEAD** | PRESERVED | `23148a4956c2c1975d940f14fd064eaa3a9cb7ad` |

