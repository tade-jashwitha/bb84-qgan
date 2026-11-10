"""Unit tests for Bob module in BB84 protocol.

Validates:
1. Bob generates the correct number of random bases.
2. Bob's basis selection is reproducible with a fixed seed.
3. Bob's basis selection supports both Z and X (and only valid bases).
4. Z-basis measurement of |0> produces 0.
5. Z-basis measurement of |1> produces 1.
6. X-basis measurement of |+> produces 0 with probability 1.
7. X-basis measurement of |-> produces 1 with probability 1.
8. Measurement results preserve signal ordering.
9. Basis reconciliation retains only matching bases.
10. Sifted Alice and Bob keys have matching lengths.
11. QBER is zero when all sifted bits match.
12. QBER is 0.25 for one error among four sifted bits.
13. Empty sifted keys are handled correctly.
14. Invalid input values raise meaningful errors.
15. Mismatched input lengths are rejected.
16. Physical Validation Scenario A: Ideal BB84 (Efficiency ~50%, QBER = 0.0%).
17. Physical Validation Scenario B: X-basis deterministic correctness.
18. Bob class stateful lifecycle and RNG reset.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from src.bb84.alice import (
    Alice,
    prepare_bb84_state,
    prepare_bb84_circuits,
)
from src.bb84.bob import (
    Bob,
    SiftingResult,
    QBERResult,
    generate_random_bases,
    apply_measurement_basis,
    measure_pulse,
    measure_bb84_circuits,
    reconcile_bases,
    compute_qber,
    BASIS_Z,
    BASIS_X,
    VALID_BASES,
)


class TestBobRandomBases(unittest.TestCase):
    """Test suite for Bob's independent random basis generation."""

    def test_case_01_generate_bases_count_and_shape(self):
        """Case 1: Verify Bob generates the correct number of random bases."""
        for n in [1, 10, 250, 1000]:
            bases = generate_random_bases(n, rng=np.random.default_rng(42))
            self.assertEqual(len(bases), n)
            self.assertEqual(bases.shape, (n,))
            self.assertTrue(isinstance(bases, np.ndarray))

    def test_case_02_basis_selection_reproducible_with_seed(self):
        """Case 2: Verify Bob's basis selection is reproducible with a fixed seed."""
        seed = 98765
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        bases1 = generate_random_bases(200, rng=rng1)
        bases2 = generate_random_bases(200, rng=rng2)

        np.testing.assert_array_equal(bases1, bases2)

        # Independent seed should produce distinct sequence
        rng_other = np.random.default_rng(seed + 1)
        bases_other = generate_random_bases(200, rng=rng_other)
        self.assertFalse(np.array_equal(bases1, bases_other))

    def test_case_03_basis_selection_supports_both_z_and_x(self):
        """Case 3: Verify Bob's basis selection supports both Z and X and only valid bases."""
        n = 1000
        bases = generate_random_bases(n, rng=np.random.default_rng(123))
        unique_bases = set(bases)

        self.assertEqual(unique_bases, {BASIS_Z, BASIS_X})
        self.assertTrue(all(b in VALID_BASES for b in bases))

        # Uniform distribution check: proportions close to 0.5
        z_frac = np.mean(bases == BASIS_Z)
        x_frac = np.mean(bases == BASIS_X)
        self.assertAlmostEqual(z_frac, 0.5, delta=0.06)
        self.assertAlmostEqual(x_frac, 0.5, delta=0.06)


class TestBobQuantumMeasurement(unittest.TestCase):
    """Test suite for Bob's quantum measurement mechanics in conjugate bases."""

    def setUp(self):
        self.sim = AerSimulator()

    def test_case_04_z_basis_measurement_state_zero(self):
        """Case 4: Z-basis measurement of |0> produces 0 deterministically."""
        qc = prepare_bb84_state(bit=0, basis=BASIS_Z)
        for seed in [10, 20, 30]:
            outcome = measure_pulse(qc, basis=BASIS_Z, backend=self.sim, seed_simulator=seed)
            self.assertEqual(outcome, 0)

    def test_case_05_z_basis_measurement_state_one(self):
        """Case 5: Z-basis measurement of |1> produces 1 deterministically."""
        qc = prepare_bb84_state(bit=1, basis=BASIS_Z)
        for seed in [10, 20, 30]:
            outcome = measure_pulse(qc, basis=BASIS_Z, backend=self.sim, seed_simulator=seed)
            self.assertEqual(outcome, 1)

    def test_case_06_x_basis_measurement_state_plus(self):
        """Case 6: X-basis measurement of |+> produces 0 with probability 1.

        Mathematical proof: H|+> = H(H|0>) = |0> -> computational measurement yields 0.
        """
        qc = prepare_bb84_state(bit=0, basis=BASIS_X)
        # Test across multiple shots / random seeds
        for seed in range(100, 110):
            outcome = measure_pulse(qc, basis=BASIS_X, backend=self.sim, seed_simulator=seed)
            self.assertEqual(outcome, 0)

    def test_case_07_x_basis_measurement_state_minus(self):
        """Case 7: X-basis measurement of |-> produces 1 with probability 1.

        Mathematical proof: H|-> = H(X H|0> ... ) = H(H X|0>) = X|0> = |1> -> yields 1.
        """
        qc = prepare_bb84_state(bit=1, basis=BASIS_X)
        for seed in range(200, 210):
            outcome = measure_pulse(qc, basis=BASIS_X, backend=self.sim, seed_simulator=seed)
            self.assertEqual(outcome, 1)

    def test_case_08_measurement_preserves_signal_ordering(self):
        """Case 8: Measurement results preserve the original signal ordering."""
        # Prepare an alternating sequence of distinct states:
        # Index 0: |0> (Z) -> 0
        # Index 1: |1> (Z) -> 1
        # Index 2: |+> (X) -> 0
        # Index 3: |-> (X) -> 1
        # Index 4: |1> (Z) -> 1
        # Index 5: |-> (X) -> 1
        # Index 6: |0> (Z) -> 0
        # Index 7: |+> (X) -> 0
        bits = [0, 1, 0, 1, 1, 1, 0, 0]
        bases = [BASIS_Z, BASIS_Z, BASIS_X, BASIS_X, BASIS_Z, BASIS_X, BASIS_Z, BASIS_X]
        circuits = prepare_bb84_circuits(bits, bases)

        # Bob measures in the matching bases
        measured_bits = measure_bb84_circuits(
            circuits,
            bases,
            backend=self.sim,
            seed_simulator=42,
        )

        self.assertEqual(len(measured_bits), len(bits))
        np.testing.assert_array_equal(measured_bits, np.array(bits, dtype=np.int8))


class TestBasisReconciliation(unittest.TestCase):
    """Test suite for classical basis reconciliation (sifting)."""

    def test_case_09_reconciliation_retains_only_matching_bases(self):
        """Case 9: Basis reconciliation retains only matching bases."""
        alice_bases = ["Z", "X", "Z", "X", "Z", "X"]
        bob_bases   = ["Z", "Z", "Z", "X", "X", "Z"]
        # Matching at indices: 0 (Z==Z), 2 (Z==Z), 3 (X==X) -> Total 3 matching

        alice_bits = [0, 1, 1, 0, 1, 0]
        bob_bits   = [0, 0, 1, 0, 0, 1]

        res = reconcile_bases(alice_bases, bob_bases, alice_bits, bob_bits)

        self.assertIsInstance(res, SiftingResult)
        np.testing.assert_array_equal(res.sifted_indices, np.array([0, 2, 3], dtype=np.int64))
        self.assertEqual(res.matching_bases_count, 3)
        self.assertEqual(res.total_signals, 6)
        self.assertAlmostEqual(res.sifting_efficiency, 3 / 6)

        # Sifted bits should correspond strictly to indices [0, 2, 3]
        np.testing.assert_array_equal(res.alice_sifted_bits, np.array([0, 1, 0], dtype=np.int8))
        np.testing.assert_array_equal(res.bob_sifted_bits, np.array([0, 1, 0], dtype=np.int8))

    def test_case_10_sifted_alice_and_bob_keys_have_matching_lengths(self):
        """Case 10: Sifted Alice and Bob keys have matching lengths."""
        rng = np.random.default_rng(777)
        n = 500
        alice_bases = rng.choice([BASIS_Z, BASIS_X], size=n)
        bob_bases = rng.choice([BASIS_Z, BASIS_X], size=n)
        alice_bits = rng.integers(0, 2, size=n, dtype=np.int8)
        bob_bits = rng.integers(0, 2, size=n, dtype=np.int8)

        res = reconcile_bases(alice_bases, bob_bases, alice_bits, bob_bits)

        self.assertEqual(len(res.alice_sifted_bits), len(res.bob_sifted_bits))
        self.assertEqual(len(res.alice_sifted_bits), res.matching_bases_count)
        self.assertEqual(len(res.sifted_indices), res.matching_bases_count)


class TestQBERCalculation(unittest.TestCase):
    """Test suite for Quantum Bit Error Rate (QBER) calculation."""

    def test_case_11_qber_is_zero_when_all_sifted_bits_match(self):
        """Case 11: QBER is zero when all sifted bits match."""
        alice_bits = [0, 1, 1, 0, 1, 0, 0, 1]
        bob_bits   = [0, 1, 1, 0, 1, 0, 0, 1]

        res = compute_qber(alice_bits, bob_bits)

        self.assertIsInstance(res, QBERResult)
        self.assertEqual(res.qber, 0.0)
        self.assertEqual(res.error_count, 0)
        self.assertEqual(res.total_bits, 8)

    def test_case_12_qber_is_025_for_one_error_among_four_sifted_bits(self):
        """Case 12: QBER is 0.25 for one error among four sifted bits."""
        alice_bits = [0, 1, 0, 1]
        bob_bits   = [0, 1, 1, 1]  # Error at index 2 (0 != 1)

        res = compute_qber(alice_bits, bob_bits)

        self.assertEqual(res.qber, 0.25)
        self.assertEqual(res.error_count, 1)
        self.assertEqual(res.total_bits, 4)

    def test_case_13_empty_sifted_keys_are_handled_correctly(self):
        """Case 13: Empty sifted keys are handled correctly without ZeroDivisionError."""
        res = compute_qber([], [])

        self.assertIsInstance(res, QBERResult)
        self.assertEqual(res.qber, 0.0)
        self.assertEqual(res.error_count, 0)
        self.assertEqual(res.total_bits, 0)

        # Empty sifting result check
        sift_res = reconcile_bases([], [])
        self.assertEqual(sift_res.total_signals, 0)
        self.assertEqual(sift_res.matching_bases_count, 0)
        self.assertEqual(sift_res.sifting_efficiency, 0.0)
        self.assertEqual(len(sift_res.alice_sifted_bits), 0)
        self.assertEqual(len(sift_res.bob_sifted_bits), 0)


class TestValidationAndErrorHandling(unittest.TestCase):
    """Test suite for input validation and error handling."""

    def test_case_14_invalid_input_values_raise_meaningful_errors(self):
        """Case 14: Invalid input values raise meaningful errors."""
        qc = QuantumCircuit(1)

        # Invalid basis in apply_measurement_basis
        with self.assertRaises(ValueError) as ctx:
            apply_measurement_basis(qc, basis="Y")
        self.assertIn("Invalid basis", str(ctx.exception))

        # Invalid circuit type in apply_measurement_basis
        with self.assertRaises(TypeError) as ctx:
            apply_measurement_basis("not_a_circuit", basis="Z")
        self.assertIn("Expected circuit to be a QuantumCircuit", str(ctx.exception))

        # Invalid multi-qubit circuit in apply_measurement_basis
        qc_2q = QuantumCircuit(2)
        with self.assertRaises(ValueError) as ctx:
            apply_measurement_basis(qc_2q, basis="Z")
        self.assertIn("must have exactly 1 qubit", str(ctx.exception))

        # Invalid pulse count n in generate_random_bases
        with self.assertRaises(ValueError):
            generate_random_bases(0)
        with self.assertRaises(ValueError):
            generate_random_bases(-5)

        # Invalid bit value in compute_qber
        with self.assertRaises(ValueError) as ctx:
            compute_qber([0, 2], [0, 1])
        self.assertIn("Invalid bit value", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            compute_qber([0, 1], [0, -1])
        self.assertIn("Invalid bit value", str(ctx.exception))

        # Invalid basis in reconcile_bases
        with self.assertRaises(ValueError) as ctx:
            reconcile_bases(["Z", "INVALID"], ["Z", "X"])
        self.assertIn("Invalid Alice basis", str(ctx.exception))

    def test_case_15_mismatched_input_lengths_are_rejected(self):
        """Case 15: Mismatched input lengths are rejected with ValueError."""
        # Circuits vs bases length mismatch
        qc = QuantumCircuit(1)
        with self.assertRaises(ValueError) as ctx:
            measure_bb84_circuits([qc, qc], bases=["Z"])
        self.assertIn("Length mismatch", str(ctx.exception))

        # Alice bases vs Bob bases length mismatch in reconcile_bases
        with self.assertRaises(ValueError) as ctx:
            reconcile_bases(["Z", "X"], ["Z"])
        self.assertIn("Length mismatch", str(ctx.exception))

        # Bases vs bits length mismatch in reconcile_bases
        with self.assertRaises(ValueError) as ctx:
            reconcile_bases(["Z", "X"], ["Z", "X"], alice_bits=[0])
        self.assertIn("Length mismatch", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            reconcile_bases(["Z", "X"], ["Z", "X"], bob_bits=[0, 1, 0])
        self.assertIn("Length mismatch", str(ctx.exception))

        # Sifted keys length mismatch in compute_qber
        with self.assertRaises(ValueError) as ctx:
            compute_qber([0, 1, 0], [0, 1])
        self.assertIn("Length mismatch", str(ctx.exception))


class TestPhysicalValidation(unittest.TestCase):
    """Test suite for theoretical physical BB84 scenarios."""

    def test_scenario_a_ideal_bb84_simulation(self):
        """Scenario A: Ideal BB84 without noise or Eve.

        Alice prepares random bits and bases.
        Bob independently chooses random bases.
        Transmission through ideal channel (no perturbation, no Eve).
        Expectation:
        - Sifting efficiency is approximately 50% (+/- 4%).
        - Sifted QBER is exactly 0.0%.
        """
        n_pulses = 2000
        alice = Alice(seed=12345)
        bob = Bob(seed=67890)

        # Alice generates bits, bases, and prepares quantum circuits
        circuits = alice.prepare_pulses(n_pulses)

        # Bob independently generates bases and measures
        measured_bits = bob.measure_pulses(circuits)

        # Perform basis reconciliation
        sift_res = bob.sift(alice.bases, alice.bits)

        # Sifting efficiency should be close to 50%
        self.assertAlmostEqual(sift_res.sifting_efficiency, 0.50, delta=0.04)
        self.assertGreater(sift_res.matching_bases_count, 900)
        self.assertLess(sift_res.matching_bases_count, 1100)

        # Ideal channel without eavesdropper: QBER must be exactly 0.0%
        qber_res = bob.compute_qber(sift_res.alice_sifted_bits)
        self.assertEqual(qber_res.error_count, 0)
        self.assertEqual(qber_res.qber, 0.0)

    def test_scenario_b_x_basis_deterministic_correctness(self):
        """Scenario B: Deterministic X-basis correctness.

        Verify that:
        - |+> measured in X produces bit 0 with probability 1.
        - |-> measured in X produces bit 1 with probability 1.
        """
        n = 50
        # Batch of 50 |+> states
        plus_circuits = [prepare_bb84_state(bit=0, basis=BASIS_X) for _ in range(n)]
        plus_bases = [BASIS_X] * n
        plus_outcomes = measure_bb84_circuits(plus_circuits, plus_bases, seed_simulator=101)
        np.testing.assert_array_equal(plus_outcomes, np.zeros(n, dtype=np.int8))

        # Batch of 50 |-> states
        minus_circuits = [prepare_bb84_state(bit=1, basis=BASIS_X) for _ in range(n)]
        minus_bases = [BASIS_X] * n
        minus_outcomes = measure_bb84_circuits(minus_circuits, minus_bases, seed_simulator=202)
        np.testing.assert_array_equal(minus_outcomes, np.ones(n, dtype=np.int8))


class TestBobClassIntegration(unittest.TestCase):
    """Test suite for Bob agent class stateful lifecycle and API integration."""

    def test_bob_lifecycle_and_reset(self):
        """Verify Bob's complete stateful lifecycle."""
        bob = Bob(seed=42)
        self.assertIsNone(bob.bases)
        self.assertIsNone(bob.measured_bits)
        self.assertIsNone(bob.sifted_bits)

        # Calling sift or compute_qber before measuring should raise ValueError
        with self.assertRaises(ValueError):
            bob.sift(["Z", "X"])
        with self.assertRaises(ValueError):
            bob.compute_qber([0, 1])

        # Prepare 10 circuits
        circuits = [prepare_bb84_state(0, "Z") for _ in range(10)]
        bob.measure_pulses(circuits)

        self.assertEqual(len(bob.bases), 10)
        self.assertEqual(len(bob.measured_bits), 10)

        # Reset RNG
        bob.reset_rng(seed=999)
        new_bases = bob.generate_bases(10)
        self.assertEqual(len(new_bases), 10)


if __name__ == "__main__":
    unittest.main()

