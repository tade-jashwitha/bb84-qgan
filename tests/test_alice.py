"""Unit tests for Alice module in BB84 protocol.

Validates:
1. Random bit generation (binary values, statistical uniformity, reproducibility).
2. Random basis selection ('Z' and 'X' sets, uniformity, reproducibility).
3. Statevector verification for all four BB84 states (|0>, |1>, |+>, |->):
   - Numerical amplitudes match theoretical values to high precision.
   - Relative phase between |0> and |1> components matches theoretical expectations (0 for |+>, pi for |->).
   - Qiskit Statevector equivalence check.
4. Input validation and error handling.
5. Alice class integration and pulse batch preparation.
"""

import unittest
import numpy as np
from qiskit.quantum_info import Statevector

from src.bb84.alice import (
    Alice,
    generate_random_bits,
    generate_random_bases,
    prepare_bb84_state,
    prepare_bb84_circuits,
    BASIS_Z,
    BASIS_X,
)


class TestAliceRandomGeneration(unittest.TestCase):
    """Test suite for Alice's random bit and basis generation."""

    def test_random_bits_values_and_shape(self):
        """Verify generated bits are 1D array containing only 0 and 1."""
        n = 1000
        bits = generate_random_bits(n, rng=np.random.default_rng(42))
        
        self.assertEqual(bits.shape, (n,))
        self.assertTrue(set(bits).issubset({0, 1}))
        # Statistical check: mean should be close to 0.5 for n=1000
        mean_bit = np.mean(bits)
        self.assertAlmostEqual(mean_bit, 0.5, delta=0.06)

    def test_random_bases_values_and_shape(self):
        """Verify generated bases are 1D array containing only 'Z' and 'X'."""
        n = 1000
        bases = generate_random_bases(n, rng=np.random.default_rng(42))
        
        self.assertEqual(bases.shape, (n,))
        self.assertTrue(set(bases).issubset({BASIS_Z, BASIS_X}))
        # Statistical check: proportion of Z should be close to 0.5
        z_prop = np.mean(bases == BASIS_Z)
        self.assertAlmostEqual(z_prop, 0.5, delta=0.06)

    def test_reproducibility_with_seed(self):
        """Verify identical outputs when using fixed RNG seeds."""
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)
        
        bits1 = generate_random_bits(100, rng1)
        bits2 = generate_random_bits(100, rng2)
        np.testing.assert_array_equal(bits1, bits2)

        rng3 = np.random.default_rng(12345)
        rng4 = np.random.default_rng(12345)
        bases1 = generate_random_bases(100, rng3)
        bases2 = generate_random_bases(100, rng4)
        np.testing.assert_array_equal(bases1, bases2)

    def test_invalid_input_n(self):
        """Verify ValueError raised for invalid pulse counts."""
        with self.assertRaises(ValueError):
            generate_random_bits(0)
        with self.assertRaises(ValueError):
            generate_random_bits(-10)
        with self.assertRaises(ValueError):
            generate_random_bases(0)
        with self.assertRaises(ValueError):
            generate_random_bases(-5)


class TestAliceBB84StatePreparation(unittest.TestCase):
    """Test suite for BB84 quantum state preparation using Qiskit Statevector."""

    def setUp(self):
        """Define analytical theoretical statevectors for comparison."""
        self.inv_sqrt2 = 1.0 / np.sqrt(2.0)
        
        # Theoretical statevectors [c_0, c_1]^T
        self.theo_state_0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
        self.theo_state_1 = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=complex)
        self.theo_state_plus = np.array([self.inv_sqrt2 + 0.0j, self.inv_sqrt2 + 0.0j], dtype=complex)
        self.theo_state_minus = np.array([self.inv_sqrt2 + 0.0j, -self.inv_sqrt2 + 0.0j], dtype=complex)

    def test_state_zero_computational_basis(self):
        """State |0>: bit=0, basis='Z'.
        Theoretical state: [1, 0]^T.
        """
        qc = prepare_bb84_state(bit=0, basis="Z")
        sv = Statevector(qc)
        
        # 1. Exact numerical amplitude match
        np.testing.assert_allclose(sv.data, self.theo_state_0, atol=1e-12)
        # 2. Qiskit Statevector equivalence
        self.assertTrue(sv.equiv(Statevector(self.theo_state_0)))
        # 3. Measurement probabilities
        probs = sv.probabilities_dict()
        self.assertAlmostEqual(probs.get("0", 0.0), 1.0, places=12)
        self.assertAlmostEqual(probs.get("1", 0.0), 0.0, places=12)

    def test_state_one_computational_basis(self):
        """State |1>: bit=1, basis='Z'.
        Theoretical state: [0, 1]^T.
        """
        qc = prepare_bb84_state(bit=1, basis="Z")
        sv = Statevector(qc)
        
        # 1. Exact numerical amplitude match
        np.testing.assert_allclose(sv.data, self.theo_state_1, atol=1e-12)
        # 2. Qiskit Statevector equivalence
        self.assertTrue(sv.equiv(Statevector(self.theo_state_1)))
        # 3. Measurement probabilities
        probs = sv.probabilities_dict()
        self.assertAlmostEqual(probs.get("0", 0.0), 0.0, places=12)
        self.assertAlmostEqual(probs.get("1", 0.0), 1.0, places=12)

    def test_state_plus_hadamard_basis(self):
        """State |+>: bit=0, basis='X'.
        Theoretical state: [1/sqrt(2), 1/sqrt(2)]^T.
        Relative phase between |0> and |1> amplitudes must be 0 (in-phase).
        """
        qc = prepare_bb84_state(bit=0, basis="X")
        sv = Statevector(qc)
        
        # 1. Exact numerical amplitude match
        np.testing.assert_allclose(sv.data, self.theo_state_plus, atol=1e-12)
        # 2. Qiskit Statevector equivalence
        self.assertTrue(sv.equiv(Statevector(self.theo_state_plus)))
        # 3. Verify relative phase: arg(c_1 / c_0) == 0.0
        c0, c1 = sv.data[0], sv.data[1]
        relative_phase = np.angle(c1 / c0)
        self.assertAlmostEqual(relative_phase, 0.0, places=12)
        # 4. Measurement probabilities: 50% in Z-basis
        probs = sv.probabilities_dict()
        self.assertAlmostEqual(probs.get("0", 0.0), 0.5, places=12)
        self.assertAlmostEqual(probs.get("1", 0.0), 0.5, places=12)

    def test_state_minus_hadamard_basis(self):
        """State |->: bit=1, basis='X'.
        Theoretical state: [1/sqrt(2), -1/sqrt(2)]^T.
        Relative phase between |0> and |1> amplitudes must be pi (anti-phase).
        """
        qc = prepare_bb84_state(bit=1, basis="X")
        sv = Statevector(qc)
        
        # 1. Exact numerical amplitude match
        np.testing.assert_allclose(sv.data, self.theo_state_minus, atol=1e-12)
        # 2. Qiskit Statevector equivalence
        self.assertTrue(sv.equiv(Statevector(self.theo_state_minus)))
        # 3. Verify relative phase: arg(c_1 / c_0) == +pi or -pi
        c0, c1 = sv.data[0], sv.data[1]
        relative_phase = np.angle(c1 / c0)
        self.assertAlmostEqual(abs(relative_phase), np.pi, places=12)
        # 4. Measurement probabilities: 50% in Z-basis
        probs = sv.probabilities_dict()
        self.assertAlmostEqual(probs.get("0", 0.0), 0.5, places=12)
        self.assertAlmostEqual(probs.get("1", 0.0), 0.5, places=12)

    def test_orthogonality_within_bases(self):
        """Verify that states within the same basis are mutually orthogonal:
        <0|1> = 0 and <+|- > = 0.
        """
        sv_0 = Statevector(prepare_bb84_state(0, "Z"))
        sv_1 = Statevector(prepare_bb84_state(1, "Z"))
        inner_z = np.vdot(sv_0.data, sv_1.data)
        self.assertAlmostEqual(abs(inner_z), 0.0, places=12)

        sv_plus = Statevector(prepare_bb84_state(0, "X"))
        sv_minus = Statevector(prepare_bb84_state(1, "X"))
        inner_x = np.vdot(sv_plus.data, sv_minus.data)
        self.assertAlmostEqual(abs(inner_x), 0.0, places=12)

    def test_mutually_unbiased_overlap(self):
        """Verify that Z and X bases are mutually unbiased:
        |<z|x>|^2 = 1/2 for all z in {0,1} and x in {+,-}.
        """
        sv_0 = Statevector(prepare_bb84_state(0, "Z"))
        sv_1 = Statevector(prepare_bb84_state(1, "Z"))
        sv_plus = Statevector(prepare_bb84_state(0, "X"))
        sv_minus = Statevector(prepare_bb84_state(1, "X"))

        for z in [sv_0, sv_1]:
            for x in [sv_plus, sv_minus]:
                overlap_squared = abs(np.vdot(z.data, x.data)) ** 2
                self.assertAlmostEqual(overlap_squared, 0.5, places=12)

    def test_invalid_state_preparation_arguments(self):
        """Verify proper exceptions on invalid bit or basis inputs."""
        with self.assertRaises(ValueError):
            prepare_bb84_state(bit=2, basis="Z")
        with self.assertRaises(ValueError):
            prepare_bb84_state(bit=-1, basis="Z")
        with self.assertRaises(ValueError):
            prepare_bb84_state(bit=0, basis="Y")
        with self.assertRaises(ValueError):
            prepare_bb84_state(bit=1, basis="invalid")


class TestAliceClassAndBatch(unittest.TestCase):
    """Test suite for Alice class and batch pulse preparation."""

    def test_batch_pulse_preparation(self):
        """Verify batch circuit preparation from matching bit/basis arrays."""
        bits = [0, 1, 0, 1]
        bases = ["Z", "Z", "X", "X"]
        circuits = prepare_bb84_circuits(bits, bases)
        
        self.assertEqual(len(circuits), 4)
        expected_states = [
            np.array([1.0, 0.0], dtype=complex),
            np.array([0.0, 1.0], dtype=complex),
            np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=complex),
            np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2)], dtype=complex),
        ]
        
        for qc, exp in zip(circuits, expected_states):
            sv = Statevector(qc)
            np.testing.assert_allclose(sv.data, exp, atol=1e-12)

    def test_batch_length_mismatch(self):
        """Verify ValueError raised when bits and bases lengths differ."""
        with self.assertRaises(ValueError):
            prepare_bb84_circuits([0, 1], ["Z"])

    def test_alice_class_lifecycle(self):
        """Verify Alice class initialization, generation, and circuit preparation."""
        alice = Alice(seed=999)
        circuits = alice.prepare_pulses(n=50)
        
        self.assertEqual(len(circuits), 50)
        self.assertIsNotNone(alice.bits)
        self.assertIsNotNone(alice.bases)
        self.assertEqual(len(alice.bits), 50)
        self.assertEqual(len(alice.bases), 50)

        # Reproducibility check with another Alice instance
        alice2 = Alice(seed=999)
        alice2.prepare_pulses(n=50)
        np.testing.assert_array_equal(alice.bits, alice2.bits)
        np.testing.assert_array_equal(alice.bases, alice2.bases)


if __name__ == "__main__":
    unittest.main()
