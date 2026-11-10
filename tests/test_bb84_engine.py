"""Unit tests for BB84 Protocol Engine.

Validates Test Group A:
1. End-to-end ideal BB84 execution and result container integrity.
2. Zero QBER on noiseless unintercepted channel.
3. Approximately 50% sifting efficiency.
4. Deterministic reproducibility with fixed seeds.
5. Correct signal ordering preservation.
6. Correct sifted key lengths and sub-basis partitioning.
7. Invalid signal counts raise meaningful exceptions.
8. Invalid attack configurations raise meaningful exceptions.
9. Alice, Bob, and Eve data isolation (no private bit leakage).
"""

import unittest
import numpy as np
from qiskit_aer import AerSimulator

from src.bb84.alice import BASIS_X, BASIS_Z, VALID_BASES, VALID_BITS
from src.bb84.attacks import InterceptResendAttack, FuchsPeresBrandtProbe
from src.bb84.bb84_engine import BB84Engine, BB84SimulationResult


class TestBB84Engine(unittest.TestCase):
    """Test suite for BB84 Protocol Engine."""

    def setUp(self):
        self.engine = BB84Engine(alice_seed=12345, bob_seed=67890)

    def test_case_01_end_to_end_ideal_bb84(self):
        """Case 1: Full end-to-end execution of ideal BB84 produces valid result container."""
        n = 500
        res = self.engine.run(n_signals=n)

        self.assertIsInstance(res, BB84SimulationResult)
        self.assertEqual(res.num_signals, n)
        self.assertEqual(len(res.alice_bits), n)
        self.assertEqual(len(res.alice_bases), n)
        self.assertEqual(len(res.bob_bases), n)
        self.assertEqual(len(res.bob_measured_bits), n)
        self.assertFalse(res.attack_applied)
        self.assertEqual(res.attack_name, "None")

        # Check types and ranges
        self.assertTrue(set(res.alice_bits).issubset(VALID_BITS))
        self.assertTrue(set(res.bob_measured_bits).issubset(VALID_BITS))
        self.assertTrue(set(res.alice_bases).issubset(VALID_BASES))
        self.assertTrue(set(res.bob_bases).issubset(VALID_BASES))

    def test_case_02_zero_qber_without_eve(self):
        """Case 2: QBER is strictly 0.0% on noiseless unintercepted channel."""
        res = self.engine.run(n_signals=1500)

        self.assertEqual(res.error_count, 0)
        self.assertEqual(res.qber, 0.0)
        self.assertEqual(res.qber_z, 0.0)
        self.assertEqual(res.qber_x, 0.0)
        self.assertEqual(res.error_count_z, 0)
        self.assertEqual(res.error_count_x, 0)
        np.testing.assert_array_equal(res.alice_sifted_bits, res.bob_sifted_bits)

    def test_case_03_approximately_50_percent_sifting_efficiency(self):
        """Case 3: Sifting efficiency is close to theoretical 50% (+/- 4%)."""
        n = 3000
        res = self.engine.run(n_signals=n)

        self.assertAlmostEqual(res.sifting_efficiency, 0.50, delta=0.04)
        self.assertGreater(res.sifted_key_length, 1350)
        self.assertLess(res.sifted_key_length, 1650)

    def test_case_04_reproducibility_with_fixed_seeds(self):
        """Case 4: Fixed seeds guarantee identical simulation runs."""
        engine1 = BB84Engine(alice_seed=999, bob_seed=888)
        engine2 = BB84Engine(alice_seed=999, bob_seed=888)

        res1 = engine1.run(n_signals=200, seed_simulator=77)
        res2 = engine2.run(n_signals=200, seed_simulator=77)

        np.testing.assert_array_equal(res1.alice_bits, res2.alice_bits)
        np.testing.assert_array_equal(res1.alice_bases, res2.alice_bases)
        np.testing.assert_array_equal(res1.bob_bases, res2.bob_bases)
        np.testing.assert_array_equal(res1.bob_measured_bits, res2.bob_measured_bits)
        np.testing.assert_array_equal(res1.sifted_indices, res2.sifted_indices)
        self.assertEqual(res1.qber, res2.qber)

        # Different seeds produce different sequences
        engine3 = BB84Engine(alice_seed=111, bob_seed=222)
        res3 = engine3.run(n_signals=200, seed_simulator=77)
        self.assertFalse(np.array_equal(res1.alice_bits, res3.alice_bits))

    def test_case_05_correct_signal_ordering(self):
        """Case 5: Output sifted keys preserve original signal index ordering."""
        res = self.engine.run(n_signals=300)

        # Check sifted indices strictly increasing
        self.assertTrue(np.all(np.diff(res.sifted_indices) > 0))

        # Check that sifted bits are exact slices of raw bits at sifted_indices
        np.testing.assert_array_equal(
            res.alice_sifted_bits,
            res.alice_bits[res.sifted_indices],
        )
        np.testing.assert_array_equal(
            res.bob_sifted_bits,
            res.bob_measured_bits[res.sifted_indices],
        )

    def test_case_06_correct_sifted_key_lengths(self):
        """Case 6: Sifted key lengths match sub-basis partition lengths."""
        res = self.engine.run(n_signals=1000)

        total_sifted = res.sifted_key_length
        self.assertEqual(len(res.alice_sifted_bits), total_sifted)
        self.assertEqual(len(res.bob_sifted_bits), total_sifted)
        self.assertEqual(len(res.sifted_indices), total_sifted)
        self.assertEqual(res.sifted_count_z + res.sifted_count_x, total_sifted)
        self.assertEqual(res.error_count, res.error_count_z + res.error_count_x)

    def test_case_07_invalid_signal_counts_rejected(self):
        """Case 7: Invalid signal counts raise ValueError."""
        with self.assertRaises(ValueError):
            self.engine.run(n_signals=0)
        with self.assertRaises(ValueError):
            self.engine.run(n_signals=-50)
        with self.assertRaises(ValueError):
            self.engine.run(n_signals=10.5)  # non-integer

    def test_case_08_invalid_attack_configuration_rejected(self):
        """Case 8: Invalid attack configurations raise TypeError."""
        with self.assertRaises(TypeError):
            self.engine.run(n_signals=100, attack="not_an_attack")
        with self.assertRaises(TypeError):
            self.engine.run(n_signals=100, attack={"attack": "pir"})

    def test_case_09_alice_and_bob_data_isolation(self):
        """Case 9: Alice, Bob, and Eve are strictly isolated; no private bit leakage."""
        # Run engine with intercept-resend attack
        attack = InterceptResendAttack(alpha=0.5, beta=0.5, seed=42)
        res = self.engine.run(n_signals=1000, attack=attack)

        # Eve should only know her own bases and bits, not Alice's private key
        meta = res.attack_metadata
        self.assertIn("eve_bases", meta)
        self.assertIn("eve_bits", meta)
        self.assertNotIn("alice_bits", meta)
        self.assertNotIn("alice_bases", meta)

        # Bob should not have direct access to Alice's raw bits
        self.assertNotEqual(id(self.engine.bob.measured_bits), id(self.engine.alice.bits))


if __name__ == "__main__":
    unittest.main()

