"""Unit tests for Physical Eavesdropping Attacks in BB84.

Validates:
Test Group B (Probabilistic Intercept-Resend Attack):
1. Alpha = 0 produces zero interceptions.
2. Alpha = 1 intercepts every signal.
3. Beta = 0 selects strictly X basis.
4. Beta = 1 selects strictly Z basis.
5. Beta = 0.5 yields approximately uniform basis distribution.
6. Full intercept-resend (alpha=1.0) produces approximately 25% QBER.
7. Partial intercept-resend follows alpha / 4 asymptotically.
8. Basis-conditioned QBERs match theoretical formulas:
   QBER_Z = 0.5 * alpha * (1 - beta), QBER_X = 0.5 * alpha * beta.
9. Replacement states maintain valid statevector normalization.
10. Intercept-Resend results are deterministic and reproducible with fixed seeds.

Test Group C (Fuchs-Peres-Brandt Unitary Probe):
1. Unitary matrix dimensions (4x4 complex matrix).
2. Unitarity condition U^\dagger U = I_4 verified numerically.
3. Parameter boundaries validation (0 <= theta_E <= pi/2).
4. Joint output statevector normalization ||psi_joint|| = 1.0.
5. Bob's and Eve's reduced density matrices have trace = 1.0.
6. Reduced density matrices are positive semidefinite (eigenvalues >= 0).
7. Limiting case theta_E = 0 produces identity (zero disturbance, zero info gain).
8. Theoretical disturbance D(theta_E) and mutual information I_AE(theta_E)
   match analytical derivations.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix

from src.bb84.alice import (
    Alice,
    BASIS_X,
    BASIS_Z,
    VALID_BASES,
    prepare_bb84_state,
    prepare_bb84_circuits,
)
from src.bb84.bob import Bob, reconcile_bases, compute_qber
from src.bb84.attacks import (
    InterceptResendAttack,
    InterceptResendResult,
    FuchsPeresBrandtProbe,
    UnitaryProbeResult,
    build_probe_unitary,
    binary_entropy,
)
from src.bb84.bb84_engine import BB84Engine


class TestInterceptResendAttack(unittest.TestCase):
    """Test suite for Probabilistic Intercept-Resend attack (Test Group B)."""

    def setUp(self):
        self.circuits_100 = [prepare_bb84_state(0, BASIS_Z) for _ in range(100)]

    def test_case_01_alpha_zero_produces_no_interceptions(self):
        """Case 1: Alpha = 0 intercepts 0 pulses and leaves circuits unchanged."""
        attack = InterceptResendAttack(alpha=0.0, beta=0.5, seed=42)
        res = attack.intercept_pulses(self.circuits_100)

        self.assertEqual(res.intercept_count, 0)
        self.assertEqual(res.intercept_ratio, 0.0)
        self.assertFalse(np.any(res.intercepted_flags))
        self.assertTrue(np.all(res.eve_bits == -1))
        self.assertEqual(len(res.transmitted_circuits), 100)

    def test_case_02_alpha_one_intercepts_every_signal(self):
        """Case 2: Alpha = 1 intercepts 100% of signals."""
        attack = InterceptResendAttack(alpha=1.0, beta=0.5, seed=42)
        res = attack.intercept_pulses(self.circuits_100)

        self.assertEqual(res.intercept_count, 100)
        self.assertEqual(res.intercept_ratio, 1.0)
        self.assertTrue(np.all(res.intercepted_flags))
        self.assertTrue(np.all(res.eve_bits != -1))

    def test_case_03_beta_zero_selects_only_x(self):
        """Case 3: Beta = 0 selects strictly X basis for all intercepted pulses."""
        attack = InterceptResendAttack(alpha=1.0, beta=0.0, seed=123)
        res = attack.intercept_pulses(self.circuits_100)

        self.assertTrue(np.all(res.eve_bases == BASIS_X))
        self.assertFalse(np.any(res.eve_bases == BASIS_Z))

    def test_case_04_beta_one_selects_only_z(self):
        """Case 4: Beta = 1 selects strictly Z basis for all intercepted pulses."""
        attack = InterceptResendAttack(alpha=1.0, beta=1.0, seed=123)
        res = attack.intercept_pulses(self.circuits_100)

        self.assertTrue(np.all(res.eve_bases == BASIS_Z))
        self.assertFalse(np.any(res.eve_bases == BASIS_X))

    def test_case_05_beta_half_gives_approximately_uniform_basis(self):
        """Case 5: Beta = 0.5 gives approximately uniform basis selection."""
        n = 2000
        circuits = [prepare_bb84_state(0, BASIS_Z) for _ in range(n)]
        attack = InterceptResendAttack(alpha=1.0, beta=0.5, seed=456)
        res = attack.intercept_pulses(circuits)

        z_frac = np.mean(res.eve_bases == BASIS_Z)
        x_frac = np.mean(res.eve_bases == BASIS_X)
        self.assertAlmostEqual(z_frac, 0.50, delta=0.04)
        self.assertAlmostEqual(x_frac, 0.50, delta=0.04)

    def test_case_06_full_intercept_resend_produces_approximately_25_percent_qber(self):
        """Case 6: Full intercept-resend (alpha=1.0, beta=0.5) produces ~25% QBER."""
        engine = BB84Engine(alice_seed=101, bob_seed=202)
        attack = InterceptResendAttack(alpha=1.0, beta=0.5, seed=303)

        n = 5000
        res = engine.run(n_signals=n, attack=attack)

        # Expected QBER = 25.0% (+/- 2.5% for sample of ~2500 sifted bits)
        self.assertAlmostEqual(res.qber, 0.25, delta=0.025)
        self.assertAlmostEqual(res.qber_z, 0.25, delta=0.035)
        self.assertAlmostEqual(res.qber_x, 0.25, delta=0.035)

    def test_case_07_partial_intercept_resend_follows_alpha_over_4(self):
        """Case 7: Partial intercept-resend follows QBER = alpha / 4."""
        engine = BB84Engine(alice_seed=111, bob_seed=222)
        for alpha in [0.2, 0.4, 0.6, 0.8]:
            attack = InterceptResendAttack(alpha=alpha, beta=0.5, seed=444)
            res = engine.run(n_signals=6000, attack=attack)

            expected_qber = alpha / 4.0
            self.assertAlmostEqual(res.qber, expected_qber, delta=0.025)

    def test_case_08_basis_conditioned_qber_calculations(self):
        """Case 8: Biased beta shifts QBER between Z and X according to theory."""
        # Theoretical: QBER_Z = 0.5 * alpha * (1 - beta), QBER_X = 0.5 * alpha * beta
        engine = BB84Engine(alice_seed=555, bob_seed=666)

        # Case A: beta = 1.0 (Eve only measures Z)
        # Expected: QBER_Z = 0.0%, QBER_X = 50.0%, total QBER = 25.0%
        attack_z = InterceptResendAttack(alpha=1.0, beta=1.0, seed=777)
        res_z = engine.run(n_signals=4000, attack=attack_z, seed_simulator=101)

        self.assertAlmostEqual(res_z.qber_z, 0.0, delta=0.02)
        self.assertAlmostEqual(res_z.qber_x, 0.50, delta=0.04)
        self.assertAlmostEqual(res_z.qber, 0.25, delta=0.025)

        # Case B: beta = 0.0 (Eve only measures X)
        # Expected: QBER_Z = 50.0%, QBER_X = 0.0%, total QBER = 25.0%
        engine.reset_seeds(alice_seed=555, bob_seed=666)
        attack_x = InterceptResendAttack(alpha=1.0, beta=0.0, seed=888)
        res_x = engine.run(n_signals=4000, attack=attack_x, seed_simulator=102)

        self.assertAlmostEqual(res_x.qber_z, 0.50, delta=0.04)
        self.assertAlmostEqual(res_x.qber_x, 0.0, delta=0.02)
        self.assertAlmostEqual(res_x.qber, 0.25, delta=0.025)

    def test_case_09_state_normalization(self):
        """Case 9: Replacement BB84 circuits maintain unit statevector norm."""
        attack = InterceptResendAttack(alpha=1.0, beta=0.5, seed=999)
        res = attack.intercept_pulses(self.circuits_100)

        for qc in res.transmitted_circuits:
            sv = Statevector.from_instruction(qc)
            norm = np.linalg.norm(sv.data)
            self.assertAlmostEqual(norm, 1.0, places=7)

    def test_case_10_reproducibility(self):
        """Case 10: Fixed seeds produce bit-identical attack records."""
        attack1 = InterceptResendAttack(alpha=0.6, beta=0.4, seed=54321)
        attack2 = InterceptResendAttack(alpha=0.6, beta=0.4, seed=54321)

        res1 = attack1.intercept_pulses(self.circuits_100)
        res2 = attack2.intercept_pulses(self.circuits_100)

        np.testing.assert_array_equal(res1.intercepted_flags, res2.intercepted_flags)
        np.testing.assert_array_equal(res1.eve_bases, res2.eve_bases)
        np.testing.assert_array_equal(res1.eve_bits, res2.eve_bits)


class TestFuchsPeresBrandtProbe(unittest.TestCase):
    """Test suite for Fuchs-Peres-Brandt entangling unitary probe (Test Group C)."""

    def test_case_01_unitary_matrix_dimensions(self):
        """Case 1: Unitary matrix has exact shape 4x4."""
        for theta in [0.0, np.pi / 6, np.pi / 4, np.pi / 2]:
            probe = FuchsPeresBrandtProbe(theta_E=theta)
            self.assertEqual(probe.unitary_matrix.shape, (4, 4))
            self.assertEqual(probe.unitary_matrix.dtype, np.complex128)

    def test_case_02_unitarity_condition(self):
        """Case 2: U^\dagger U = I_4 holds to high precision across parameter range."""
        thetas = [0.0, 0.1, np.pi / 8, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
        for theta in thetas:
            U = build_probe_unitary(theta)
            prod = U.conj().T @ U
            identity = np.eye(4, dtype=np.complex128)
            np.testing.assert_allclose(prod, identity, atol=1e-12)

    def test_case_03_valid_parameter_boundaries(self):
        """Case 3: Parameter boundaries [0, pi/2] are validated."""
        # Valid boundaries should not raise
        FuchsPeresBrandtProbe(theta_E=0.0)
        FuchsPeresBrandtProbe(theta_E=np.pi / 2.0)

        # Invalid angles raise ValueError
        with self.assertRaises(ValueError):
            FuchsPeresBrandtProbe(theta_E=-0.01)
        with self.assertRaises(ValueError):
            FuchsPeresBrandtProbe(theta_E=np.pi / 2.0 + 0.1)

    def test_case_04_joint_state_normalization(self):
        """Case 4: Joint output statevector is strictly normalized (||psi|| = 1.0)."""
        thetas = [0.0, np.pi / 6, np.pi / 4]
        carrier_states = [
            [1.0, 0.0],                          # |0>
            [0.0, 1.0],                          # |1>
            [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],  # |+>
            [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)], # |->
        ]
        for theta in thetas:
            probe = FuchsPeresBrandtProbe(theta_E=theta)
            for state in carrier_states:
                _, _, psi_joint = probe.compute_reduced_states(state)
                norm = np.linalg.norm(psi_joint.data)
                self.assertAlmostEqual(norm, 1.0, places=12)

    def test_case_05_reduced_density_matrix_trace_equals_one(self):
        """Case 5: Partial traces Tr_E and Tr_A yield unit-trace density matrices."""
        probe = FuchsPeresBrandtProbe(theta_E=np.pi / 6)
        carrier_states = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],
            [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)],
        ]
        for state in carrier_states:
            rho_A, rho_E, _ = probe.compute_reduced_states(state)
            self.assertAlmostEqual(np.trace(rho_A.data), 1.0 + 0.0j, places=12)
            self.assertAlmostEqual(np.trace(rho_E.data), 1.0 + 0.0j, places=12)

    def test_case_06_positive_semidefinite_density_matrix(self):
        """Case 6: Reduced density matrices are Hermitian and positive semidefinite."""
        probe = FuchsPeresBrandtProbe(theta_E=np.pi / 5)
        carrier_states = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],
            [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)],
        ]
        for state in carrier_states:
            rho_A, rho_E, _ = probe.compute_reduced_states(state)

            # Hermiticity
            np.testing.assert_allclose(rho_A.data, rho_A.data.conj().T, atol=1e-12)
            np.testing.assert_allclose(rho_E.data, rho_E.data.conj().T, atol=1e-12)

            # Eigenvalues >= 0
            eigs_A = np.linalg.eigvalsh(rho_A.data)
            eigs_E = np.linalg.eigvalsh(rho_E.data)
            self.assertTrue(np.all(eigs_A >= -1e-12))
            self.assertTrue(np.all(eigs_E >= -1e-12))

    def test_case_07_limiting_cases(self):
        """Case 7: At theta_E = 0, probe is identity with 0 disturbance and 0 information."""
        probe_zero = FuchsPeresBrandtProbe(theta_E=0.0)

        # Matrix is identity
        np.testing.assert_allclose(probe_zero.unitary_matrix, np.eye(4), atol=1e-12)
        self.assertEqual(probe_zero.theoretical_disturbance(), 0.0)
        self.assertEqual(probe_zero.theoretical_information_gain(), 0.0)

        # No disturbance on states
        rho_A, _, _ = probe_zero.compute_reduced_states([1.0, 0.0])
        np.testing.assert_allclose(
            rho_A.data,
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            atol=1e-12,
        )

    def test_case_08_comparison_against_analytical_derivations(self):
        """Case 8: Compare D(theta_E) and I_AE(theta_E) against analytical formulas."""
        # Test across distinct angles (Audited: D(theta_E) = sin^2(theta_E))
        test_points = [
            (0.0, 0.0, 0.0),
            (np.pi / 12, float(np.sin(np.pi / 12) ** 2), 1.0 - binary_entropy(0.75)),
            (np.pi / 4, 0.50, 1.0),
        ]
        for theta, exp_d, exp_i in test_points:
            probe = FuchsPeresBrandtProbe(theta_E=theta)
            self.assertAlmostEqual(probe.theoretical_disturbance(), exp_d, places=6)
            self.assertAlmostEqual(probe.theoretical_information_gain(), exp_i, places=6)


if __name__ == "__main__":
    unittest.main()

