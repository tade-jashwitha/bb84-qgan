"""Unit tests for NISQ Noise Models in BB84 QKD.

Validates:
1. Parameter validation: Acceptance of valid parameters and rejection of invalid ones:
   - p, gamma, lambda outside [0, 1] raise ValueError.
   - T1 <= 0, T2 <= 0, t < 0 raise ValueError.
   - T2 > 2 * T1 physical violation raises ValueError.
2. Mathematical correctness of Kraus operators:
   - Kraus completeness sum K_i^dagger K_i = I_2 to high precision (< 1e-12).
   - Trace preservation: Tr(E(rho)) = 1 for all BB84 states (|0>, |1>, |+>, |->).
   - Complete positivity: E(rho) is Hermitian and positive semi-definite (eigenvalues >= -1e-12).
3. Limiting cases:
   - Zero noise: E(rho) = rho, K_0 = I, all other K_i = 0.
   - Maximal depolarizing (p = 1): E(rho) = I / 2 (completely mixed state).
   - Maximal amplitude damping (gamma = 1): E(|1><1|) = |0><0| (full relaxation to ground state).
   - Maximal phase damping (lambda = 1): complete dephasing, off-diagonals vanish.
4. Analytical QBER predictors:
   - Matches density matrix measurement expectation values across conjugate bases.
5. Qiskit Aer NoiseModel builders:
   - Noise models successfully construct and produce valid NoiseChannelProfile containers.
"""

import unittest
import numpy as np

from src.channel.noise_models import (
    NoiseType,
    NoiseChannelProfile,
    get_depolarizing_kraus,
    get_amplitude_damping_kraus,
    get_phase_damping_kraus,
    get_thermal_relaxation_kraus,
    verify_kraus_completeness,
    apply_kraus_channel,
    verify_density_matrix,
    compute_state_fidelity,
    compute_coherence,
    compute_channel_fidelity,
    compute_basis_qber_from_density_matrices,
    predict_depolarizing_qber,
    predict_amplitude_damping_qber,
    predict_phase_damping_qber,
    predict_thermal_relaxation_qber,
    validate_thermal_parameters,
    build_depolarizing_noise_model,
    build_amplitude_damping_noise_model,
    build_phase_damping_noise_model,
    build_thermal_relaxation_noise_model,
    build_combined_noise_model,
)

# Canonical BB84 pure state density matrices
RHO_0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
RHO_1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
RHO_PLUS = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
RHO_MINUS = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)
BB84_STATES = [RHO_0, RHO_1, RHO_PLUS, RHO_MINUS]


class TestNoiseParameterValidation(unittest.TestCase):
    """Test parameter domain boundaries and exception handling."""

    def test_depolarizing_parameter_validation(self):
        """Depolarizing parameter p must be in [0, 1]."""
        # Valid boundaries
        self.assertEqual(len(get_depolarizing_kraus(0.0)), 4)
        self.assertEqual(len(get_depolarizing_kraus(1.0)), 4)
        self.assertEqual(len(get_depolarizing_kraus(0.5)), 4)

        # Invalid bounds
        with self.assertRaises(ValueError):
            get_depolarizing_kraus(-0.01)
        with self.assertRaises(ValueError):
            get_depolarizing_kraus(1.01)

    def test_amplitude_damping_parameter_validation(self):
        """Amplitude damping parameter gamma must be in [0, 1]."""
        self.assertEqual(len(get_amplitude_damping_kraus(0.0)), 2)
        self.assertEqual(len(get_amplitude_damping_kraus(1.0)), 2)

        with self.assertRaises(ValueError):
            get_amplitude_damping_kraus(-0.001)
        with self.assertRaises(ValueError):
            get_amplitude_damping_kraus(1.001)

    def test_phase_damping_parameter_validation(self):
        """Phase damping parameter lambda must be in [0, 1]."""
        self.assertEqual(len(get_phase_damping_kraus(0.0)), 2)
        self.assertEqual(len(get_phase_damping_kraus(1.0)), 2)

        with self.assertRaises(ValueError):
            get_phase_damping_kraus(-0.001)
        with self.assertRaises(ValueError):
            get_phase_damping_kraus(1.001)

    def test_thermal_relaxation_parameter_validation(self):
        """Thermal relaxation requires T1 > 0, T2 > 0, time >= 0, and T2 <= 2*T1."""
        # Valid parameters
        validate_thermal_parameters(t1=50e-6, t2=70e-6, time=20e-9)
        validate_thermal_parameters(t1=50e-6, t2=100e-6, time=0.0)

        # T1 <= 0
        with self.assertRaises(ValueError):
            validate_thermal_parameters(t1=0.0, t2=50e-6, time=20e-9)
        with self.assertRaises(ValueError):
            validate_thermal_parameters(t1=-10e-6, t2=50e-6, time=20e-9)

        # T2 <= 0
        with self.assertRaises(ValueError):
            validate_thermal_parameters(t1=50e-6, t2=0.0, time=20e-9)

        # time < 0
        with self.assertRaises(ValueError):
            validate_thermal_parameters(t1=50e-6, t2=70e-6, time=-1e-9)

        # Physical violation: T2 > 2 * T1
        with self.assertRaises(ValueError):
            validate_thermal_parameters(t1=50e-6, t2=100.1e-6, time=20e-9)


class TestKrausMathematicalIntegrity(unittest.TestCase):
    """Test Kraus completeness, trace preservation, Hermiticity, and positive semi-definiteness."""

    def test_depolarizing_kraus_completeness(self):
        """Depolarizing Kraus operators satisfy sum K_i^dagger K_i = I."""
        for p in [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 1.0]:
            kraus = get_depolarizing_kraus(p)
            self.assertTrue(verify_kraus_completeness(kraus))

    def test_amplitude_damping_kraus_completeness(self):
        """Amplitude damping Kraus operators satisfy sum K_i^dagger K_i = I."""
        for gamma in [0.0, 0.02, 0.1, 0.3, 0.7, 1.0]:
            kraus = get_amplitude_damping_kraus(gamma)
            self.assertTrue(verify_kraus_completeness(kraus))

    def test_phase_damping_kraus_completeness(self):
        """Phase damping Kraus operators satisfy sum K_i^dagger K_i = I."""
        for lam in [0.0, 0.02, 0.1, 0.3, 0.7, 1.0]:
            kraus = get_phase_damping_kraus(lam)
            self.assertTrue(verify_kraus_completeness(kraus))

    def test_thermal_relaxation_kraus_completeness(self):
        """Thermal relaxation Kraus operators satisfy completeness across regimes."""
        test_cases = [
            (50e-6, 50e-6, 20e-9),    # T2 < T1
            (50e-6, 70e-6, 50e-9),    # T1 < T2 < 2*T1
            (50e-6, 100e-6, 100e-9),  # T2 = 2*T1
            (100e-6, 150e-6, 0.0),    # time = 0
        ]
        for t1, t2, t in test_cases:
            kraus = get_thermal_relaxation_kraus(t1=t1, t2=t2, time=t)
            self.assertTrue(verify_kraus_completeness(kraus))

    def test_trace_preservation_and_positivity(self):
        """All noise channels produce valid density matrices: Tr(rho)=1, rho >= 0."""
        channels = [
            get_depolarizing_kraus(0.15),
            get_amplitude_damping_kraus(0.20),
            get_phase_damping_kraus(0.20),
            get_thermal_relaxation_kraus(t1=50e-6, t2=70e-6, time=5e-6),
        ]
        for kraus in channels:
            for rho_in in BB84_STATES:
                rho_out = apply_kraus_channel(rho_in, kraus)
                is_herm, is_pos, is_trace_one = verify_density_matrix(rho_out)
                self.assertTrue(is_herm)
                self.assertTrue(is_pos)
                self.assertTrue(is_trace_one)


class TestNoiseLimitingCases(unittest.TestCase):
    """Test physical limiting cases (zero noise and maximal noise)."""

    def test_zero_noise_identity(self):
        """Zero noise leaves all BB84 states unchanged."""
        zero_channels = [
            get_depolarizing_kraus(0.0),
            get_amplitude_damping_kraus(0.0),
            get_phase_damping_kraus(0.0),
            get_thermal_relaxation_kraus(t1=50e-6, t2=70e-6, time=0.0),
        ]
        for kraus in zero_channels:
            for rho_in in BB84_STATES:
                rho_out = apply_kraus_channel(rho_in, kraus)
                np.testing.assert_allclose(rho_out, rho_in, atol=1e-12)

    def test_maximal_depolarizing_produces_completely_mixed_state(self):
        """At p = 1.0, depolarizing channel maps any state to I/2."""
        kraus = get_depolarizing_kraus(1.0)
        expected = np.eye(2, dtype=np.complex128) * 0.5
        for rho_in in BB84_STATES:
            rho_out = apply_kraus_channel(rho_in, kraus)
            np.testing.assert_allclose(rho_out, expected, atol=1e-12)

    def test_maximal_amplitude_damping_collapses_to_ground(self):
        """At gamma = 1.0, |1><1| completely decays to |0><0|."""
        kraus = get_amplitude_damping_kraus(1.0)
        rho_out = apply_kraus_channel(RHO_1, kraus)
        np.testing.assert_allclose(rho_out, RHO_0, atol=1e-12)

    def test_maximal_phase_damping_destroys_coherence(self):
        """At lambda = 1.0, off-diagonal coherence terms vanish completely."""
        kraus = get_phase_damping_kraus(1.0)
        rho_out = apply_kraus_channel(RHO_PLUS, kraus)
        expected = np.eye(2, dtype=np.complex128) * 0.5
        np.testing.assert_allclose(rho_out, expected, atol=1e-12)


class TestAnalyticalQBERPredictors(unittest.TestCase):
    """Test that analytical QBER formulas exactly match density matrix expectations."""

    def test_depolarizing_qber_formula(self):
        """Depolarizing QBER is symmetric: Q_total = Q_Z = Q_X = p / 2."""
        for p in [0.0, 0.02, 0.06, 0.10, 0.20]:
            q_tot, q_z, q_x = predict_depolarizing_qber(p)
            self.assertAlmostEqual(q_tot, p / 2.0, places=12)
            self.assertAlmostEqual(q_z, p / 2.0, places=12)
            self.assertAlmostEqual(q_x, p / 2.0, places=12)

    def test_amplitude_damping_qber_asymmetry(self):
        """Amplitude damping causes bit asymmetry: Q_Z = gamma / 2, Q_X = (1 - sqrt(1 - gamma)) / 2."""
        for gamma in [0.0, 0.05, 0.10, 0.20]:
            q_tot, q_z, q_x = predict_amplitude_damping_qber(gamma)
            expected_qz = gamma / 2.0
            expected_qx = (1.0 - np.sqrt(1.0 - gamma)) / 2.0
            self.assertAlmostEqual(q_z, expected_qz, places=12)
            self.assertAlmostEqual(q_x, expected_qx, places=12)
            self.assertAlmostEqual(q_tot, 0.5 * (expected_qz + expected_qx), places=12)

    def test_phase_damping_qber_asymmetry(self):
        """Phase damping preserves computational basis: Q_Z = 0, Q_X = (1 - sqrt(1 - lambda)) / 2."""
        for lam in [0.0, 0.05, 0.10, 0.20]:
            q_tot, q_z, q_x = predict_phase_damping_qber(lam)
            expected_qz = 0.0
            expected_qx = (1.0 - np.sqrt(1.0 - lam)) / 2.0
            self.assertEqual(q_z, 0.0)
            self.assertAlmostEqual(q_x, expected_qx, places=12)
            self.assertAlmostEqual(q_tot, 0.5 * expected_qx, places=12)

    def test_thermal_relaxation_qber_prediction(self):
        """Thermal relaxation correctly models longitudinal decay and transverse dephasing."""
        # Case A: T2 > T1 -> decay rate 1/T1 > 1/T2, so Q_Z > Q_X
        q_tot, q_z, q_x = predict_thermal_relaxation_qber(t1=50e-6, t2=70e-6, time=20e-9)
        self.assertGreater(q_tot, 0.0)
        self.assertGreater(q_z, q_x)

        # Case B: Strong pure dephasing T2 < T1 -> dephasing rate 1/T2 > 1/T1, so Q_X > Q_Z
        q_tot2, q_z2, q_x2 = predict_thermal_relaxation_qber(t1=50e-6, t2=20e-6, time=20e-9)
        self.assertGreater(q_tot2, 0.0)
        self.assertGreater(q_x2, q_z2)

    def test_density_matrix_qber_matches_analytical_predictions(self):
        """Density matrix measurements exactly match analytical QBER predictors."""
        # Depolarizing
        for p in [0.01, 0.05, 0.15]:
            kraus = get_depolarizing_kraus(p)
            q_tot, q_z, q_x = compute_basis_qber_from_density_matrices(kraus)
            p_tot, p_z, p_x = predict_depolarizing_qber(p)
            self.assertAlmostEqual(q_tot, p_tot, places=10)
            self.assertAlmostEqual(q_z, p_z, places=10)
            self.assertAlmostEqual(q_x, p_x, places=10)

        # Amplitude damping
        for gamma in [0.02, 0.10, 0.25]:
            kraus = get_amplitude_damping_kraus(gamma)
            q_tot, q_z, q_x = compute_basis_qber_from_density_matrices(kraus)
            p_tot, p_z, p_x = predict_amplitude_damping_qber(gamma)
            self.assertAlmostEqual(q_tot, p_tot, places=10)
            self.assertAlmostEqual(q_z, p_z, places=10)
            self.assertAlmostEqual(q_x, p_x, places=10)

        # Phase damping
        for lam in [0.02, 0.10, 0.25]:
            kraus = get_phase_damping_kraus(lam)
            q_tot, q_z, q_x = compute_basis_qber_from_density_matrices(kraus)
            p_tot, p_z, p_x = predict_phase_damping_qber(lam)
            self.assertAlmostEqual(q_tot, p_tot, places=10)
            self.assertAlmostEqual(q_z, p_z, places=10)
            self.assertAlmostEqual(q_x, p_x, places=10)

    def test_state_fidelity_and_coherence(self):
        """State fidelity and coherence decay follow analytical Kraus expectations."""
        # Depolarizing: F = 1 - p/2
        for p in [0.0, 0.1, 0.3]:
            kraus = get_depolarizing_kraus(p)
            avg_f = compute_channel_fidelity(kraus)
            self.assertAlmostEqual(avg_f, 1.0 - p / 2.0, places=10)

        # Phase damping: coherence C_l1 of |+><+| is sqrt(1 - lambda)
        for lam in [0.0, 0.2, 0.5]:
            kraus = get_phase_damping_kraus(lam)
            rho_out = apply_kraus_channel(RHO_PLUS, kraus)
            coherence = compute_coherence(rho_out)
            self.assertAlmostEqual(coherence, np.sqrt(1.0 - lam), places=10)


class TestQiskitNoiseModelBuilders(unittest.TestCase):
    """Test constructing Aer NoiseModels and NoiseChannelProfile containers."""

    def test_build_depolarizing_noise_model(self):
        """Build depolarizing profile."""
        profile = build_depolarizing_noise_model(p=0.05)
        self.assertIsInstance(profile, NoiseChannelProfile)
        self.assertEqual(profile.noise_type, NoiseType.DEPOLARIZING)
        self.assertEqual(len(profile.kraus_operators), 4)
        self.assertAlmostEqual(profile.theoretical_qber, 0.025)

    def test_build_amplitude_damping_noise_model(self):
        """Build amplitude damping profile."""
        profile = build_amplitude_damping_noise_model(gamma=0.08)
        self.assertIsInstance(profile, NoiseChannelProfile)
        self.assertEqual(profile.noise_type, NoiseType.AMPLITUDE_DAMPING)
        self.assertEqual(len(profile.kraus_operators), 2)
        self.assertAlmostEqual(profile.theoretical_qber_z, 0.04)

    def test_build_phase_damping_noise_model(self):
        """Build phase damping profile."""
        profile = build_phase_damping_noise_model(lam=0.08)
        self.assertIsInstance(profile, NoiseChannelProfile)
        self.assertEqual(profile.noise_type, NoiseType.PHASE_DAMPING)
        self.assertEqual(len(profile.kraus_operators), 2)
        self.assertEqual(profile.theoretical_qber_z, 0.0)

    def test_build_thermal_relaxation_noise_model(self):
        """Build thermal relaxation profile."""
        profile = build_thermal_relaxation_noise_model(t1=50e-6, t2=70e-6, time=20e-9)
        self.assertIsInstance(profile, NoiseChannelProfile)
        self.assertEqual(profile.noise_type, NoiseType.THERMAL_RELAXATION)

    def test_build_combined_noise_model(self):
        """Build combined multi-source noise model."""
        profile = build_combined_noise_model(
            depol_p=0.02,
            amp_gamma=0.02,
            phase_lam=0.02,
        )
        self.assertIsInstance(profile, NoiseChannelProfile)
        self.assertEqual(profile.noise_type, NoiseType.COMBINED)


if __name__ == "__main__":
    unittest.main()
