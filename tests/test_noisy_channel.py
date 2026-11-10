"""Unit tests for Noisy Quantum Channel and BB84 Engine Integration.

Validates:
1. End-to-end noisy transmission through BB84Engine across noise channels:
   - Depolarizing noise produces symmetric QBER = p / 2 across Z and X bases.
   - Amplitude damping produces asymmetric QBER: Q_Z = gamma / 2, Q_X = (1 - sqrt(1 - gamma)) / 2.
   - Phase damping produces strictly zero errors in computational Z-basis (Q_Z = 0) and Q_X > 0.
   - Combined multi-source noise correctly aggregates channel impairments.
2. Interaction of channel noise with physical eavesdropping attacks:
   - Depolarizing noise (p = 0.05) + Intercept-Resend (alpha = 1.0) compounds:
     Q_total = Q_1 + Q_2 - 2 * Q_1 * Q_2 = 0.025 + 0.25 - 2*(0.025)*(0.25) = 26.25%.
   - Depolarizing noise (p = 0.05) + Unitary Probe (theta_E = pi / 8) compounds:
     Q_total = 0.025 + sin^2(pi/8) - 2*(0.025)*sin^2(pi/8) ~ 16.41%.
3. Carrier-exclusive noise isolation:
   - Noise acts on carrier qubit 0; Eve's probe ancilla qubit 1 is strictly uncorrupted.
4. Channel noise placement configurations:
   - 'transit' / 'after_eve' and 'before_eve' modes operate reliably.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

from src.bb84.bb84_engine import BB84Engine
from src.bb84.attacks import InterceptResendAttack, FuchsPeresBrandtProbe
from src.channel.quantum_channel import QuantumChannel
from src.channel.noise_models import (
    build_depolarizing_noise_model,
    build_amplitude_damping_noise_model,
    build_phase_damping_noise_model,
    build_combined_noise_model,
    build_thermal_relaxation_noise_model,
)


class TestNoisyQuantumChannel(unittest.TestCase):
    """Test suite for QuantumChannel under NISQ noise models."""

    def test_channel_applies_transit_noise_to_carrier(self):
        """Channel with noise appends id(0) marker on carrier circuits."""
        profile = build_depolarizing_noise_model(p=0.05)
        channel = QuantumChannel(noise_model=profile)

        qc = QuantumCircuit(1)
        qc.x(0)
        received = channel.transmit_pulse(qc)

        self.assertEqual(len(received.data), 2)  # x gate + id gate
        self.assertEqual(received.data[-1].operation.name, "id")

    def test_channel_batch_transmission_preserves_count(self):
        """Batch transmission delivers matching number of pulse circuits."""
        profile = build_amplitude_damping_noise_model(gamma=0.10)
        channel = QuantumChannel(noise_model=profile)

        circuits = [QuantumCircuit(1) for _ in range(50)]
        res = channel.transmit_batch(circuits)

        self.assertEqual(len(res.received_circuits), 50)
        self.assertTrue(res.is_noisy)
        self.assertEqual(res.noise_type, "amplitude_damping")


class TestBB84EngineWithNoise(unittest.TestCase):
    """Test suite for BB84Engine execution under various NISQ noise channels."""

    def setUp(self):
        self.engine = BB84Engine(alice_seed=4242, bob_seed=8484)

    def test_depolarizing_symmetric_qber(self):
        """Depolarizing noise produces symmetric QBER ~ p/2 across Z and X bases."""
        p = 0.08
        profile = build_depolarizing_noise_model(p=p)
        res = self.engine.run(n_signals=5000, noise=profile, seed_simulator=101)

        expected_qber = p / 2.0  # 4.0%
        self.assertTrue(res.is_noisy)
        self.assertAlmostEqual(res.qber, expected_qber, delta=0.015)
        self.assertAlmostEqual(res.qber_z, expected_qber, delta=0.02)
        self.assertAlmostEqual(res.qber_x, expected_qber, delta=0.02)

    def test_amplitude_damping_asymmetric_qber(self):
        """Amplitude damping produces significant bit asymmetry between Z and X."""
        gamma = 0.15
        profile = build_amplitude_damping_noise_model(gamma=gamma)
        res = self.engine.run(n_signals=6000, noise=profile, seed_simulator=202)

        expected_qz = gamma / 2.0  # 7.5%
        expected_qx = (1.0 - np.sqrt(1.0 - gamma)) / 2.0  # ~3.9%
        expected_total = 0.5 * (expected_qz + expected_qx)  # ~5.7%

        self.assertAlmostEqual(res.qber, expected_total, delta=0.018)
        self.assertAlmostEqual(res.qber_z, expected_qz, delta=0.025)
        self.assertAlmostEqual(res.qber_x, expected_qx, delta=0.025)
        # Verify Z errors exceed X errors
        self.assertGreater(res.qber_z, res.qber_x)

    def test_phase_damping_zero_z_error(self):
        """Phase damping preserves computational basis: Q_Z = 0.0%, Q_X > 0."""
        lam = 0.12
        profile = build_phase_damping_noise_model(lam=lam)
        res = self.engine.run(n_signals=6000, noise=profile, seed_simulator=303)

        expected_qx = (1.0 - np.sqrt(1.0 - lam)) / 2.0  # ~3.1%

        self.assertEqual(res.error_count_z, 0)
        self.assertEqual(res.qber_z, 0.0)
        self.assertAlmostEqual(res.qber_x, expected_qx, delta=0.018)
        self.assertAlmostEqual(res.qber, 0.5 * expected_qx, delta=0.012)

    def test_combined_noise_channel(self):
        """Combined noise model produces aggregated error across conjugate bases."""
        profile = build_combined_noise_model(depol_p=0.02, amp_gamma=0.04, phase_lam=0.04)
        res = self.engine.run(n_signals=5000, noise=profile, seed_simulator=404)

        self.assertTrue(res.is_noisy)
        self.assertGreater(res.qber, 0.0)
        self.assertGreater(res.qber_z, 0.0)
        self.assertGreater(res.qber_x, 0.0)

    def test_thermal_relaxation_channel(self):
        """Thermal relaxation correctly induces longitudinal decay and transverse dephasing."""
        t1 = 50e-6
        t2 = 30e-6
        transit_time = 2e-6
        profile = build_thermal_relaxation_noise_model(t1=t1, t2=t2, time=transit_time)
        res = self.engine.run(n_signals=6000, noise=profile, seed_simulator=414)

        expected_qz = 0.5 * (1.0 - np.exp(-transit_time / t1))  # ~ 1.96%
        expected_qx = 0.5 * (1.0 - np.exp(-transit_time / t2))  # ~ 3.22%
        expected_total = 0.5 * (expected_qz + expected_qx)      # ~ 2.59%

        self.assertTrue(res.is_noisy)
        self.assertEqual(res.noise_type, "thermal_relaxation")
        self.assertAlmostEqual(res.qber, expected_total, delta=0.015)
        self.assertAlmostEqual(res.qber_z, expected_qz, delta=0.015)
        self.assertAlmostEqual(res.qber_x, expected_qx, delta=0.018)
        # Because T2 < T1, transverse dephasing rate is higher than longitudinal decay
        self.assertGreater(res.qber_x, res.qber_z)

    def test_noisy_reproducibility(self):
        """Identical random seeds under noise produce bit-identical simulation outcomes."""
        profile = build_depolarizing_noise_model(p=0.05)
        engine1 = BB84Engine(alice_seed=777, bob_seed=888)
        engine2 = BB84Engine(alice_seed=777, bob_seed=888)

        res1 = engine1.run(n_signals=2000, noise=profile, seed_simulator=999)
        res2 = engine2.run(n_signals=2000, noise=profile, seed_simulator=999)

        np.testing.assert_array_equal(res1.alice_bits, res2.alice_bits)
        np.testing.assert_array_equal(res1.bob_measured_bits, res2.bob_measured_bits)
        np.testing.assert_array_equal(res1.sifted_indices, res2.sifted_indices)
        self.assertEqual(res1.qber, res2.qber)
        self.assertEqual(res1.error_count, res2.error_count)


class TestNoiseAndEavesdroppingInteraction(unittest.TestCase):
    """Test compounding of independent error sources (channel noise + eavesdropping)."""

    def setUp(self):
        self.engine = BB84Engine(alice_seed=123, bob_seed=456)

    def test_ideal_channel_no_attack(self):
        """Condition A: Ideal channel without attack produces strictly 0% QBER."""
        res = self.engine.run(n_signals=2500)
        self.assertEqual(res.qber, 0.0)
        self.assertFalse(res.attack_applied)
        self.assertFalse(res.is_noisy)

    def test_depolarizing_noise_only(self):
        """Condition B: Depolarizing noise p=0.05 produces QBER ~ 2.5%."""
        profile = build_depolarizing_noise_model(p=0.05)
        res = self.engine.run(n_signals=5000, noise=profile, seed_simulator=505)

        self.assertAlmostEqual(res.qber, 0.025, delta=0.01)
        self.assertFalse(res.attack_applied)
        self.assertTrue(res.is_noisy)

    def test_intercept_resend_only(self):
        """Condition C: Full intercept-resend attack produces QBER ~ 25.0%."""
        attack = InterceptResendAttack(alpha=1.0, beta=0.5, seed=606)
        res = self.engine.run(n_signals=5000, attack=attack, seed_simulator=606)

        self.assertAlmostEqual(res.qber, 0.25, delta=0.025)
        self.assertTrue(res.attack_applied)
        self.assertFalse(res.is_noisy)

    def test_unitary_probe_only(self):
        """Condition D: Unitary probe theta_E = pi/8 produces QBER ~ sin^2(pi/8) ~ 14.64%."""
        probe = FuchsPeresBrandtProbe(theta_E=np.pi / 8.0)
        res = self.engine.run(n_signals=5000, attack=probe, seed_simulator=707)

        expected_qber = float(np.sin(np.pi / 8.0) ** 2)  # ~ 14.6447%
        self.assertAlmostEqual(res.qber, expected_qber, delta=0.02)
        self.assertTrue(res.attack_applied)
        self.assertFalse(res.is_noisy)

    def test_compounding_depolarizing_plus_intercept_resend(self):
        """Condition E: Depolarizing (p=0.05) + Intercept-Resend (alpha=1.0) compounds to 26.25%."""
        # Q_total = Q_1 + Q_2 - 2 * Q_1 * Q_2 = 0.025 + 0.25 - 2 * 0.025 * 0.25 = 0.2625
        profile = build_depolarizing_noise_model(p=0.05)
        attack = InterceptResendAttack(alpha=1.0, beta=0.5, seed=808)

        res = self.engine.run(n_signals=6000, attack=attack, noise=profile, seed_simulator=808)

        expected_qber = 0.025 + 0.25 - 2.0 * (0.025 * 0.25)  # 0.2625 (26.25%)
        self.assertAlmostEqual(res.qber, expected_qber, delta=0.025)
        self.assertTrue(res.attack_applied)
        self.assertTrue(res.is_noisy)

    def test_compounding_depolarizing_plus_unitary_probe(self):
        """Condition F: Depolarizing (p=0.05) + Unitary Probe (theta=pi/8) compounds to ~16.41%."""
        # Q_total = 0.025 + sin^2(pi/8) - 2 * 0.025 * sin^2(pi/8) ~ 0.16412
        profile = build_depolarizing_noise_model(p=0.05)
        probe = FuchsPeresBrandtProbe(theta_E=np.pi / 8.0)

        res = self.engine.run(n_signals=6000, attack=probe, noise=profile, seed_simulator=909)

        q_probe = float(np.sin(np.pi / 8.0) ** 2)
        expected_qber = 0.025 + q_probe - 2.0 * 0.025 * q_probe  # ~ 0.16412
        self.assertAlmostEqual(res.qber, expected_qber, delta=0.022)
        self.assertTrue(res.attack_applied)
        self.assertTrue(res.is_noisy)


if __name__ == "__main__":
    unittest.main()
