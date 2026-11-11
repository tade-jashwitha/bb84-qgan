import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit_aer import AerSimulator

from src.bb84.alice import Alice, BASIS_Z, BASIS_X, prepare_bb84_state
from src.bb84.bob import Bob, apply_measurement_basis, reconcile_bases, compute_qber
from src.bb84.attacks import FuchsPeresBrandtProbe, build_probe_unitary, InterceptResendAttack
from src.bb84.bb84_engine import BB84Engine
from src.channel.noise_models import (
    get_depolarizing_kraus,
    get_amplitude_damping_kraus,
    get_phase_damping_kraus,
    get_thermal_relaxation_kraus,
    build_depolarizing_noise_model,
    build_amplitude_damping_noise_model,
    build_phase_damping_noise_model,
)
from src.channel.quantum_channel import QuantumChannel


class TestQuantumPhysicsAudit(unittest.TestCase):
    """Rigorous audit and regression tests for Phase 1 quantum physics verification."""

    # --------------------------------------------------------------------------
    # Issue 1: attacks.py - Unitary probe matrix, basis ordering, and unitarity
    # --------------------------------------------------------------------------
    def test_issue1_unitary_probe_matrix_and_basis_ordering(self):
        """Verify U_E(theta) matrix unitarity, basis ordering, and sign consistency."""
        for theta in [0.0, np.pi / 12.0, np.pi / 8.0, np.pi / 6.0, np.pi / 4.0, np.pi / 2.0]:
            U = build_probe_unitary(theta)
            c = np.cos(theta)
            s = np.sin(theta)

            # Unitarity test
            diff = np.max(np.abs(U.conj().T @ U - np.eye(4, dtype=np.complex128)))
            self.assertLess(diff, 1e-14, f"Matrix not unitary at theta={theta}")

            # Verify action on standard Qiskit basis |q1, q0> = |ancilla, carrier>
            # |00>_EA -> c |00> + s |11>
            col0 = U[:, 0]
            expected_col0 = np.array([c, 0.0, 0.0, s], dtype=np.complex128)
            np.testing.assert_allclose(col0, expected_col0, atol=1e-14)

            # |01>_EA -> c |01> - s |10> (carrier=1, ancilla=0)
            col1 = U[:, 1]
            expected_col1 = np.array([0.0, c, -s, 0.0], dtype=np.complex128)
            np.testing.assert_allclose(col1, expected_col1, atol=1e-14)

            # |10>_EA -> c |10> + s |01> (carrier=0, ancilla=1)
            col2 = U[:, 2]
            expected_col2 = np.array([0.0, s, c, 0.0], dtype=np.complex128)
            np.testing.assert_allclose(col2, expected_col2, atol=1e-14)

            # |11>_EA -> c |11> - s |00> (carrier=1, ancilla=1)
            col3 = U[:, 3]
            expected_col3 = np.array([-s, 0.0, 0.0, c], dtype=np.complex128)
            np.testing.assert_allclose(col3, expected_col3, atol=1e-14)

    # --------------------------------------------------------------------------
    # Issue 2: attacks.py - Physical disturbance D(theta_E) = sin^2(theta_E)
    # --------------------------------------------------------------------------
    def test_issue2_probe_disturbance_exact_all_bb84_states(self):
        """Verify D(theta_E) = sin^2(theta_E) holds exactly for all 4 BB84 states."""
        states = {
            "|0>": (np.array([1.0, 0.0], dtype=np.complex128), BASIS_Z, 0),
            "|1>": (np.array([0.0, 1.0], dtype=np.complex128), BASIS_Z, 1),
            "|+>": (np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2), BASIS_X, 0),
            "|->": (np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2), BASIS_X, 1),
        }

        H_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        for theta in [0.0, np.pi / 12.0, np.pi / 8.0, np.pi / 6.0, np.pi / 4.0]:
            probe = FuchsPeresBrandtProbe(theta_E=theta)
            expected_dist = float(np.sin(theta) ** 2)
            self.assertAlmostEqual(probe.theoretical_disturbance(), expected_dist, places=14)

            for name, (amp, basis, bit) in states.items():
                rho_A, rho_E, _ = probe.compute_reduced_states(amp)

                # Compute Bob's error probability in the matching basis
                if basis == BASIS_Z:
                    # In Z: error is probability of opposite bit
                    error_prob = float(np.real(rho_A.data[1 - bit, 1 - bit]))
                else:
                    # In X: transform to X basis via H rho H
                    rho_x = H_gate @ rho_A.data @ H_gate
                    error_prob = float(np.real(rho_x[1 - bit, 1 - bit]))

                self.assertAlmostEqual(
                    error_prob,
                    expected_dist,
                    places=14,
                    msg=f"Disturbance mismatch for {name} at theta={theta}",
                )

    # --------------------------------------------------------------------------
    # Issue 3: attacks.py - Eve's information and ancilla states
    # --------------------------------------------------------------------------
    def test_issue3_eve_ancilla_states_and_theoretical_bound(self):
        """Verify Eve's ancilla density matrices and theoretical Helstrom information bound."""
        states = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],
            [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)],
        ]

        for theta in [0.0, np.pi / 8.0, np.pi / 6.0, np.pi / 4.0]:
            probe = FuchsPeresBrandtProbe(theta_E=theta)
            c2 = np.cos(theta) ** 2
            s2 = np.sin(theta) ** 2
            expected_rho_E = np.diag([c2, s2])

            # In the 2-qubit circuit model, ancilla state is identical across all BB84 states
            rho_E_list = [probe.compute_reduced_states(s)[1] for s in states]
            for idx, rE in enumerate(rho_E_list):
                np.testing.assert_allclose(
                    rE.data,
                    expected_rho_E,
                    atol=1e-14,
                    err_msg=f"Ancilla state differs for state {idx} at theta={theta}",
                )

            # Theoretical mutual information bound evaluates the Helstrom bound
            I_th = probe.theoretical_information_gain()
            if np.isclose(theta, 0.0):
                self.assertAlmostEqual(I_th, 0.0, places=14)
            elif np.isclose(theta, np.pi / 4.0):
                # At theta=pi/4, sin(2*theta)=1, p_succ=1.0 -> I = 1.0
                self.assertAlmostEqual(I_th, 1.0, places=14)
            else:
                p_succ = 0.5 * (1.0 + np.sin(2.0 * theta))
                h2 = -p_succ * np.log2(p_succ) - (1.0 - p_succ) * np.log2(1.0 - p_succ)
                self.assertAlmostEqual(I_th, 1.0 - h2, places=14)

    # --------------------------------------------------------------------------
    # Issue 4: bb84_engine.py - QBER strictly uses sifted bits
    # --------------------------------------------------------------------------
    def test_issue4_qber_uses_sifted_bits_denominator(self):
        """Verify that QBER uses only sifted bits and handles empty sifting safely."""
        # Case A: Deterministic basis choices and bit values
        alice_bits = [0, 1, 0, 1, 0, 1, 0, 1]
        alice_bases = ["Z", "Z", "Z", "Z", "X", "X", "X", "X"]
        bob_bases = ["Z", "X", "Z", "X", "Z", "X", "Z", "X"]
        # Matching basis indices: 0, 2, 5, 7 -> total 4 sifted bits
        bob_bits = [0, 0, 1, 0, 0, 1, 0, 1]  # error at index 2 (1 != 0)

        sift_res = reconcile_bases(
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            alice_bits=alice_bits,
            bob_bits=bob_bits,
        )
        self.assertEqual(sift_res.matching_bases_count, 4)
        np.testing.assert_array_equal(sift_res.sifted_indices, [0, 2, 5, 7])

        qber_res = compute_qber(sift_res.alice_sifted_bits, sift_res.bob_sifted_bits)
        self.assertEqual(qber_res.total_bits, 4)  # Denominator is 4, NOT 8!
        self.assertEqual(qber_res.error_count, 1)
        self.assertAlmostEqual(qber_res.qber, 0.25, places=14)

        # Case B: Zero matching bases (orthogonal basis sets)
        bob_bases_orth = ["X", "X", "X", "X", "Z", "Z", "Z", "Z"]
        sift_orth = reconcile_bases(alice_bases, bob_bases_orth, alice_bits, bob_bits)
        self.assertEqual(sift_orth.matching_bases_count, 0)
        qber_empty = compute_qber(sift_orth.alice_sifted_bits, sift_orth.bob_sifted_bits)
        self.assertEqual(qber_empty.qber, 0.0)
        self.assertEqual(qber_empty.error_count, 0)
        self.assertEqual(qber_empty.total_bits, 0)

    # --------------------------------------------------------------------------
    # Issue 5: bob.py - X-basis measurement implementation
    # --------------------------------------------------------------------------
    def test_issue5_bob_x_basis_measurement_all_states(self):
        """Verify that Bob's X-basis measurement implements H before computational measurement."""
        backend = AerSimulator()

        # Deterministic: |+> in X basis must yield 0 with 100% certainty
        qc_plus = prepare_bb84_state(0, BASIS_X)
        meas_x = apply_measurement_basis(qc_plus, BASIS_X)
        res_plus = backend.run(meas_x, shots=1000, seed_simulator=101).result().get_counts()
        self.assertEqual(res_plus.get("0", 0), 1000)
        self.assertEqual(res_plus.get("1", 0), 0)

        # Deterministic: |-> in X basis must yield 1 with 100% certainty
        qc_minus = prepare_bb84_state(1, BASIS_X)
        meas_minus = apply_measurement_basis(qc_minus, BASIS_X)
        res_minus = backend.run(meas_minus, shots=1000, seed_simulator=102).result().get_counts()
        self.assertEqual(res_minus.get("1", 0), 1000)
        self.assertEqual(res_minus.get("0", 0), 0)

        # Statistical: |0> in X basis must yield uniform 50/50 distribution
        qc_zero = prepare_bb84_state(0, BASIS_Z)
        meas_z_in_x = apply_measurement_basis(qc_zero, BASIS_X)
        res_zx = backend.run(meas_z_in_x, shots=5000, seed_simulator=103).result().get_counts()
        p0 = res_zx.get("0", 0) / 5000.0
        self.assertAlmostEqual(p0, 0.50, delta=0.035)

    # --------------------------------------------------------------------------
    # Issue 6: noise_models.py - Kraus operator completeness and positivity
    # --------------------------------------------------------------------------
    def test_issue6_kraus_completeness_and_trace_preservation(self):
        """Verify sum K_i^dagger K_i = I and physical density matrix preservation for all channels."""
        test_rhos = [
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
            np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128),
            np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128),
            np.array([[0.7, 0.2 - 0.1j], [0.2 + 0.1j, 0.3]], dtype=np.complex128),
        ]

        channel_kraus_sets = []
        # Depolarizing sweeps
        for p in [0.0, 0.05, 0.5, 1.0]:
            channel_kraus_sets.append((f"Depol(p={p})", get_depolarizing_kraus(p)))
        # Amplitude damping sweeps
        for gamma in [0.0, 0.05, 0.5, 1.0]:
            channel_kraus_sets.append((f"AmpDamp(g={gamma})", get_amplitude_damping_kraus(gamma)))
        # Phase damping sweeps
        for lam in [0.0, 0.05, 0.5, 1.0]:
            channel_kraus_sets.append((f"PhaseDamp(l={lam})", get_phase_damping_kraus(lam)))
        # Thermal relaxation sweeps
        for t1, t2, time in [(100.0, 100.0, 1.0), (50.0, 30.0, 5.0), (100.0, 200.0, 10.0)]:
            channel_kraus_sets.append((f"Thermal({t1},{t2},{time})", get_thermal_relaxation_kraus(t1, t2, time)))

        I2 = np.eye(2, dtype=np.complex128)
        for name, kraus_ops in channel_kraus_sets:
            # Completeness: sum K_i^\dagger K_i = I
            completeness = sum(K.conj().T @ K for K in kraus_ops)
            np.testing.assert_allclose(completeness, I2, atol=1e-14, err_msg=f"{name} completeness failed")

            # Trace preservation and positivity on all test density matrices
            for rho in test_rhos:
                rho_out = sum(K @ rho @ K.conj().T for K in kraus_ops)
                tr = float(np.real(np.trace(rho_out)))
                self.assertAlmostEqual(tr, 1.0, places=14, msg=f"{name} trace preservation failed")

                # Hermiticity
                herm_diff = np.max(np.abs(rho_out - rho_out.conj().T))
                self.assertLess(herm_diff, 1e-14, msg=f"{name} Hermiticity failed")

                # Eigenvalues positive
                evals = np.linalg.eigvalsh(rho_out)
                self.assertGreaterEqual(np.min(evals), -1e-14, msg=f"{name} negative eigenvalue: {evals}")

    # --------------------------------------------------------------------------
    # Issue 7: quantum_channel.py & bb84_engine.py - Operation order verification
    # --------------------------------------------------------------------------
    def test_issue7_combined_attack_and_noise_order(self):
        """Verify that Eve attack occurs before channel transit noise and noise is on qubit 0."""
        # 1. Verify that QuantumChannel.transmit_batch applies transit noise to probed circuits
        probe = FuchsPeresBrandtProbe(theta_E=np.pi / 8.0)
        noise_profile = build_depolarizing_noise_model(p=0.05)
        channel = QuantumChannel(attack=probe, noise_model=noise_profile)

        alice_qc = prepare_bb84_state(0, BASIS_Z)
        res = channel.transmit_batch([alice_qc])
        self.assertTrue(res.is_noisy)
        self.assertTrue(res.is_eavesdropped)
        probed_qc = res.received_circuits[0]
        # Verify circuit structure: qubit 0 has carrier prep, 2-qubit gate, and id(0) transit noise
        self.assertEqual(probed_qc.num_qubits, 2)
        op_names = [inst.operation.name for inst in probed_qc.data]
        self.assertTrue("unitary" in op_names or "U_probe" in op_names)
        self.assertIn("id", op_names)

        # 2. Verify BB84Engine simulation order of operations:
        # Probe gate comes before transit id gate in probe_meas circuits
        engine = BB84Engine(alice_seed=555, bob_seed=555)
        sim_res = engine.run(
            n_signals=5000,
            attack=probe,
            noise=noise_profile,
            seed_simulator=555,
        )
        # Expected compounded QBER: Q_tot = 0.025 + sin^2(pi/8) - 2 * 0.025 * sin^2(pi/8) ~ 0.1641
        expected_compounded_qber = 0.025 + np.sin(np.pi / 8.0)**2 - 2.0 * 0.025 * np.sin(np.pi / 8.0)**2
        self.assertAlmostEqual(sim_res.qber, expected_compounded_qber, delta=0.02)


if __name__ == "__main__":
    unittest.main()
