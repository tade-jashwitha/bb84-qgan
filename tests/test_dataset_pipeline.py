"""Unit and integration tests for the BB84 attack-policy dataset generation pipeline.

Tests:
1. Wilson score interval calculations and boundary behavior.
2. Three-state feasibility classification.
3. Attack configuration generator and physical instantiation.
4. Schema serialization and deserialization integrity.
5. BB84 simulator integration and bit partitioning.
6. Deterministic reproducibility with random seeds.
7. Mini-dataset generation and programmatic integrity checks.
"""

import json
import math
import os
import tempfile
import unittest
import numpy as np

from src.bb84.attacks import FuchsPeresBrandtProbe, InterceptResendAttack
from src.dataset.schema import (
    AttackConfiguration,
    AttackFamily,
    AttackParameters,
    DatasetContainer,
    DatasetRecord,
    DerivedEvaluation,
    EnvironmentalCondition,
    RawObservation,
    SimulationMetadata,
    StatisticalFeasibility,
)
from src.dataset.statistics import (
    classify_feasibility,
    compute_empirical_frequencies,
    compute_wilson_score_interval,
    get_standard_normal_quantile,
)
from src.dataset.policy_generator import (
    get_all_attack_configurations,
    get_attack_configuration,
    instantiate_attack,
)
from src.dataset.generator import (
    execute_single_simulation,
    generate_dataset,
    generate_environmental_condition,
    get_pilot_environmental_conditions,
)


class TestWilsonScoreInterval(unittest.TestCase):
    """Test suite for Wilson score confidence intervals and boundary cases."""

    def test_standard_normal_quantiles(self):
        """Verify normal quantiles for standard confidence levels."""
        z95 = get_standard_normal_quantile(0.95)
        self.assertAlmostEqual(z95, 1.959964, places=5)
        z99 = get_standard_normal_quantile(0.99)
        self.assertAlmostEqual(z99, 2.575829, places=5)
        z90 = get_standard_normal_quantile(0.90)
        self.assertAlmostEqual(z90, 1.644854, places=5)

    def test_invalid_arguments_raise_error(self):
        """Verify proper exceptions on invalid inputs."""
        with self.assertRaises(ValueError):
            compute_wilson_score_interval(k=0, n=0)  # n < 1
        with self.assertRaises(ValueError):
            compute_wilson_score_interval(k=-1, n=100)  # k < 0
        with self.assertRaises(ValueError):
            compute_wilson_score_interval(k=101, n=100)  # k > n
        with self.assertRaises(ValueError):
            compute_wilson_score_interval(k=5, n=10, confidence=1.5)  # conf > 1

    def test_zero_errors_boundary(self):
        """Test boundary case with zero observed errors (k = 0)."""
        n = 500
        res = compute_wilson_score_interval(k=0, n=n, confidence=0.95)
        self.assertEqual(res.k, 0)
        self.assertEqual(res.n, n)
        self.assertEqual(res.lcb, 0.0)  # Strictly zero lower bound
        self.assertGreater(res.ucb, 0.0)
        # Analytical Wilson UCB for k=0: z^2 / (n + z^2)
        z = get_standard_normal_quantile(0.95)
        expected_ucb = (z * z) / (n + z * z)
        self.assertAlmostEqual(res.ucb, expected_ucb, places=6)
        self.assertLess(res.ucb, 0.01)  # At n=500, UCB is < 1%

    def test_all_errors_boundary(self):
        """Test boundary case where every trial is an error (k = n)."""
        n = 500
        res = compute_wilson_score_interval(k=n, n=n, confidence=0.95)
        self.assertEqual(res.k, n)
        self.assertEqual(res.ucb, 1.0)  # Strictly unit upper bound
        self.assertLess(res.lcb, 1.0)
        z = get_standard_normal_quantile(0.95)
        expected_lcb = 1.0 - (z * z) / (n + z * z)
        self.assertAlmostEqual(res.lcb, expected_lcb, places=6)

    def test_ordinary_intermediate_qber(self):
        """Test realistic intermediate QBER: k = 25, n = 500 (p_hat = 0.05)."""
        n = 500
        k = 25
        p_hat = k / n  # 0.05
        res = compute_wilson_score_interval(k=k, n=n, confidence=0.95)
        self.assertLess(res.lcb, p_hat)
        self.assertGreater(res.ucb, p_hat)
        self.assertTrue(0.03 <= res.lcb <= 0.04)
        self.assertTrue(0.06 <= res.ucb <= 0.08)
        self.assertAlmostEqual(res.margin, (res.ucb - res.lcb) / 2.0, places=6)

    def test_small_sample_sizes(self):
        """Test small sample sizes: n = 5, 10, 20."""
        for n in [5, 10, 20]:
            for k in range(n + 1):
                res = compute_wilson_score_interval(k=k, n=n, confidence=0.95)
                self.assertGreaterEqual(res.lcb, 0.0)
                self.assertLessEqual(res.ucb, 1.0)
                self.assertLessEqual(res.lcb, res.ucb)


class TestFeasibilityClassification(unittest.TestCase):
    """Test suite for three-state statistical feasibility classification."""

    def test_statistically_feasible(self):
        """Verify Category A: Q_UCB <= Q_budget."""
        status = classify_feasibility(q_lcb=0.02, q_ucb=0.045, q_budget=0.05)
        self.assertEqual(status, StatisticalFeasibility.FEASIBLE)

    def test_boundary_feasible(self):
        """Verify feasibility on exact equality Q_UCB == Q_budget."""
        status = classify_feasibility(q_lcb=0.02, q_ucb=0.05, q_budget=0.05)
        self.assertEqual(status, StatisticalFeasibility.FEASIBLE)

    def test_statistically_infeasible(self):
        """Verify Category B: Q_LCB > Q_budget."""
        status = classify_feasibility(q_lcb=0.055, q_ucb=0.085, q_budget=0.05)
        self.assertEqual(status, StatisticalFeasibility.INFEASIBLE)

    def test_statistically_indeterminate(self):
        """Verify Category C: Q_LCB <= Q_budget < Q_UCB."""
        status = classify_feasibility(q_lcb=0.035, q_ucb=0.065, q_budget=0.05)
        self.assertEqual(status, StatisticalFeasibility.INDETERMINATE)

    def test_invalid_feasibility_inputs(self):
        """Verify proper validation exceptions for feasibility classifier."""
        with self.assertRaises(ValueError):
            classify_feasibility(q_lcb=0.06, q_ucb=0.04, q_budget=0.05)  # lcb > ucb
        with self.assertRaises(ValueError):
            classify_feasibility(q_lcb=-0.01, q_ucb=0.05, q_budget=0.05)  # lcb < 0
        with self.assertRaises(ValueError):
            classify_feasibility(q_lcb=0.02, q_ucb=1.05, q_budget=0.05)  # ucb > 1

    def test_empirical_frequencies(self):
        """Verify repeated-trial empirical pass/abort frequency computation."""
        # 10 trials: 8 pass, 2 abort
        outcomes = [False, False, True, False, False, True, False, False, False, False]
        stats = compute_empirical_frequencies(outcomes, confidence=0.95)
        self.assertEqual(stats.n_trials, 10)
        self.assertEqual(stats.n_pass, 8)
        self.assertEqual(stats.n_abort, 2)
        self.assertAlmostEqual(stats.pass_frequency, 0.80)
        self.assertAlmostEqual(stats.abort_frequency, 0.20)
        self.assertLessEqual(stats.pass_lcb, 0.80)
        self.assertGreaterEqual(stats.pass_ucb, 0.80)


class TestAttackPolicyGenerator(unittest.TestCase):
    """Test suite for deterministic attack configuration generation and instantiation."""

    def test_all_16_configurations_present(self):
        """Verify that exactly 16 configurations ATT-00 through ATT-15 exist."""
        configs = get_all_attack_configurations()
        self.assertEqual(len(configs), 16)
        ids = [c.attack_id for c in configs]
        expected_ids = [f"ATT-{i:02d}" for i in range(16)]
        self.assertEqual(ids, expected_ids)

    def test_family_distribution(self):
        """Verify family counts: 1 Null, 7 Probe, 8 PIR (6 symmetric, 2 biased)."""
        configs = get_all_attack_configurations()
        null_count = sum(1 for c in configs if c.family == AttackFamily.NULL)
        probe_count = sum(1 for c in configs if c.family == AttackFamily.UNITARY_PROBE)
        pir_count = sum(
            1 for c in configs if c.family == AttackFamily.PROBABILISTIC_INTERCEPT_RESEND
        )

        self.assertEqual(null_count, 1)
        self.assertEqual(probe_count, 7)
        self.assertEqual(pir_count, 8)

    def test_unitary_probe_parameters(self):
        """Verify Unitary Probe configurations ATT-01 to ATT-07."""
        expected_thetas = [0.0500, 0.1000, 0.1500, 0.2000, 0.2500, 0.3000, 0.3398]
        for i, exp_th in enumerate(expected_thetas, start=1):
            cfg = get_attack_configuration(f"ATT-{i:02d}")
            self.assertEqual(cfg.family, AttackFamily.UNITARY_PROBE)
            self.assertAlmostEqual(cfg.parameters.theta_E_rad, exp_th, places=4)
            self.assertIsNotNone(cfg.parameters.theta_E_deg)
            # Theoretical disturbance sin^2(theta)
            exp_d = math.sin(exp_th) ** 2
            self.assertAlmostEqual(cfg.theoretical_disturbance, exp_d, places=5)

    def test_pir_parameters(self):
        """Verify PIR configurations ATT-08 through ATT-15."""
        # Symmetric cases: ATT-08 to ATT-13
        expected_alphas = [0.05, 0.10, 0.20, 0.30, 0.40, 0.44]
        for i, exp_a in enumerate(expected_alphas, start=8):
            cfg = get_attack_configuration(f"ATT-{i:02d}")
            self.assertEqual(cfg.family, AttackFamily.PROBABILISTIC_INTERCEPT_RESEND)
            self.assertAlmostEqual(cfg.parameters.alpha, exp_a)
            self.assertAlmostEqual(cfg.parameters.beta, 0.50)
            self.assertAlmostEqual(cfg.theoretical_disturbance, exp_a / 4.0)

        # Biased cases: ATT-14 and ATT-15
        att14 = get_attack_configuration("ATT-14")
        self.assertAlmostEqual(att14.parameters.alpha, 0.20)
        self.assertAlmostEqual(att14.parameters.beta, 0.30)

        att15 = get_attack_configuration("ATT-15")
        self.assertAlmostEqual(att15.parameters.alpha, 0.20)
        self.assertAlmostEqual(att15.parameters.beta, 0.70)

    def test_attack_instantiation(self):
        """Verify class instantiation from attack configurations."""
        # Null attack
        null_cfg = get_attack_configuration("ATT-00")
        self.assertIsNone(instantiate_attack(null_cfg))

        # Probe attack
        probe_cfg = get_attack_configuration("ATT-04")
        probe_inst = instantiate_attack(probe_cfg)
        self.assertIsInstance(probe_inst, FuchsPeresBrandtProbe)
        self.assertAlmostEqual(probe_inst.theta_E, 0.20)

        # PIR attack
        pir_cfg = get_attack_configuration("ATT-10")
        pir_inst = instantiate_attack(pir_cfg, seed=42)
        self.assertIsInstance(pir_inst, InterceptResendAttack)
        self.assertAlmostEqual(pir_inst.alpha, 0.20)
        self.assertAlmostEqual(pir_inst.beta, 0.50)


class TestDatasetSchemaSerialization(unittest.TestCase):
    """Test suite for dataset schema serialization and deserialization."""

    def test_environmental_condition_roundtrip(self):
        """Verify EnvironmentalCondition serialization roundtrip."""
        cond = generate_environmental_condition("C01", p_depol=0.04, q_budget=0.08, seed_calib=42)
        c_dict = cond.to_dict()
        reconstructed = EnvironmentalCondition.from_dict(c_dict)
        self.assertEqual(cond, reconstructed)

    def test_attack_configuration_roundtrip(self):
        """Verify AttackConfiguration serialization roundtrip."""
        cfg = get_attack_configuration("ATT-04")
        cfg_dict = cfg.to_dict()
        reconstructed = AttackConfiguration.from_dict(cfg_dict)
        self.assertEqual(cfg, reconstructed)

    def test_dataset_record_roundtrip(self):
        """Verify DatasetRecord serialization and deserialization."""
        cond = generate_environmental_condition("C01", p_depol=0.02, q_budget=0.05, seed_calib=42)
        cfg = get_attack_configuration("ATT-00")

        # Run minimal simulation (100 pulses) to get realistic record
        record = execute_single_simulation(cond, cfg, seed_sim=101, n_pulses=100)
        rec_dict = record.to_dict()
        reconstructed = DatasetRecord.from_dict(rec_dict)

        self.assertEqual(record.sample_id, reconstructed.sample_id)
        self.assertEqual(record.condition_id, reconstructed.condition_id)
        self.assertEqual(record.attack_id, reconstructed.attack_id)
        self.assertEqual(record.attack_family, reconstructed.attack_family)
        self.assertEqual(record.raw_observation, reconstructed.raw_observation)
        self.assertEqual(record.derived_evaluation, reconstructed.derived_evaluation)

    def test_dataset_container_json_io(self):
        """Verify DatasetContainer save_json and load_json integrity."""
        cond = generate_environmental_condition("C01", p_depol=0.00, q_budget=0.05, seed_calib=42)
        cfg = get_attack_configuration("ATT-00")
        record = execute_single_simulation(cond, cfg, seed_sim=101, n_pulses=100)

        container = DatasetContainer(
            schema_version="1.0.0",
            dataset_type="test",
            created_at_utc="2026-11-10T12:00:00Z",
            total_records=1,
            total_conditions=1,
            total_attacks=1,
            total_seeds=1,
            records=[record],
            hardware_info={"backend": "AerSimulator"},
            summary_statistics={"yield": 1.0},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            container.save_json(tmp_path)
            loaded = DatasetContainer.load_json(tmp_path)

            self.assertEqual(container.schema_version, loaded.schema_version)
            self.assertEqual(container.total_records, loaded.total_records)
            self.assertEqual(len(loaded.records), 1)
            self.assertEqual(loaded.records[0].sample_id, record.sample_id)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestBB84DatasetIntegration(unittest.TestCase):
    """Test suite for BB84 simulator integration and statistical evaluation."""

    def test_single_simulation_bit_partitioning(self):
        """Verify that N_test + N_key == N_sift and N_test_Z + N_test_X == N_test."""
        cond = generate_environmental_condition("C01", p_depol=0.02, q_budget=0.05, seed_calib=42)
        cfg = get_attack_configuration("ATT-08")  # PIR alpha=0.05
        rec = execute_single_simulation(cond, cfg, seed_sim=101, n_pulses=500, rho_test=0.50)

        raw = rec.raw_observation
        self.assertEqual(raw.n_transmitted_pulses, 500)
        self.assertGreater(raw.n_sifted, 150)
        self.assertEqual(raw.n_test + raw.n_key, raw.n_sifted)
        self.assertEqual(raw.sifted_count_z_test + raw.sifted_count_x_test, raw.n_test)
        self.assertEqual(raw.error_count_z_test + raw.error_count_x_test, raw.error_count_test)

    def test_protocol_abort_rule_consistency(self):
        """Verify that protocol_abort is True if and only if qber_test > q_budget."""
        # Condition with very low budget (0.01) and high noise (0.10) => should abort
        cond_abort = generate_environmental_condition("C_HIGH", p_depol=0.10, q_budget=0.01, seed_calib=42)
        cfg_probe = get_attack_configuration("ATT-06")  # theta = 0.30 => D ~ 8.7%
        rec_abort = execute_single_simulation(cond_abort, cfg_probe, seed_sim=101, n_pulses=500)
        self.assertTrue(rec_abort.raw_observation.protocol_abort)

        # Condition with zero noise, zero attack, high budget (0.11) => should pass
        cond_pass = generate_environmental_condition("C_PASS", p_depol=0.00, q_budget=0.11, seed_calib=42)
        cfg_null = get_attack_configuration("ATT-00")
        rec_pass = execute_single_simulation(cond_pass, cfg_null, seed_sim=101, n_pulses=500)
        self.assertFalse(rec_pass.raw_observation.protocol_abort)
        self.assertEqual(rec_pass.raw_observation.qber_test, 0.0)
        self.assertEqual(rec_pass.derived_evaluation.feasibility_category, StatisticalFeasibility.FEASIBLE)

    def test_deterministic_seed_reproducibility(self):
        """Verify that identical simulation seeds produce bit-identical records."""
        cond = generate_environmental_condition("C01", p_depol=0.04, q_budget=0.08, seed_calib=42)
        cfg = get_attack_configuration("ATT-04")

        rec1 = execute_single_simulation(cond, cfg, seed_sim=202, n_pulses=400)
        rec2 = execute_single_simulation(cond, cfg, seed_sim=202, n_pulses=400)

        self.assertEqual(rec1.raw_observation.qber_test, rec2.raw_observation.qber_test)
        self.assertEqual(rec1.raw_observation.error_count_test, rec2.raw_observation.error_count_test)
        self.assertEqual(rec1.derived_evaluation.wilson_q_ucb, rec2.derived_evaluation.wilson_q_ucb)
        self.assertEqual(rec1.derived_evaluation.feasibility_category, rec2.derived_evaluation.feasibility_category)

    def test_mini_dataset_generation(self):
        """Verify small dataset generator execution (1 condition, 2 attacks, 2 seeds = 4 runs)."""
        cond = generate_environmental_condition("C_MINI", p_depol=0.02, q_budget=0.05, seed_calib=42)
        attacks = [get_attack_configuration("ATT-00"), get_attack_configuration("ATT-01")]
        seeds = [301, 302]

        dataset = generate_dataset(
            conditions=[cond],
            attacks=attacks,
            seeds=seeds,
            n_pulses=200,
            dataset_type="test_mini",
        )

        self.assertEqual(dataset.total_records, 4)
        self.assertEqual(len(dataset.records), 4)
        self.assertIn("feasibility_counts", dataset.summary_statistics)
        self.assertIn("family_breakdown", dataset.summary_statistics)


class TestCalibrationProvenanceAndInformationBoundary(unittest.TestCase):
    """Test suite verifying calibration provenance and Eve's observable information boundary."""

    def test_information_boundary_hidden_noise_invariance(self):
        """Test 1: Changing hidden ground-truth noise does not directly change generator input when observable calibration is held fixed."""
        # Condition A: p_depol = 0.04, gamma_noise_hat fixed at 0.02
        cond_a = EnvironmentalCondition(
            condition_id="C_test_a",
            p_depol=0.04,
            q_env=0.02,
            q_budget=0.08,
            gamma_noise_hat=0.02,
            sigma_calib=0.0044,
            n_calib=1000,
            k_calib=20,
            seed_calib=123,
        )
        # Condition B: Different ground truth p_depol = 0.08, but observable gamma_noise_hat identical at 0.02
        cond_b = EnvironmentalCondition(
            condition_id="C_test_b",
            p_depol=0.08,
            q_env=0.04,
            q_budget=0.08,
            gamma_noise_hat=0.02,
            sigma_calib=0.0044,
            n_calib=1000,
            k_calib=20,
            seed_calib=123,
        )

        c_a = cond_a.get_observable_condition()
        c_b = cond_b.get_observable_condition()

        # Generator input c = [Q_budget, gamma_noise_hat] is strictly identical
        self.assertEqual(c_a, c_b)
        self.assertEqual(c_a, (0.08, 0.02))
        # Ground truths differ
        self.assertNotEqual(cond_a.p_depol, cond_b.p_depol)
        self.assertNotEqual(cond_a.q_env, cond_b.q_env)

    def test_generator_receives_stored_calibration_estimate(self):
        """Test 2: The generator input vector receives the stored calibration estimate."""
        for q_bud in [0.05, 0.08, 0.11]:
            for g_hat in [0.00, 0.015, 0.029]:
                cond = EnvironmentalCondition(
                    condition_id="C_test",
                    p_depol=0.06,
                    q_env=0.03,
                    q_budget=q_bud,
                    gamma_noise_hat=g_hat,
                    sigma_calib=0.005,
                    n_calib=1000,
                    k_calib=int(round(g_hat * 1000)),
                )
                obs_c = cond.get_observable_condition()
                self.assertEqual(obs_c, (q_bud, g_hat))
                self.assertNotIn(cond.p_depol, obs_c)

    def test_calibration_counts_estimates_and_errors_consistency(self):
        """Test 3: Calibration counts, estimates, and standard errors are internally consistent."""
        conditions = get_pilot_environmental_conditions()
        for cond in conditions:
            # 1. k_calib in [0, n_calib]
            self.assertTrue(0 <= cond.k_calib <= cond.n_calib)
            # 2. gamma_noise_hat == k_calib / n_calib
            expected_gamma = cond.k_calib / float(cond.n_calib)
            self.assertAlmostEqual(cond.gamma_noise_hat, expected_gamma, places=6)
            # 3. sigma_calib consistency: sqrt(q_env * (1 - q_env) / n_calib)
            if cond.q_env > 0:
                expected_sigma = math.sqrt(cond.q_env * (1.0 - cond.q_env) / cond.n_calib)
                self.assertAlmostEqual(cond.sigma_calib, expected_sigma, places=6)
            else:
                self.assertEqual(cond.sigma_calib, 0.0)
            # 4. Plug-in standard error property: sqrt(gamma_hat * (1 - gamma_hat) / n_calib)
            if cond.gamma_noise_hat > 0:
                expected_sigma_est = math.sqrt(cond.gamma_noise_hat * (1.0 - cond.gamma_noise_hat) / cond.n_calib)
                self.assertAlmostEqual(cond.sigma_estimated, expected_sigma_est, places=6)
            else:
                self.assertEqual(cond.sigma_estimated, 0.0)
            # 5. Provenance seed is retained
            self.assertIsNotNone(cond.seed_calib)


if __name__ == "__main__":
    unittest.main()

