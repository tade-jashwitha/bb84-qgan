"""Unit and regression tests for baseline benchmark engines in BB84 QKD.

Verifies:
1. AnalyticalProbePolicyEngine:
   - Exact mathematical inverse mapping theta_E* = arcsin(sqrt(D*)) for Fuchs-Peres-Brandt probe.
   - Physical disturbance matching: D(theta_E*) == D* under U_E(theta_E).
   - Compounding noise model: D* = max(0, (Q_budget - gamma_hat) / (1 - 2*gamma_hat)).
   - Parameter clamping and domain boundaries: [0.0, pi/4].
   - Numerical robustness under extreme/singular inputs (gamma >= 0.5, Q_budget <= gamma).
   - Candidate proposal structure and K=16 query budget constraints.
2. UniformRandomPolicyEngine:
   - Valid bounded parameter generation across PIR and Probe families.
3. GridSearchPolicyEngine:
   - Exact 16-lattice configuration inventory.
"""

import math
import os
import sys
import unittest
from typing import List

import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.benchmark_baselines import (
    AnalyticalProbePolicyEngine,
    GridSearchPolicyEngine,
    UniformRandomPolicyEngine,
)
from src.bb84.attacks import FuchsPeresBrandtProbe
from src.dataset.generator import get_full_environmental_conditions
from src.dataset.schema import AttackFamily, EnvironmentalCondition


class TestAnalyticalProbePolicyEngine(unittest.TestCase):
    """Test suite for AnalyticalProbePolicyEngine mathematical correctness and boundary safety."""

    def setUp(self):
        self.engine = AnalyticalProbePolicyEngine()

    def test_case_01_exact_inverse_mapping_fuchs_peres_brandt(self):
        """Case 1: Optimal angle theta* exactly inverts Fuchs-Peres disturbance D(theta) = sin^2(theta)."""
        test_budgets = [0.03, 0.05, 0.08, 0.10, 0.11]
        noise_rates = [0.00, 0.01, 0.02, 0.03]

        for q_bud in test_budgets:
            for gamma in noise_rates:
                if q_bud <= gamma:
                    continue
                theta_star, d_target = self.engine.compute_optimal_angle(q_bud, gamma)
                probe = FuchsPeresBrandtProbe(theta_E=theta_star)
                sim_disturbance = probe.theoretical_disturbance()

                # Verify D(theta*) matches target disturbance D*
                expected_d = min(0.5, d_target)
                self.assertAlmostEqual(
                    sim_disturbance,
                    expected_d,
                    places=5,
                    msg=f"Disturbance mismatch for Q_budget={q_bud}, gamma={gamma}: "
                    f"sim={sim_disturbance:.6f}, expected={expected_d:.6f}",
                )

                # Verify theta* is exactly arcsin(sqrt(D*))
                expected_theta = math.asin(math.sqrt(expected_d))
                self.assertAlmostEqual(
                    theta_star,
                    expected_theta,
                    places=6,
                    msg=f"Angle mismatch for Q_budget={q_bud}, gamma={gamma}",
                )

    def test_case_02_compounding_error_model(self):
        """Case 2: Target disturbance D* satisfies total compounded QBER == Q_budget."""
        q_bud = 0.08
        gamma_hat = 0.02
        theta_star, d_target = self.engine.compute_optimal_angle(q_bud, gamma_hat)

        # Expected compounding: Q_tot = gamma + D*(1 - 2*gamma)
        compounded_qber = gamma_hat + d_target * (1.0 - 2.0 * gamma_hat)
        self.assertAlmostEqual(compounded_qber, q_bud, places=6)

    def test_case_03_angle_monotonicity(self):
        """Case 3: theta* increases with Q_budget and decreases with ambient noise."""
        gamma_fixed = 0.02
        budgets = [0.03, 0.05, 0.08, 0.11]
        thetas_bud = [self.engine.compute_optimal_angle(q, gamma_fixed)[0] for q in budgets]
        for i in range(len(thetas_bud) - 1):
            self.assertLess(thetas_bud[i], thetas_bud[i + 1])

        q_fixed = 0.08
        noises = [0.00, 0.01, 0.03, 0.05, 0.07]
        thetas_noise = [self.engine.compute_optimal_angle(q_fixed, g)[0] for g in noises]
        for i in range(len(thetas_noise) - 1):
            self.assertGreater(thetas_noise[i], thetas_noise[i + 1])

    def test_case_04_boundary_zero_disturbance(self):
        """Case 4: D* and theta* clamp to exactly 0 when ambient noise equals or exceeds budget."""
        # Noise equals budget
        theta, d = self.engine.compute_optimal_angle(q_budget=0.05, gamma_noise_hat=0.05)
        self.assertEqual(theta, 0.0)
        self.assertEqual(d, 0.0)

        # Noise exceeds budget
        theta, d = self.engine.compute_optimal_angle(q_budget=0.05, gamma_noise_hat=0.08)
        self.assertEqual(theta, 0.0)
        self.assertEqual(d, 0.0)

    def test_case_05_upper_bound_clamping(self):
        """Case 5: theta* is strictly clamped to [0, pi/4] for extreme/unphysical budgets."""
        # Infinite / maximal budget
        theta, _ = self.engine.compute_optimal_angle(q_budget=1.0, gamma_noise_hat=0.0)
        self.assertAlmostEqual(theta, math.pi / 4.0, places=6)

    def test_case_06_singular_noise_denominator_guard(self):
        """Case 6: High noise (gamma >= 0.5) does not raise ZeroDivisionError or negative denom."""
        theta, d = self.engine.compute_optimal_angle(q_budget=0.05, gamma_noise_hat=0.50)
        self.assertEqual(theta, 0.0)
        self.assertEqual(d, 0.0)

        theta2, d2 = self.engine.compute_optimal_angle(q_budget=0.05, gamma_noise_hat=0.60)
        self.assertEqual(theta2, 0.0)
        self.assertEqual(d2, 0.0)

    def test_case_07_candidate_proposals_structure(self):
        """Case 7: Propose exactly K=16 valid AttackConfigurations within [0, pi/4]."""
        all_conds = get_full_environmental_conditions()
        cond = next(c for c in all_conds if c.condition_id == "C21")
        candidates = self.engine.propose_candidate_configurations(cond, num_candidates=16)

        self.assertEqual(len(candidates), 16)
        for cand in candidates:
            self.assertEqual(cand.family, AttackFamily.UNITARY_PROBE)
            self.assertIsNotNone(cand.parameters.theta_E_rad)
            theta = cand.parameters.theta_E_rad
            self.assertTrue(0.0 <= theta <= math.pi / 4.0 + 1e-9)
            # Verify theoretical disturbance matches sin^2(theta)
            self.assertAlmostEqual(cand.theoretical_disturbance, math.sin(theta) ** 2, places=6)

    def test_case_08_held_out_test_conditions_analytical_targets(self):
        """Case 8: Verify analytical angle targets for held-out Test Conditions C21 to C24."""
        test_conds = [
            ("C21", 0.05, 0.025, 0.05),  # p=0.05, q_env=0.025, Q_budget=0.05
            ("C22", 0.05, 0.025, 0.11),  # p=0.05, q_env=0.025, Q_budget=0.11
            ("C23", 0.07, 0.035, 0.05),  # p=0.07, q_env=0.035, Q_budget=0.05
            ("C24", 0.07, 0.035, 0.11),  # p=0.07, q_env=0.035, Q_budget=0.11
        ]
        for cid, p_dep, q_env, q_bud in test_conds:
            theta, d = self.engine.compute_optimal_angle(q_budget=q_bud, gamma_noise_hat=q_env)
            self.assertTrue(0.0 <= theta <= math.pi / 4.0)
            self.assertTrue(0.0 <= d <= 0.5)
            # Check induced disturbance matches d
            self.assertAlmostEqual(math.sin(theta) ** 2, min(0.5, d), places=6)


class TestOtherBenchmarkEngines(unittest.TestCase):
    """Test suite for UniformRandomPolicyEngine and GridSearchPolicyEngine."""

    def test_case_09_uniform_random_proposals(self):
        """Case 9: UniformRandomPolicyEngine generates valid candidates across PIR and Probe."""
        engine = UniformRandomPolicyEngine(seed=42)
        all_conds = get_full_environmental_conditions()
        cond = next(c for c in all_conds if c.condition_id == "C22")
        candidates = engine.propose_candidate_configurations(cond, num_candidates=16, seed=123)
        self.assertEqual(len(candidates), 16)
        for cand in candidates:
            if cand.family == AttackFamily.PROBABILISTIC_INTERCEPT_RESEND:
                self.assertTrue(0.0 <= cand.parameters.alpha <= 1.0)
                self.assertTrue(0.0 <= cand.parameters.beta <= 1.0)
            elif cand.family == AttackFamily.UNITARY_PROBE:
                self.assertTrue(0.0 <= cand.parameters.theta_E_rad <= math.pi / 4.0)

    def test_case_10_grid_search_proposals(self):
        """Case 10: GridSearchPolicyEngine proposes exactly 16 lattice configurations."""
        engine = GridSearchPolicyEngine()
        all_conds = get_full_environmental_conditions()
        cond = next(c for c in all_conds if c.condition_id == "C23")
        candidates = engine.propose_candidate_configurations(cond, num_candidates=16)
        self.assertEqual(len(candidates), 16)
        attack_ids = [c.attack_id for c in candidates]
        expected_ids = [f"ATT-{i:02d}" for i in range(16)]
        self.assertEqual(attack_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
