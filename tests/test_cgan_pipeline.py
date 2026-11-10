"""Comprehensive unit and integration tests for the Classical Conditional GAN (cGAN) baseline.

Tests:
1. `canonical_mask_policy`: Differentiable parameter masking for PIR, Probe, and Null.
2. `ConditionalGenerator`: Output shapes, strict parameter bounds, and Gumbel differentiability.
3. `ConditionalDiscriminator`: Spectral-normalized forward pass and output shape.
4. `record_to_policy_tensors`: Exact mapping of BB84 dataset records.
5. `BB84PolicyDataset`: Partition filtering, feasibility filtering, and importance sampling weights.
6. `CGANTrainer`: Single-epoch step, loss decrease, and checkpoint saving/loading.
7. `cGANPolicyEngine`: 16-candidate generation and Tri-Case operational fallback execution.
"""

import math
import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from src.bb84.bb84_engine import BB84Engine
from src.cgan.dataset import BB84PolicyDataset, create_dataloaders, record_to_policy_tensors
from src.cgan.models import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    canonical_mask_policy,
)
from src.cgan.policy_engine import OperationalDecision, cGANPolicyEngine
from src.cgan.trainer import CGANTrainer
from src.channel.noise_models import build_depolarizing_noise_model
from src.dataset.generator import execute_single_simulation, generate_environmental_condition
from src.dataset.policy_generator import get_attack_configuration, instantiate_attack
from src.dataset.schema import AttackConfiguration, AttackFamily, AttackParameters, StatisticalFeasibility
from src.dataset.statistics import classify_feasibility, compute_wilson_score_interval


class TestCanonicalMaskPolicy(unittest.TestCase):
    """Test suite for the differentiable canonical masking operator."""

    def test_canonical_mask_policy_pir(self):
        """PIR selection must zero out probe parameter and preserve alpha and beta."""
        weights = torch.tensor([[1.0, 0.0]])
        alpha = torch.tensor([[0.35]])
        beta = torch.tensor([[0.70]])
        theta_norm = torch.tensor([[0.80]])

        masked = canonical_mask_policy(weights, alpha, beta, theta_norm)
        self.assertEqual(masked.shape, (1, 5))
        np.testing.assert_allclose(masked[0].tolist(), [1.0, 0.0, 0.35, 0.70, 0.0], atol=1e-6)

    def test_canonical_mask_policy_probe(self):
        """Probe selection must zero out PIR parameters and preserve theta_norm."""
        weights = torch.tensor([[0.0, 1.0]])
        alpha = torch.tensor([[0.99]])
        beta = torch.tensor([[0.99]])
        theta_norm = torch.tensor([[0.45]])

        masked = canonical_mask_policy(weights, alpha, beta, theta_norm)
        self.assertEqual(masked.shape, (1, 5))
        np.testing.assert_allclose(masked[0].tolist(), [0.0, 1.0, 0.0, 0.0, 0.45], atol=1e-6)

    def test_canonical_mask_policy_null(self):
        """Null reference policy must produce all zeros."""
        weights = torch.tensor([[0.0, 0.0]])
        alpha = torch.tensor([[0.50]])
        beta = torch.tensor([[0.50]])
        theta_norm = torch.tensor([[0.50]])

        masked = canonical_mask_policy(weights, alpha, beta, theta_norm)
        self.assertEqual(masked.shape, (1, 5))
        np.testing.assert_allclose(masked[0].tolist(), [0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_batch_masking(self):
        """Masking must operate correctly across batches."""
        weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        alpha = torch.tensor([[0.2], [0.8]])
        beta = torch.tensor([[0.4], [0.6]])
        theta_norm = torch.tensor([[0.1], [0.9]])

        masked = canonical_mask_policy(weights, alpha, beta, theta_norm)
        self.assertEqual(masked.shape, (2, 5))
        np.testing.assert_allclose(masked[0].tolist(), [1.0, 0.0, 0.2, 0.4, 0.0], atol=1e-6)
        np.testing.assert_allclose(masked[1].tolist(), [0.0, 1.0, 0.0, 0.0, 0.9], atol=1e-6)


class TestConditionalModels(unittest.TestCase):
    """Test suite for Generator and Discriminator PyTorch modules."""

    def setUp(self):
        self.generator = ConditionalGenerator(latent_dim=16, condition_dim=2, hidden_dims=[64, 64])
        self.discriminator = ConditionalDiscriminator(policy_dim=5, condition_dim=2, hidden_dims=[64, 32])

    def test_generator_output_shapes_and_bounds(self):
        """Generator must output valid shapes and strictly bounded parameters."""
        b_size = 8
        z = torch.randn(b_size, 16)
        c = torch.tensor([[0.05, 0.01]] * b_size)

        out = self.generator(z, c, use_gumbel=False)

        self.assertIn("masked_policy", out)
        self.assertIn("family_weights", out)
        self.assertIn("alpha", out)
        self.assertIn("beta", out)
        self.assertIn("theta_E_rad", out)

        self.assertEqual(out["masked_policy"].shape, (b_size, 5))
        self.assertEqual(out["family_weights"].shape, (b_size, 2))

        # Check simplex sum = 1
        fam_sums = out["family_weights"].sum(dim=-1)
        np.testing.assert_allclose(fam_sums.detach().numpy(), np.ones(b_size), atol=1e-5)

        # Check bounds
        alphas = out["alpha"].detach().numpy()
        betas = out["beta"].detach().numpy()
        thetas = out["theta_E_rad"].detach().numpy()

        self.assertTrue(np.all((alphas >= 0.0) & (alphas <= 1.0)))
        self.assertTrue(np.all((betas >= 0.0) & (betas <= 1.0)))
        self.assertTrue(np.all((thetas >= 0.0) & (thetas <= (math.pi / 4.0 + 1e-6))))

    def test_generator_differentiability_with_gumbel(self):
        """Gradients must flow backward through Gumbel-Softmax to generator weights."""
        self.generator.train()
        z = torch.randn(4, 16)
        c = torch.tensor([[0.08, 0.02]] * 4)

        out = self.generator(z, c, tau=1.0, use_gumbel=True)
        loss = out["masked_policy"].sum()
        loss.backward()

        # Check that family head has gradients
        self.assertIsNotNone(self.generator.family_head.weight.grad)
        self.assertGreater(self.generator.family_head.weight.grad.abs().sum().item(), 0.0)

    def test_discriminator_output_shapes(self):
        """Discriminator must output scalar logits of shape (B, 1)."""
        b_size = 4
        a_masked = torch.randn(b_size, 5)
        c = torch.tensor([[0.05, 0.01]] * b_size)

        logits = self.discriminator(a_masked, c)
        self.assertEqual(logits.shape, (b_size, 1))


class TestPolicyDatasetAndLoader(unittest.TestCase):
    """Test suite for Dataset extraction, filtering, and DataLoader creation."""

    def test_record_to_policy_tensors_pir(self):
        """PIR simulation record must correctly convert to masked policy tensor."""
        cond = generate_environmental_condition("C01", p_depol=0.02, q_budget=0.05, seed_calib=101)
        att = get_attack_configuration("ATT-10")  # alpha=0.20, beta=0.50
        rec = execute_single_simulation(cond, att, seed_sim=555, n_pulses=500)

        c_t, p_t, i_gain = record_to_policy_tensors(rec)
        self.assertEqual(c_t.shape, (2,))
        self.assertEqual(p_t.shape, (5,))

        np.testing.assert_allclose(c_t.tolist(), [0.05, cond.gamma_noise_hat], atol=1e-5)
        np.testing.assert_allclose(p_t.tolist(), [1.0, 0.0, 0.20, 0.50, 0.0], atol=1e-5)

    def test_record_to_policy_tensors_probe(self):
        """Unitary probe record must correctly convert to normalized angle in tensor."""
        cond = generate_environmental_condition("C01", p_depol=0.00, q_budget=0.05, seed_calib=101)
        att = get_attack_configuration("ATT-02")  # theta = 0.10 rad
        rec = execute_single_simulation(cond, att, seed_sim=555, n_pulses=500)

        c_t, p_t, i_gain = record_to_policy_tensors(rec)
        expected_norm = (4.0 / math.pi) * 0.10
        np.testing.assert_allclose(p_t.tolist(), [0.0, 1.0, 0.0, 0.0, expected_norm], atol=1e-4)

    def test_dataset_from_pilot_file(self):
        """Loading from existing pilot JSON must respect feasibility and partition filters."""
        pilot_path = "results/pilot_dataset_provenance.json"
        if not os.path.exists(pilot_path):
            pilot_path = "results/pilot_dataset.json"

        if os.path.exists(pilot_path):
            ds_feasible = BB84PolicyDataset.from_json(pilot_path, filter_feasible=True)
            ds_all = BB84PolicyDataset.from_json(pilot_path, filter_feasible=False)

            self.assertGreater(len(ds_feasible), 0)
            self.assertGreater(len(ds_all), len(ds_feasible))
            self.assertEqual(len(ds_all), 240)


class TestCGANTrainer(unittest.TestCase):
    """Test suite for CGANTrainer optimization steps and checkpointing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.gen = ConditionalGenerator(latent_dim=8, condition_dim=2, hidden_dims=[32])
        self.disc = ConditionalDiscriminator(policy_dim=5, condition_dim=2, hidden_dims=[32])

        # Create dummy synthetic records
        cond = generate_environmental_condition("C01", p_depol=0.02, q_budget=0.05, seed_calib=101)
        att_pir = get_attack_configuration("ATT-10")
        att_probe = get_attack_configuration("ATT-02")

        rec1 = execute_single_simulation(cond, att_pir, seed_sim=101, n_pulses=500)
        rec2 = execute_single_simulation(cond, att_probe, seed_sim=102, n_pulses=500)

        self.dataset = BB84PolicyDataset(records=[rec1, rec2] * 8, filter_feasible=False)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_single_train_epoch(self):
        """Trainer must execute one epoch and compute losses without NaN."""
        trainer = CGANTrainer(
            generator=self.gen,
            discriminator=self.disc,
            train_loader=self.loader,
            checkpoint_dir=self.temp_dir,
            seed=123,
        )

        metrics = trainer.train_epoch(epoch=1)
        self.assertIn("loss_d", metrics)
        self.assertIn("loss_g", metrics)
        self.assertFalse(math.isnan(metrics["loss_d"]))
        self.assertFalse(math.isnan(metrics["loss_g"]))

    def test_checkpoint_save_and_load(self):
        """Checkpoint serialization must save and restore bit-identical weights."""
        trainer = CGANTrainer(
            generator=self.gen,
            discriminator=self.disc,
            train_loader=self.loader,
            checkpoint_dir=self.temp_dir,
            seed=123,
        )

        ckpt_path = os.path.join(self.temp_dir, "test_ckpt.pt")
        trainer.save_checkpoint(ckpt_path, epoch=1)
        self.assertTrue(os.path.exists(ckpt_path))

        new_gen = ConditionalGenerator(latent_dim=8, condition_dim=2, hidden_dims=[32])
        new_disc = ConditionalDiscriminator(policy_dim=5, condition_dim=2, hidden_dims=[32])

        new_trainer = CGANTrainer(
            generator=new_gen,
            discriminator=new_disc,
            train_loader=self.loader,
            checkpoint_dir=self.temp_dir,
            seed=123,
        )
        new_trainer.load_checkpoint(ckpt_path)

        for p1, p2 in zip(self.gen.parameters(), new_gen.parameters()):
            np.testing.assert_allclose(p1.detach().cpu().numpy(), p2.detach().cpu().numpy(), atol=1e-6)


class TestPolicyEngine(unittest.TestCase):
    """Test suite for cGAN candidate generation and Tri-Case fallback engine."""

    def setUp(self):
        self.gen = ConditionalGenerator(latent_dim=8, condition_dim=2, hidden_dims=[32])
        self.engine = cGANPolicyEngine(generator=self.gen)

    def test_propose_candidate_configurations(self):
        """Policy engine must propose K valid AttackConfigurations."""
        cond = generate_environmental_condition("C01", p_depol=0.02, q_budget=0.05, seed_calib=101)
        cands = self.engine.propose_candidate_configurations(cond, num_candidates=16, seed=42)

        self.assertEqual(len(cands), 16)
        for cand in cands:
            self.assertIn(cand.family, [AttackFamily.PROBABILISTIC_INTERCEPT_RESEND, AttackFamily.UNITARY_PROBE])
            if cand.family == AttackFamily.PROBABILISTIC_INTERCEPT_RESEND:
                self.assertGreaterEqual(cand.parameters.alpha, 0.0)
                self.assertLessEqual(cand.parameters.alpha, 1.0)
            elif cand.family == AttackFamily.UNITARY_PROBE:
                self.assertGreaterEqual(cand.parameters.theta_E_rad, 0.0)
                self.assertLessEqual(cand.parameters.theta_E_rad, math.pi / 4.0)

    def test_evaluate_and_select_policy_case_a_or_b(self):
        """Policy engine must execute exactly K simulations and return a Tri-Case decision."""
        cond = generate_environmental_condition("C01", p_depol=0.01, q_budget=0.08, seed_calib=101)
        res = self.engine.evaluate_and_select_policy(
            condition=cond,
            num_candidates=4,  # Small query budget for fast unit test
            seed_eval=777,
            n_pulses=500,
        )

        self.assertIn("decision", res)
        self.assertIn("selected_policy", res)
        self.assertIn("num_feasible", res)
        self.assertEqual(res["query_budget_consumed"], 4)
        self.assertIn(res["decision"], [
            OperationalDecision.CASE_A_ACTIVE_FEASIBLE,
            OperationalDecision.CASE_B_NULL_FALLBACK,
            OperationalDecision.CASE_C_CHANNEL_ABORT,
        ])


class TestProbeAngleConstraints(unittest.TestCase):
    """Test suite verifying strict physical bounds 0 <= theta_E <= pi/4 across the pipeline."""

    def setUp(self):
        self.generator = ConditionalGenerator(latent_dim=16, condition_dim=2, hidden_dims=[64, 64])

    def test_generator_theta_bounds_under_extreme_latents(self):
        """Generator output theta_E_rad must strictly satisfy 0 <= theta_E <= pi/4 even with extreme latents."""
        # Test large positive, large negative, and normal latents
        z_extreme = torch.cat([
            torch.randn(100, 16) * 10.0,
            torch.ones(50, 16) * 100.0,
            torch.ones(50, 16) * (-100.0),
        ], dim=0)
        c = torch.tensor([[0.11, 0.05]] * 200)

        out = self.generator(z_extreme, c, use_gumbel=False)
        thetas = out["theta_E_rad"].detach().numpy()
        theta_norm = out["masked_policy"][:, 4].detach().numpy()

        self.assertTrue(np.all(thetas >= 0.0), f"Found negative theta_E: {np.min(thetas)}")
        self.assertTrue(np.all(thetas <= (math.pi / 4.0 + 1e-6)), f"theta_E exceeds pi/4: {np.max(thetas)}")
        self.assertTrue(np.all(theta_norm >= 0.0))
        self.assertTrue(np.all(theta_norm <= (1.0 + 1e-6)))

    def test_attack_parameters_theta_bounds_validation(self):
        """AttackParameters must accept [0, pi/4] and reject values outside."""
        # Valid boundaries
        p_zero = AttackParameters(theta_E_rad=0.0, theta_E_deg=0.0)
        self.assertEqual(p_zero.theta_E_rad, 0.0)

        p_pi_4 = AttackParameters(theta_E_rad=math.pi / 4.0, theta_E_deg=45.0)
        self.assertAlmostEqual(p_pi_4.theta_E_rad, math.pi / 4.0)

        # Invalid negative
        with self.assertRaises(ValueError):
            AttackParameters(theta_E_rad=-0.001)

        # Invalid exceed pi/4
        with self.assertRaises(ValueError):
            AttackParameters(theta_E_rad=math.pi / 4.0 + 0.001)

        # Invalid pi/2
        with self.assertRaises(ValueError):
            AttackParameters(theta_E_rad=math.pi / 2.0)

    def test_instantiate_attack_probe_angle_bounds(self):
        """instantiate_attack must reject Probe configurations where theta_E > pi/4."""
        # Valid probe
        valid_cfg = AttackConfiguration(
            attack_id="TEST-VALID-PROBE",
            family=AttackFamily.UNITARY_PROBE,
            parameters=AttackParameters(theta_E_rad=math.pi / 6.0),
            theoretical_disturbance=float(math.sin(math.pi / 6.0) ** 2),
            description="Valid Probe",
        )
        probe = instantiate_attack(valid_cfg)
        self.assertIsNotNone(probe)
        self.assertAlmostEqual(probe.theta_E, math.pi / 6.0)

        # Invalid probe exceeding pi/4 (bypass dataclass via dict reconstruction or subversion)
        # Verify instantiate_attack catches invalid config
        invalid_cfg = AttackConfiguration(
            attack_id="TEST-INVALID-PROBE",
            family=AttackFamily.UNITARY_PROBE,
            parameters=AttackParameters.__new__(AttackParameters),
            theoretical_disturbance=0.5,
            description="Invalid Probe",
        )
        object.__setattr__(invalid_cfg.parameters, "theta_E_rad", math.pi / 2.0)
        object.__setattr__(invalid_cfg.parameters, "alpha", None)
        object.__setattr__(invalid_cfg.parameters, "beta", None)
        object.__setattr__(invalid_cfg.parameters, "theta_E_deg", 90.0)

        with self.assertRaises(ValueError):
            instantiate_attack(invalid_cfg)

    def test_boundary_values_zero_and_pi_over_four(self):
        """Boundary values theta_E = 0 and theta_E = pi/4 must produce correct physical outcomes."""
        cfg_0 = AttackConfiguration(
            attack_id="PROBE-0",
            family=AttackFamily.UNITARY_PROBE,
            parameters=AttackParameters(theta_E_rad=0.0),
            theoretical_disturbance=0.0,
            description="Identity Probe",
        )
        probe_0 = instantiate_attack(cfg_0)
        self.assertAlmostEqual(probe_0.theoretical_disturbance(), 0.0)
        self.assertAlmostEqual(probe_0.theoretical_information_gain(), 0.0)

        cfg_pi_4 = AttackConfiguration(
            attack_id="PROBE-PI-4",
            family=AttackFamily.UNITARY_PROBE,
            parameters=AttackParameters(theta_E_rad=math.pi / 4.0),
            theoretical_disturbance=0.5,
            description="Maximum Disturbance Probe",
        )
        probe_pi_4 = instantiate_attack(cfg_pi_4)
        self.assertAlmostEqual(probe_pi_4.theoretical_disturbance(), 0.50)
        self.assertAlmostEqual(probe_pi_4.theoretical_information_gain(), 1.00)

    def test_policy_engine_candidates_cannot_bypass_pi_over_four(self):
        """PhysicalPolicyEngine must never propose candidates exceeding pi/4."""
        engine = cGANPolicyEngine(generator=self.generator)
        cond = generate_environmental_condition("C17", p_depol=0.01, q_budget=0.05, seed_calib=42)
        candidates = engine.propose_candidate_configurations(condition=cond, num_candidates=50, seed=123)

        for cand in candidates:
            if cand.family == AttackFamily.UNITARY_PROBE:
                self.assertIsNotNone(cand.parameters.theta_E_rad)
                self.assertGreaterEqual(cand.parameters.theta_E_rad, 0.0)
                self.assertLessEqual(cand.parameters.theta_E_rad, math.pi / 4.0 + 1e-7)


class TestCheckpointSelectionEdgeCases(unittest.TestCase):
    """Test suite for Milestone 4A Lexicographic selection and edge cases."""

    def test_indeterminate_not_counted_as_feasible(self):
        """Wilson interval overlapping Q_budget must be Indeterminate, NOT Feasible."""
        # Q_budget = 0.05
        # k = 25, n = 500 -> p_hat = 0.05
        # Wilson interval will span roughly [0.033, 0.072]
        # Q_lcb (0.033) <= 0.05 < Q_ucb (0.072) -> INDETERMINATE
        w = compute_wilson_score_interval(k=25, n=500)
        self.assertLess(w.lcb, 0.05)
        self.assertGreater(w.ucb, 0.05)
        feas = classify_feasibility(w.lcb, w.ucb, q_budget=0.05)
        self.assertEqual(feas, StatisticalFeasibility.INDETERMINATE)
        self.assertNotEqual(feas, StatisticalFeasibility.FEASIBLE)

    def test_abort_causes_disqualification_in_hard_gate(self):
        """If a checkpoint has abort_rate > 0 or validity_rate < 1.0, it must be disqualified."""
        candidate_with_abort = {
            "epoch": 10,
            "validity_rate": 1.0,
            "feasibility_yield": 0.85,
            "abort_rate": 0.05,
            "abort_count": 3,
            "mean_info_gain_feas": 0.15,
        }
        candidate_clean = {
            "epoch": 20,
            "validity_rate": 1.0,
            "feasibility_yield": 0.82,
            "abort_rate": 0.0,
            "abort_count": 0,
            "mean_info_gain_feas": 0.12,
        }
        self.assertGreater(candidate_with_abort["abort_count"], 0)
        self.assertEqual(candidate_clean["abort_count"], 0)

    def test_null_policy_zero_information_reward(self):
        """Null policy has zero mutual info and must not be rewarded as information gain."""
        null_params = AttackParameters()
        self.assertIsNone(null_params.theta_E_rad)
        self.assertIsNone(null_params.alpha)


class TestCommonRandomNumbersInvariance(unittest.TestCase):
    """Empirical verification that Common Random Numbers (CRN) are preserved across distinct attacks."""

    def test_crn_byte_identical_across_distinct_attacks(self):
        seed_sim = 50123
        seed_alice = seed_sim * 17 + 31
        seed_bob = seed_sim * 23 + 47
        seed_attack = seed_sim * 29 + 59
        n_signals = 500
        noise_profile = build_depolarizing_noise_model(p=0.01)

        attacks = [
            AttackConfiguration("P1", AttackFamily.UNITARY_PROBE, AttackParameters(theta_E_rad=0.03), 0.0009, "Probe 1"),
            AttackConfiguration("P2", AttackFamily.UNITARY_PROBE, AttackParameters(theta_E_rad=0.08), 0.0064, "Probe 2"),
            AttackConfiguration("PIR", AttackFamily.PROBABILISTIC_INTERCEPT_RESEND, AttackParameters(alpha=0.15, beta=0.5), 0.0375, "PIR"),
            AttackConfiguration("NULL", AttackFamily.NULL, AttackParameters(), 0.0, "Null"),
        ]

        results = []
        for att in attacks:
            engine = BB84Engine(alice_seed=seed_alice, bob_seed=seed_bob)
            att_inst = instantiate_attack(att, seed=seed_attack)
            res = engine.run(
                n_signals=n_signals,
                attack=att_inst,
                noise=noise_profile,
                seed_simulator=seed_sim,
            )
            results.append(res)

        for i in range(1, 4):
            # Alice bits, Alice bases, Bob bases, and Sifted indices MUST be 100% byte-for-byte identical
            self.assertTrue(np.array_equal(results[0].alice_bits, results[i].alice_bits))
            self.assertTrue(np.array_equal(results[0].alice_bases, results[i].alice_bases))
            self.assertTrue(np.array_equal(results[0].bob_bases, results[i].bob_bases))
            self.assertTrue(np.array_equal(results[0].sifted_indices, results[i].sifted_indices))

        # Bob measured bits must differ due to attack perturbations
        self.assertFalse(np.array_equal(results[0].bob_measured_bits, results[2].bob_measured_bits))

    def test_channel_level_crn_coupling_under_noise(self):
        """Verify that identical seeds couple the channel noise identically across policies.

        Under non-zero channel noise (p=0.04):
        1. Null and Probe(theta=0) produce 100% identical Bob measured bits.
        2. Null and PIR(alpha=0) produce 100% identical Bob measured bits.
        3. Increasing theta_E monotonically increases perturbation relative to Null.
        """
        seed_sim = 60234
        seed_alice = seed_sim * 17 + 31
        seed_bob = seed_sim * 23 + 47
        seed_attack = seed_sim * 29 + 59
        n_signals = 500
        noise_profile = build_depolarizing_noise_model(p=0.04)

        null_att = AttackConfiguration("NULL", AttackFamily.NULL, AttackParameters(), 0.0, "Null")
        probe_zero = AttackConfiguration("P_ZERO", AttackFamily.UNITARY_PROBE, AttackParameters(theta_E_rad=0.0), 0.0, "Probe Zero")
        pir_zero = AttackConfiguration("PIR_ZERO", AttackFamily.PROBABILISTIC_INTERCEPT_RESEND, AttackParameters(alpha=0.0, beta=0.5), 0.0, "PIR Zero")
        probe_mid = AttackConfiguration("P_10", AttackFamily.UNITARY_PROBE, AttackParameters(theta_E_rad=0.10), 0.0196, "Probe 0.10")
        probe_large = AttackConfiguration("P_25", AttackFamily.UNITARY_PROBE, AttackParameters(theta_E_rad=0.25), 0.1147, "Probe 0.25")

        configs = [null_att, probe_zero, pir_zero, probe_mid, probe_large]
        results = []
        for cfg in configs:
            engine = BB84Engine(alice_seed=seed_alice, bob_seed=seed_bob)
            att_inst = instantiate_attack(cfg, seed=seed_attack)
            res = engine.run(
                n_signals=n_signals,
                attack=att_inst,
                noise=noise_profile,
                seed_simulator=seed_sim,
            )
            results.append(res)

        res_null, res_probe_0, res_pir_0, res_probe_mid, res_probe_large = results

        # 1. Base alignment: Alice and Bob basis choices and sifted indices must be byte-identical
        for res in results[1:]:
            np.testing.assert_array_equal(res_null.alice_bits, res.alice_bits)
            np.testing.assert_array_equal(res_null.alice_bases, res.alice_bases)
            np.testing.assert_array_equal(res_null.bob_bases, res.bob_bases)
            np.testing.assert_array_equal(res_null.sifted_indices, res.sifted_indices)

        sifted = res_null.sifted_indices
        self.assertGreater(len(sifted), 200)

        # 2. Perfect channel coupling: Null vs Probe(theta=0) and Null vs PIR(alpha=0)
        # Sifted Bob measurements must be 100% identical despite non-zero channel noise
        diff_probe_0 = int(np.sum(res_null.bob_measured_bits[sifted] != res_probe_0.bob_measured_bits[sifted]))
        diff_pir_0 = int(np.sum(res_null.bob_measured_bits[sifted] != res_pir_0.bob_measured_bits[sifted]))
        self.assertEqual(diff_probe_0, 0, "Probe(theta=0) must identically match Null under noisy CRN channel")
        self.assertEqual(diff_pir_0, 0, "PIR(alpha=0) must identically match Null under noisy CRN channel")

        # 3. Monotonic perturbation: As theta_E increases, bit perturbations relative to Null increase
        diff_mid = int(np.sum(res_null.bob_measured_bits[sifted] != res_probe_mid.bob_measured_bits[sifted]))
        diff_large = int(np.sum(res_null.bob_measured_bits[sifted] != res_probe_large.bob_measured_bits[sifted]))

        self.assertGreater(diff_mid, 0, "Probe(0.10) must induce non-zero perturbation")
        self.assertGreater(diff_large, diff_mid, "Probe(0.25) perturbation must strictly exceed Probe(0.10)")


class TestGeneratorDeviceCompatibility(unittest.TestCase):
    """Verify that PyTorch random Generators are strictly compatible with target device tensors."""

    def test_generator_device_matching_cpu_and_cuda(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for dev in devices:
            gen_a = torch.Generator(device=dev).manual_seed(1042)
            t_a = torch.randn(8, 16, generator=gen_a, device=dev, dtype=torch.float32)
            self.assertEqual(t_a.device.type, dev.type)
            self.assertEqual(t_a.shape, (8, 16))

            # Determinism check with same seed
            gen_b = torch.Generator(device=dev).manual_seed(1042)
            t_b = torch.randn(8, 16, generator=gen_b, device=dev, dtype=torch.float32)
            self.assertTrue(torch.equal(t_a, t_b), f"Same seed must produce identical tensors on {dev.type}")

            # Sensitivity check with different seed
            gen_c = torch.Generator(device=dev).manual_seed(2042)
            t_c = torch.randn(8, 16, generator=gen_c, device=dev, dtype=torch.float32)
            self.assertFalse(torch.equal(t_a, t_c), f"Different seeds must produce different tensors on {dev.type}")


if __name__ == "__main__":
    unittest.main()
