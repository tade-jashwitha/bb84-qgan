"""Runtime candidate policy generator and Tri-Case operational fallback engine.

Implements:
1. `cGANPolicyEngine`: Proposes K=16 candidate attack policies from G_phi for condition c = (Q_budget, gamma_hat).
2. Differentiable to physical parameter mapping (alpha, beta, theta_E).
3. Physical BB84 forward simulation of proposed candidates under the noisy channel.
4. Statistically exact Wilson score feasibility filtering: F(c) = { a_k | Q_UCB_k <= Q_budget }.
5. Tri-Case operational fallback decision logic:
   - Case A: Active feasible attack maximizing I_AE.
   - Case B: Null attack fallback when active attacks are infeasible but link noise is tolerable.
   - Case C: Null attack fallback when link noise itself exceeds budget.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.bb84.bb84_engine import BB84Engine
from src.cgan.models import ConditionalGenerator
from src.dataset.generator import execute_single_simulation
from src.dataset.policy_generator import get_attack_configuration
from src.dataset.schema import (
    AttackConfiguration,
    AttackFamily,
    AttackParameters,
    DatasetRecord,
    EnvironmentalCondition,
    StatisticalFeasibility,
)


class OperationalDecision(str):
    """Tri-case operational selection outcome."""

    CASE_A_ACTIVE_FEASIBLE = "Case A: Active Attack Feasible"
    CASE_B_NULL_FALLBACK = "Case B: Null Attack Fallback"
    CASE_C_CHANNEL_ABORT = "Case C: Channel Noise Abort"


class cGANPolicyEngine:
    """Policy engine coordinating candidate generation, forward evaluation, and Tri-Case selection."""

    def __init__(
        self,
        generator: ConditionalGenerator,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator.to(self.device)
        self.generator.eval()

    def propose_candidate_configurations(
        self,
        condition: EnvironmentalCondition,
        num_candidates: int = 16,
        seed: Optional[int] = None,
    ) -> List[AttackConfiguration]:
        """Propose K candidate physical attack configurations from the generator.

        Args:
            condition: EnvironmentalCondition specification.
            num_candidates: Number of candidates K (default 16).
            seed: Optional RNG seed for reproducibility.

        Returns:
            List of K AttackConfiguration instances with validated parameters.
        """
        if seed is not None:
            torch.manual_seed(seed)

        c_tuple = condition.get_observable_condition()
        c_tensor = torch.tensor(
            [[c_tuple[0], c_tuple[1]]],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            gen_out = self.generator.sample_policy(c_tensor, num_samples=num_candidates)

        fam_weights = gen_out["family_weights"].cpu().numpy()
        alphas = gen_out["alpha"].squeeze(-1).cpu().numpy()
        betas = gen_out["beta"].squeeze(-1).cpu().numpy()
        theta_rads = gen_out["theta_E_rad"].squeeze(-1).cpu().numpy()

        candidates: List[AttackConfiguration] = []
        for i in range(num_candidates):
            w_pir = float(fam_weights[i, 0])
            w_probe = float(fam_weights[i, 1])

            # Select family based on argmax probability
            if w_pir >= w_probe:
                # PIR Attack
                alpha_val = float(np.clip(alphas[i], 0.0, 1.0))
                beta_val = float(np.clip(betas[i], 0.0, 1.0))
                params = AttackParameters(
                    alpha=alpha_val,
                    beta=beta_val,
                    theta_E_rad=None,
                    theta_E_deg=None,
                )
                dist = alpha_val / 4.0
                desc = f"cGAN-generated PIR (alpha={alpha_val:.4f}, beta={beta_val:.4f})"
                config = AttackConfiguration(
                    attack_id=f"cGAN-PIR-{i:02d}",
                    family=AttackFamily.PROBABILISTIC_INTERCEPT_RESEND,
                    parameters=params,
                    theoretical_disturbance=dist,
                    description=desc,
                )
            else:
                # Unitary Probe Attack
                theta_val = float(np.clip(theta_rads[i], 0.0, math.pi / 4.0))
                theta_deg = float(math.degrees(theta_val))
                params = AttackParameters(
                    alpha=None,
                    beta=None,
                    theta_E_rad=theta_val,
                    theta_E_deg=theta_deg,
                )
                dist = float(math.sin(theta_val) ** 2)
                desc = f"cGAN-generated Probe (theta_E={theta_val:.4f} rad, {theta_deg:.2f} deg)"
                config = AttackConfiguration(
                    attack_id=f"cGAN-Probe-{i:02d}",
                    family=AttackFamily.UNITARY_PROBE,
                    parameters=params,
                    theoretical_disturbance=dist,
                    description=desc,
                )

            candidates.append(config)

        return candidates

    def evaluate_and_select_policy(
        self,
        condition: EnvironmentalCondition,
        num_candidates: int = 16,
        seed_eval: int = 42,
        n_pulses: int = 2000,
        rho_test: float = 0.50,
    ) -> Dict[str, Any]:
        """Execute the complete candidate proposal, forward evaluation, and Tri-Case selection.

        Args:
            condition: EnvironmentalCondition specification.
            num_candidates: Query budget K (default 16).
            seed_eval: Base seed for simulation executions.
            n_pulses: Carrier pulses per simulation.
            rho_test: Test bit fraction.

        Returns:
            Dictionary containing:
                - 'decision': OperationalDecision (Case A, B, or C)
                - 'selected_policy': Winning AttackConfiguration
                - 'selected_record': DatasetRecord of winning policy
                - 'candidate_records': All K evaluated DatasetRecords
                - 'num_feasible': Count of statistically feasible candidates
                - 'query_budget_consumed': Total forward simulations executed (K=16)
        """
        # 1. Propose K candidates
        candidates = self.propose_candidate_configurations(
            condition=condition,
            num_candidates=num_candidates,
            seed=seed_eval,
        )

        # 2. Forward BB84 physical simulations (exactly K queries)
        eval_records: List[DatasetRecord] = []
        for i, cand in enumerate(candidates):
            sim_seed = seed_eval * 1000 + i + 1
            record = execute_single_simulation(
                condition=condition,
                attack_config=cand,
                seed_sim=sim_seed,
                n_pulses=n_pulses,
                rho_test=rho_test,
            )
            eval_records.append(record)

        # 3. Filter feasible candidates under Wilson interval
        feasible_records = [
            r for r in eval_records
            if r.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE
        ]

        # 4. Tri-Case Operational Selection Logic
        if len(feasible_records) > 0:
            # Case A: Select feasible active attack maximizing empirical mutual information
            best_record = max(feasible_records, key=lambda r: r.raw_observation.eve_info_gain)
            best_config = AttackConfiguration(
                attack_id=best_record.attack_id,
                family=best_record.attack_family,
                parameters=best_record.attack_parameters,
                theoretical_disturbance=best_record.derived_evaluation.wilson_q_center,
                description="Winning cGAN candidate",
            )
            decision = OperationalDecision.CASE_A_ACTIVE_FEASIBLE
        else:
            # Evaluate Null Attack reference condition
            null_config = get_attack_configuration("ATT-00")
            null_record = execute_single_simulation(
                condition=condition,
                attack_config=null_config,
                seed_sim=seed_eval * 9999 + 1,
                n_pulses=n_pulses,
                rho_test=rho_test,
            )

            if null_record.derived_evaluation.feasibility_category == StatisticalFeasibility.FEASIBLE:
                # Case B: Null attack is feasible, active attacks infeasible
                best_config = null_config
                best_record = null_record
                decision = OperationalDecision.CASE_B_NULL_FALLBACK
            else:
                # Case C: Channel noise alone breaches budget
                best_config = null_config
                best_record = null_record
                decision = OperationalDecision.CASE_C_CHANNEL_ABORT

        return {
            "decision": decision,
            "selected_policy": best_config,
            "selected_record": best_record,
            "candidate_records": eval_records,
            "num_feasible": len(feasible_records),
            "query_budget_consumed": len(eval_records),
        }

