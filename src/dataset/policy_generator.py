"""Deterministic attack policy generator and configuration factory for BB84.

This module implements the approved 16-point attack parameter lattice:
- 1 Null attack configuration (`ATT-00`)
- 7 Unitary probe configurations (`ATT-01` through `ATT-07`)
- 6 Symmetric PIR configurations (`ATT-08` through `ATT-13`)
- 2 Biased PIR configurations (`ATT-14`, `ATT-15`)

Provides strict parameter boundary validation, physical unit conversion (rad/deg),
and instantiation of operational attack classes (`InterceptResendAttack`, `FuchsPeresBrandtProbe`).
"""

import math
from typing import Any, Dict, List, Optional
from src.dataset.schema import (
    AttackConfiguration,
    AttackFamily,
    AttackParameters,
)
from src.bb84.attacks import (
    BaseAttack,
    FuchsPeresBrandtProbe,
    InterceptResendAttack,
)


def _compute_probe_disturbance(theta_rad: float) -> float:
    """Compute analytical disturbance D(theta) = sin^2(theta) for Unitary Probe."""
    return float(math.sin(theta_rad) ** 2)


def _compute_pir_disturbance(alpha: float, beta: float) -> float:
    """Compute analytical overall disturbance for PIR with uniform Alice basis."""
    # For unbiased Alice (P(Z) = 0.5), overall disturbance is alpha / 4 regardless of beta
    return float(alpha / 4.0)


# Exact 16-point attack specification table
_ATTACK_CONFIGURATIONS_TABLE: Dict[str, Dict[str, Any]] = {
    "ATT-00": {
        "family": AttackFamily.NULL,
        "alpha": None,
        "beta": None,
        "theta_E_rad": 0.0,
        "theoretical_disturbance": 0.0,
        "description": "Unattacked reference channel (No Eve intervention)",
    },
    "ATT-01": {
        "family": AttackFamily.UNITARY_PROBE,
        "alpha": None,
        "beta": None,
        "theta_E_rad": 0.0500,
        "theoretical_disturbance": _compute_probe_disturbance(0.0500),
        "description": "Weak unitary entangling probe (theta_E = 0.0500 rad, ~2.86 deg)",
    },
    "ATT-02": {
        "family": AttackFamily.UNITARY_PROBE,
        "alpha": None,
        "beta": None,
        "theta_E_rad": 0.1000,
        "theoretical_disturbance": _compute_probe_disturbance(0.1000),
        "description": "Low-disturbance unitary probe (theta_E = 0.1000 rad, ~5.73 deg)",
    },
    "ATT-03": {
        "family": AttackFamily.UNITARY_PROBE,
        "alpha": None,
        "beta": None,
        "theta_E_rad": 0.1500,
        "theoretical_disturbance": _compute_probe_disturbance(0.1500),
        "description": "Moderate unitary probe (theta_E = 0.1500 rad, ~8.59 deg)",
    },
    "ATT-04": {
        "family": AttackFamily.UNITARY_PROBE,
        "alpha": None,
        "beta": None,
        "theta_E_rad": 0.2000,
        "theoretical_disturbance": _compute_probe_disturbance(0.2000),
        "description": "Medium-intensity unitary probe (theta_E = 0.2000 rad, ~11.46 deg)",
    },
    "ATT-05": {
        "family": AttackFamily.UNITARY_PROBE,
        "alpha": None,
        "beta": None,
        "theta_E_rad": 0.2500,
        "theoretical_disturbance": _compute_probe_disturbance(0.2500),
        "description": "Near-threshold probe for Q_budget=0.08 (theta_E = 0.2500 rad, ~14.32 deg)",
    },
    "ATT-06": {
        "family": AttackFamily.UNITARY_PROBE,
        "alpha": None,
        "beta": None,
        "theta_E_rad": 0.3000,
        "theoretical_disturbance": _compute_probe_disturbance(0.3000),
        "description": "High-intensity unitary probe (theta_E = 0.3000 rad, ~17.19 deg)",
    },
    "ATT-07": {
        "family": AttackFamily.UNITARY_PROBE,
        "alpha": None,
        "beta": None,
        "theta_E_rad": 0.3398,
        "theoretical_disturbance": _compute_probe_disturbance(0.3398),
        "description": "Critical threshold probe (theta_E = 0.3398 rad, D ~ 11.0%)",
    },
    "ATT-08": {
        "family": AttackFamily.PROBABILISTIC_INTERCEPT_RESEND,
        "alpha": 0.05,
        "beta": 0.50,
        "theta_E_rad": None,
        "theoretical_disturbance": _compute_pir_disturbance(0.05, 0.50),
        "description": "Symmetric PIR with alpha = 0.05, beta = 0.50 (D = 1.25%)",
    },
    "ATT-09": {
        "family": AttackFamily.PROBABILISTIC_INTERCEPT_RESEND,
        "alpha": 0.10,
        "beta": 0.50,
        "theta_E_rad": None,
        "theoretical_disturbance": _compute_pir_disturbance(0.10, 0.50),
        "description": "Symmetric PIR with alpha = 0.10, beta = 0.50 (D = 2.50%)",
    },
    "ATT-10": {
        "family": AttackFamily.PROBABILISTIC_INTERCEPT_RESEND,
        "alpha": 0.20,
        "beta": 0.50,
        "theta_E_rad": None,
        "theoretical_disturbance": _compute_pir_disturbance(0.20, 0.50),
        "description": "Symmetric PIR with alpha = 0.20, beta = 0.50 (D = 5.00%)",
    },
    "ATT-11": {
        "family": AttackFamily.PROBABILISTIC_INTERCEPT_RESEND,
        "alpha": 0.30,
        "beta": 0.50,
        "theta_E_rad": None,
        "theoretical_disturbance": _compute_pir_disturbance(0.30, 0.50),
        "description": "Symmetric PIR with alpha = 0.30, beta = 0.50 (D = 7.50%)",
    },
    "ATT-12": {
        "family": AttackFamily.PROBABILISTIC_INTERCEPT_RESEND,
        "alpha": 0.40,
        "beta": 0.50,
        "theta_E_rad": None,
        "theoretical_disturbance": _compute_pir_disturbance(0.40, 0.50),
        "description": "Symmetric PIR with alpha = 0.40, beta = 0.50 (D = 10.00%)",
    },
    "ATT-13": {
        "family": AttackFamily.PROBABILISTIC_INTERCEPT_RESEND,
        "alpha": 0.44,
        "beta": 0.50,
        "theta_E_rad": None,
        "theoretical_disturbance": _compute_pir_disturbance(0.44, 0.50),
        "description": "Critical threshold PIR with alpha = 0.44, beta = 0.50 (D = 11.00%)",
    },
    "ATT-14": {
        "family": AttackFamily.PROBABILISTIC_INTERCEPT_RESEND,
        "alpha": 0.20,
        "beta": 0.30,
        "theta_E_rad": None,
        "theoretical_disturbance": _compute_pir_disturbance(0.20, 0.30),
        "description": "Biased PIR (X-biased) with alpha = 0.20, beta = 0.30 (D = 5.00%)",
    },
    "ATT-15": {
        "family": AttackFamily.PROBABILISTIC_INTERCEPT_RESEND,
        "alpha": 0.20,
        "beta": 0.70,
        "theta_E_rad": None,
        "theoretical_disturbance": _compute_pir_disturbance(0.20, 0.70),
        "description": "Biased PIR (Z-biased) with alpha = 0.20, beta = 0.70 (D = 5.00%)",
    },
}


def get_attack_configuration(attack_id: str) -> AttackConfiguration:
    """Retrieve validated AttackConfiguration instance by canonical ID.

    Args:
        attack_id: Canonical identifier in {'ATT-00', ..., 'ATT-15'}.

    Returns:
        AttackConfiguration: Fully validated configuration object.

    Raises:
        KeyError: If attack_id is not in the recognized specification table.
    """
    clean_id = attack_id.strip().upper()
    if clean_id not in _ATTACK_CONFIGURATIONS_TABLE:
        raise KeyError(
            f"Unknown attack ID '{attack_id}'. Expected one of: "
            f"{list(_ATTACK_CONFIGURATIONS_TABLE.keys())}"
        )

    spec = _ATTACK_CONFIGURATIONS_TABLE[clean_id]
    theta_rad = spec["theta_E_rad"]
    theta_deg = math.degrees(theta_rad) if theta_rad is not None else None

    params = AttackParameters(
        alpha=spec["alpha"],
        beta=spec["beta"],
        theta_E_rad=theta_rad,
        theta_E_deg=round(theta_deg, 4) if theta_deg is not None else None,
    )

    return AttackConfiguration(
        attack_id=clean_id,
        family=spec["family"],
        parameters=params,
        theoretical_disturbance=float(spec["theoretical_disturbance"]),
        description=spec["description"],
    )


def get_all_attack_configurations() -> List[AttackConfiguration]:
    """Return all 16 canonical attack configurations in order ATT-00 through ATT-15.

    Returns:
        List[AttackConfiguration]: 16 ordered AttackConfiguration instances.
    """
    return [get_attack_configuration(f"ATT-{i:02d}") for i in range(16)]


def instantiate_attack(
    config: AttackConfiguration,
    seed: Optional[int] = None,
) -> Optional[BaseAttack]:
    """Instantiate the operational physical attack class from configuration.

    Args:
        config: AttackConfiguration instance.
        seed: Optional RNG seed for stochastic attacks (PIR basis/interception choices).

    Returns:
        Optional[BaseAttack]:
            - None for NULL attack (carrier passes completely undisturbed).
            - FuchsPeresBrandtProbe for UNITARY_PROBE family.
            - InterceptResendAttack for PROBABILISTIC_INTERCEPT_RESEND family.

    Raises:
        ValueError: If config parameters violate physical bounds.
    """
    family = config.family

    if family == AttackFamily.NULL:
        return None

    elif family == AttackFamily.UNITARY_PROBE:
        theta_rad = config.parameters.theta_E_rad
        if theta_rad is None:
            raise ValueError(f"Probe configuration {config.attack_id} missing theta_E_rad")
        if not (0.0 <= theta_rad <= (math.pi / 4.0 + 1e-9)):
            raise ValueError(
                f"theta_E_rad must be in [0.0, pi/4], got {theta_rad} for {config.attack_id}"
            )
        return FuchsPeresBrandtProbe(theta_E=theta_rad)

    elif family == AttackFamily.PROBABILISTIC_INTERCEPT_RESEND:
        alpha = config.parameters.alpha
        beta = config.parameters.beta
        if alpha is None or beta is None:
            raise ValueError(f"PIR configuration {config.attack_id} missing alpha or beta")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha} for {config.attack_id}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0.0, 1.0], got {beta} for {config.attack_id}")
        return InterceptResendAttack(alpha=alpha, beta=beta, seed=seed)

    else:
        raise ValueError(f"Unrecognized attack family: {family}")
