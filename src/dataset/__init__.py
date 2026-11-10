"""Dataset generation and statistical evaluation pipeline for classical BB84 attack policies.

This package provides:
- Documented, versioned dataset schema (`schema.py`).
- Statistical evaluation with Wilson score intervals and feasibility classification (`statistics.py`).
- Deterministic attack policy configuration generator (`policy_generator.py`).
- End-to-end dataset generation pipeline integrating with BB84Engine (`generator.py`).
"""

from src.dataset.schema import (
    AttackConfiguration,
    AttackFamily,
    DatasetContainer,
    DatasetRecord,
    EnvironmentalCondition,
    StatisticalFeasibility,
)
from src.dataset.statistics import (
    classify_feasibility,
    compute_wilson_score_interval,
)
from src.dataset.policy_generator import (
    get_attack_configuration,
    get_all_attack_configurations,
    instantiate_attack,
)

__all__ = [
    "AttackConfiguration",
    "AttackFamily",
    "DatasetContainer",
    "DatasetRecord",
    "EnvironmentalCondition",
    "StatisticalFeasibility",
    "classify_feasibility",
    "compute_wilson_score_interval",
    "get_attack_configuration",
    "get_all_attack_configurations",
    "instantiate_attack",
]

