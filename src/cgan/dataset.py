"""PyTorch Dataset and DataLoader infrastructure for BB84 attack policy learning.

Provides:
1. `record_to_policy_tensors`: Deterministic mapping from `DatasetRecord` to condition c in R^2 and masked policy a in R^5.
2. `BB84PolicyDataset`: PyTorch Dataset supporting condition-level partitioning ('train', 'val', 'test'),
   statistical feasibility filtering (Q_UCB <= Q_budget), and mutual information importance weighting.
3. `create_dataloaders`: Factory function constructing stratified DataLoaders.
"""

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.cgan.models import canonical_mask_policy
from src.dataset.schema import (
    AttackFamily,
    DatasetContainer,
    DatasetRecord,
    StatisticalFeasibility,
)


def record_to_policy_tensors(record: DatasetRecord) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Map a DatasetRecord to PyTorch condition and masked policy tensors.

    Args:
        record: DatasetRecord containing environmental condition and attack parameters.

    Returns:
        condition_tensor: Tensor of shape (2,) with [Q_budget, gamma_noise_hat].
        policy_tensor: Tensor of shape (5,) with canonical masked representation:
                       [w_PIR, w_Probe, w_PIR * alpha, w_PIR * beta, w_Probe * theta_norm].
        info_gain: Scalar float mutual information I(A; E).
    """
    # 1. Observable condition (strictly zero leakage of true p_depol or Q_env)
    c_tuple = record.condition.get_observable_condition()
    cond_tensor = torch.tensor([c_tuple[0], c_tuple[1]], dtype=torch.float32)

    # 2. Attack parameters and family mapping
    fam = record.attack_family
    params = record.attack_parameters

    if fam == AttackFamily.PROBABILISTIC_INTERCEPT_RESEND:
        w_pir = 1.0
        w_probe = 0.0
        alpha = float(params.alpha if params.alpha is not None else 0.0)
        beta = float(params.beta if params.beta is not None else 0.5)
        theta_norm = 0.0
    elif fam == AttackFamily.UNITARY_PROBE:
        w_pir = 0.0
        w_probe = 1.0
        alpha = 0.0
        beta = 0.0
        # Normalize theta_E in [0, pi/4] to [0, 1]
        theta_rad = float(params.theta_E_rad if params.theta_E_rad is not None else 0.0)
        theta_norm = float(np.clip((4.0 / math.pi) * theta_rad, 0.0, 1.0))
    elif fam == AttackFamily.NULL:
        w_pir = 0.0
        w_probe = 0.0
        alpha = 0.0
        beta = 0.0
        theta_norm = 0.0
    else:
        raise ValueError(f"Unknown attack family: {fam}")

    fam_weights = torch.tensor([[w_pir, w_probe]], dtype=torch.float32)
    alpha_t = torch.tensor([[alpha]], dtype=torch.float32)
    beta_t = torch.tensor([[beta]], dtype=torch.float32)
    theta_norm_t = torch.tensor([[theta_norm]], dtype=torch.float32)

    policy_tensor = canonical_mask_policy(
        family_weights=fam_weights,
        alpha=alpha_t,
        beta=beta_t,
        theta_norm=theta_norm_t,
    ).squeeze(0)

    info_gain = float(record.raw_observation.eve_info_gain)
    return cond_tensor, policy_tensor, info_gain


class BB84PolicyDataset(Dataset):
    """PyTorch Dataset of BB84 attack policies conditioned on environmental noise."""

    def __init__(
        self,
        records: Sequence[DatasetRecord],
        partition: Optional[str] = None,
        filter_feasible: bool = True,
        tau_I: float = 0.10,
    ):
        """Initialize dataset with records and filtering criteria.

        Args:
            records: Sequence of DatasetRecord instances.
            partition: Optional filter ('train', 'val', 'test', or None).
            filter_feasible: If True, retains only records where Q_UCB <= Q_budget.
            tau_I: Temperature parameter for information gain weighting exp(I_AE / tau_I).
        """
        self.partition = partition
        self.filter_feasible = filter_feasible
        self.tau_I = max(1e-4, tau_I)

        # Apply partition and feasibility filters
        filtered: List[DatasetRecord] = []
        for r in records:
            if partition is not None:
                # If record condition has partition assigned, use it
                r_part = r.condition.partition
                if r_part is not None and r_part != partition:
                    continue

            if filter_feasible:
                if r.derived_evaluation.feasibility_category != StatisticalFeasibility.FEASIBLE:
                    continue

            filtered.append(r)

        self.filtered_records = filtered

        # Pre-extract tensors
        cond_list: List[torch.Tensor] = []
        policy_list: List[torch.Tensor] = []
        info_list: List[float] = []

        for rec in self.filtered_records:
            c_t, p_t, i_gain = record_to_policy_tensors(rec)
            cond_list.append(c_t)
            policy_list.append(p_t)
            info_list.append(i_gain)

        if len(cond_list) > 0:
            self.conditions = torch.stack(cond_list, dim=0)
            self.policies = torch.stack(policy_list, dim=0)
            self.info_gains = torch.tensor(info_list, dtype=torch.float32).unsqueeze(-1)

            # Compute importance sampling weights exp(I_AE / tau_I)
            raw_weights = np.exp(np.array(info_list) / self.tau_I)
            self.sample_weights = torch.tensor(raw_weights, dtype=torch.float64)
        else:
            self.conditions = torch.empty((0, 2), dtype=torch.float32)
            self.policies = torch.empty((0, 5), dtype=torch.float32)
            self.info_gains = torch.empty((0, 1), dtype=torch.float32)
            self.sample_weights = torch.empty((0,), dtype=torch.float64)

    def __len__(self) -> int:
        return len(self.filtered_records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "condition": self.conditions[idx],
            "policy": self.policies[idx],
            "info_gain": self.info_gains[idx],
            "weight": self.sample_weights[idx],
        }

    @classmethod
    def from_json(
        cls,
        json_path: str,
        partition: Optional[str] = None,
        filter_feasible: bool = True,
        tau_I: float = 0.10,
    ) -> "BB84PolicyDataset":
        """Load dataset directly from JSON file."""
        container = DatasetContainer.load_json(json_path)
        return cls(
            records=container.records,
            partition=partition,
            filter_feasible=filter_feasible,
            tau_I=tau_I,
        )


def create_dataloaders(
    dataset_source: Union[str, Sequence[DatasetRecord]],
    batch_size: int = 64,
    tau_I: float = 0.10,
    use_weighted_sampling: bool = True,
) -> Dict[str, DataLoader]:
    """Create Train, Validation, and Test DataLoaders with condition-level separation.

    Args:
        dataset_source: Path to JSON dataset file or sequence of DatasetRecords.
        batch_size: Mini-batch size.
        tau_I: Information gain weighting temperature.
        use_weighted_sampling: If True, uses WeightedRandomSampler for training.

    Returns:
        Dictionary mapping 'train', 'val', 'test' to their respective DataLoaders.
    """
    if isinstance(dataset_source, str):
        container = DatasetContainer.load_json(dataset_source)
        all_records = container.records
    else:
        all_records = dataset_source

    train_ds = BB84PolicyDataset(
        records=all_records,
        partition="train",
        filter_feasible=True,
        tau_I=tau_I,
    )

    val_ds = BB84PolicyDataset(
        records=all_records,
        partition="val",
        filter_feasible=True,
        tau_I=tau_I,
    )

    test_ds = BB84PolicyDataset(
        records=all_records,
        partition="test",
        filter_feasible=True,
        tau_I=tau_I,
    )

    # Train loader with optional importance sampling
    if len(train_ds) > 0 and use_weighted_sampling:
        sampler = WeightedRandomSampler(
            weights=train_ds.sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    elif len(train_ds) > 0:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
    }
