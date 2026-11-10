"""Classical Conditional GAN (cGAN) baseline for adaptive BB84 eavesdropping under NISQ noise.

This package provides:
- `BB84PolicyDataset`: PyTorch dataset for conditioning vectors and masked attack policies.
- `ConditionalGenerator`: Multi-head conditional policy generator with Gumbel-Softmax relaxation.
- `ConditionalDiscriminator`: Spectral-normalized conditional policy discriminator.
- `CGANTrainer`: End-to-end training, validation, and checkpointing engine.
- `cGANPolicyEngine`: Candidate generation, forward BB84 evaluation, and Tri-Case operational fallback logic.
- `OperationalDecision`: Tri-case decision enumeration (Case A, B, C).
"""

from src.cgan.dataset import BB84PolicyDataset, create_dataloaders
from src.cgan.models import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    canonical_mask_policy,
)
from src.cgan.policy_engine import OperationalDecision, cGANPolicyEngine
from src.cgan.trainer import CGANTrainer

__all__ = [
    "BB84PolicyDataset",
    "create_dataloaders",
    "ConditionalGenerator",
    "ConditionalDiscriminator",
    "canonical_mask_policy",
    "CGANTrainer",
    "cGANPolicyEngine",
    "OperationalDecision",
]
