"""PyTorch models for classical Conditional GAN (cGAN) BB84 attack policy generation.

This module implements:
1. `canonical_mask_policy`: Differentiable masking operator ensuring zero gradient leakage into inactive parameter heads.
2. `ConditionalGenerator`: Multi-head conditional generator proposing family probability vectors w in Delta^1,
   bounded PIR parameters (alpha, beta in [0, 1]), and bounded Unitary Probe angle (theta_E in [0, pi/4]).
3. `ConditionalDiscriminator`: Spectral-normalized conditional discriminator distinguishing real feasible training
   exemplars from generated candidate policies conditioned on observable environment c = (Q_budget, gamma_hat).

Mathematical formulation strictly aligns with Milestone 3 Stage A & C specifications.
"""

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


def canonical_mask_policy(
    family_weights: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    theta_norm: torch.Tensor,
) -> torch.Tensor:
    """Construct the canonical 5-dimensional masked attack policy vector.

    Formula:
        a_masked = [w_PIR, w_Probe, w_PIR * alpha, w_PIR * beta, w_Probe * theta_norm]^T in R^5

    Properties:
        - When PIR active (w_PIR=1, w_Probe=0): [1, 0, alpha, beta, 0]
        - When Probe active (w_PIR=0, w_Probe=1): [0, 1, 0, 0, theta_norm]
        - When Null active (w_PIR=0, w_Probe=0): [0, 0, 0, 0, 0]

    Args:
        family_weights: Tensor of shape (B, 2) representing [w_PIR, w_Probe].
        alpha: Tensor of shape (B, 1) or (B,) in [0.0, 1.0].
        beta: Tensor of shape (B, 1) or (B,) in [0.0, 1.0].
        theta_norm: Tensor of shape (B, 1) or (B,) in [0.0, 1.0] representing (4 / pi) * theta_E.

    Returns:
        torch.Tensor of shape (B, 5).
    """
    if family_weights.dim() == 1:
        family_weights = family_weights.unsqueeze(0)
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(-1)
    if beta.dim() == 1:
        beta = beta.unsqueeze(-1)
    if theta_norm.dim() == 1:
        theta_norm = theta_norm.unsqueeze(-1)

    w_pir = family_weights[:, 0:1]
    w_probe = family_weights[:, 1:2]

    masked_alpha = w_pir * alpha
    masked_beta = w_pir * beta
    masked_theta = w_probe * theta_norm

    return torch.cat([w_pir, w_probe, masked_alpha, masked_beta, masked_theta], dim=-1)


class ConditionalGenerator(nn.Module):
    """Multi-head conditional generator for BB84 eavesdropping policy discovery.

    Condition vector:
        c = [Q_budget, gamma_noise_hat] in [0, 1]^2
    Latent noise vector:
        z ~ N(0, I_{d_z})

    Outputs:
        - family_weights: [w_PIR, w_Probe] in Delta^1 (via Softmax or Gumbel-Softmax)
        - alpha: Interception probability in [0, 1] (via Sigmoid)
        - beta: Basis bias in [0, 1] (via Sigmoid)
        - theta_E_rad: Entangling coupling angle in [0, pi/4] (via (pi/4) * Sigmoid)
        - masked_policy: Differentiably masked 5-dimensional policy vector for discriminator
    """

    def __init__(
        self,
        latent_dim: int = 16,
        condition_dim: int = 2,
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_dims = list(hidden_dims)

        in_dim = latent_dim + condition_dim

        # Shared feature trunk
        layers: List[nn.Module] = []
        curr_dim = in_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        self.trunk = nn.Sequential(*layers)

        # Dedicated multi-head linear outputs
        self.family_head = nn.Linear(curr_dim, 2)
        self.pir_head = nn.Linear(curr_dim, 2)
        self.probe_head = nn.Linear(curr_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        use_gumbel: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward evaluation of the conditional generator.

        Args:
            z: Latent noise tensor of shape (B, latent_dim).
            c: Observable condition tensor of shape (B, condition_dim).
            tau: Gumbel-Softmax temperature parameter (for training).
            hard: Whether to use straight-through hard Gumbel-Softmax.
            use_gumbel: If True and in training, apply Gumbel-Softmax; otherwise standard Softmax.

        Returns:
            Dictionary containing:
                - 'masked_policy': Tensor of shape (B, 5)
                - 'family_weights': Tensor of shape (B, 2)
                - 'alpha': Tensor of shape (B, 1)
                - 'beta': Tensor of shape (B, 1)
                - 'theta_norm': Tensor of shape (B, 1)
                - 'theta_E_rad': Tensor of shape (B, 1)
                - 'family_logits': Tensor of shape (B, 2)
        """
        # Concatenate latent noise and condition
        inp = torch.cat([z, c], dim=-1)
        h = self.trunk(inp)

        # 1. Family Head
        fam_logits = self.family_head(h)
        if use_gumbel and self.training:
            fam_weights = F.gumbel_softmax(fam_logits, tau=tau, hard=hard, dim=-1)
        else:
            fam_weights = F.softmax(fam_logits, dim=-1)

        # 2. PIR Parameter Head
        pir_params = torch.sigmoid(self.pir_head(h))
        alpha = pir_params[:, 0:1]
        beta = pir_params[:, 1:2]

        # 3. Unitary Probe Parameter Head
        theta_norm = torch.sigmoid(self.probe_head(h))
        theta_rad = (math.pi / 4.0) * theta_norm

        # 4. Canonical Masking Layer
        a_masked = canonical_mask_policy(
            family_weights=fam_weights,
            alpha=alpha,
            beta=beta,
            theta_norm=theta_norm,
        )

        return {
            "masked_policy": a_masked,
            "family_weights": fam_weights,
            "alpha": alpha,
            "beta": beta,
            "theta_norm": theta_norm,
            "theta_E_rad": theta_rad,
            "family_logits": fam_logits,
        }

    def sample_policy(
        self,
        c: torch.Tensor,
        num_samples: int = 1,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample discrete candidate policies for a given conditioning tensor c."""
        self.eval()
        with torch.no_grad():
            if c.dim() == 1:
                c = c.unsqueeze(0)
            if device is not None:
                c = c.to(device)

            if num_samples > 1 and c.size(0) == 1:
                c = c.repeat(num_samples, 1)

            b_size = c.size(0)
            z = torch.randn(b_size, self.latent_dim, device=c.device, dtype=c.dtype)
            return self.forward(z, c, use_gumbel=False)


class ConditionalDiscriminator(nn.Module):
    """Spectral-normalized conditional discriminator for BB84 eavesdropping policies.

    Inputs:
        - a_masked: Masked policy vector in R^5
        - c: Observable condition vector in R^2 (Q_budget, gamma_hat)
        Concatenated input dimension: 5 + 2 = 7

    Output:
        Scalar real/fake logit in R.
    """

    def __init__(
        self,
        policy_dim: int = 5,
        condition_dim: int = 2,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.policy_dim = policy_dim
        self.condition_dim = condition_dim
        self.hidden_dims = list(hidden_dims)
        self.use_spectral_norm = use_spectral_norm

        in_dim = policy_dim + condition_dim

        layers: List[nn.Module] = []
        curr_dim = in_dim
        for h_dim in self.hidden_dims:
            lin = nn.Linear(curr_dim, h_dim)
            layers.append(spectral_norm(lin) if use_spectral_norm else lin)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        # Final scalar logit layer
        final_lin = nn.Linear(curr_dim, 1)
        layers.append(spectral_norm(final_lin) if use_spectral_norm else final_lin)

        self.net = nn.Sequential(*layers)

    def forward(self, a_masked: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Evaluate real/fake logit for policy conditioned on environment c.

        Args:
            a_masked: Masked policy tensor of shape (B, policy_dim).
            c: Observable condition tensor of shape (B, condition_dim).

        Returns:
            Logit tensor of shape (B, 1).
        """
        if a_masked.dim() == 1:
            a_masked = a_masked.unsqueeze(0)
        if c.dim() == 1:
            c = c.unsqueeze(0)

        inp = torch.cat([a_masked, c], dim=-1)
        return self.net(inp)

