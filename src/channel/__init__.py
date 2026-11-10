"""Quantum Channel and NISQ Noise Models Package.

Exports:
- Noise channel types and profile dataclasses.
- Analytical Kraus operator generators for depolarizing, amplitude damping,
  phase damping, and thermal relaxation channels.
- Mathematical validation utilities (completeness, trace preservation, positivity).
- Analytical QBER predictors for conjugate bases.
- Qiskit Aer NoiseModel builders.
- QuantumChannel transmission link supporting attacks and noise.
"""

from src.channel.noise_models import (
    NoiseType,
    NoiseChannelProfile,
    get_depolarizing_kraus,
    get_amplitude_damping_kraus,
    get_phase_damping_kraus,
    get_thermal_relaxation_kraus,
    verify_kraus_completeness,
    apply_kraus_channel,
    verify_density_matrix,
    compute_state_fidelity,
    compute_coherence,
    compute_channel_fidelity,
    compute_basis_qber_from_density_matrices,
    predict_depolarizing_qber,
    predict_amplitude_damping_qber,
    predict_phase_damping_qber,
    predict_thermal_relaxation_qber,
    validate_thermal_parameters,
    build_depolarizing_noise_model,
    build_amplitude_damping_noise_model,
    build_phase_damping_noise_model,
    build_thermal_relaxation_noise_model,
    build_combined_noise_model,
)
from src.channel.quantum_channel import QuantumChannel, TransmissionResult

__all__ = [
    "NoiseType",
    "NoiseChannelProfile",
    "get_depolarizing_kraus",
    "get_amplitude_damping_kraus",
    "get_phase_damping_kraus",
    "get_thermal_relaxation_kraus",
    "verify_kraus_completeness",
    "apply_kraus_channel",
    "verify_density_matrix",
    "compute_state_fidelity",
    "compute_coherence",
    "compute_channel_fidelity",
    "compute_basis_qber_from_density_matrices",
    "predict_depolarizing_qber",
    "predict_amplitude_damping_qber",
    "predict_phase_damping_qber",
    "predict_thermal_relaxation_qber",
    "validate_thermal_parameters",
    "build_depolarizing_noise_model",
    "build_amplitude_damping_noise_model",
    "build_phase_damping_noise_model",
    "build_thermal_relaxation_noise_model",
    "build_combined_noise_model",
    "QuantumChannel",
    "TransmissionResult",
]
