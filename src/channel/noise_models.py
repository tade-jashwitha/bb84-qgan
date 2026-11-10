"""NISQ Noise Models for BB84 Quantum Key Distribution Channel.

This module implements realistic quantum channel noise models for simulating
noisy quantum communication channels in BB84 QKD under NISQ constraints:
1. Depolarizing Noise:
   - Parameter p_depol in [0.0, 1.0].
   - Map: E(rho) = (1 - p) * rho + p * (I / 2).
   - Pauli error representation: (1 - 3p/4) I + (p/4) X + (p/4) Y + (p/4) Z.
   - Symmetric bit-flip and phase-flip disturbance across both Z and X conjugate bases:
     QBER_Z = p / 2, QBER_X = p / 2, QBER = p / 2.

2. Amplitude Damping (Energy Relaxation):
   - Parameter gamma in [0.0, 1.0].
   - Kraus operators:
     K_0 = [[1, 0], [0, sqrt(1 - gamma)]]
     K_1 = [[0, sqrt(gamma)], [0, 0]]
   - Models physical photon loss / decay |1> -> |0>.
   - Asymmetric error characteristics:
     QBER_Z = f_1 * gamma (where f_1 is the fraction of |1> states, 0.5 * gamma for uniform),
     QBER_X = (1 - sqrt(1 - gamma)) / 2.

3. Phase Damping (Pure Dephasing / Coherence Loss):
   - Parameter lambda in [0.0, 1.0].
   - Kraus operators:
     K_0 = [[1, 0], [0, sqrt(1 - lambda)]]
     K_1 = [[0, 0], [0, sqrt(lambda)]]
   - Off-diagonal coherence decay: rho_01 -> sqrt(1 - lambda) * rho_01.
   - Basis selectivity:
     QBER_Z = 0.0 (no bit-flips in computational basis),
     QBER_X = (1 - sqrt(1 - lambda)) / 2.

4. Thermal Relaxation (T1 / T2 Physical Decay):
   - Parameters: T_1 (relaxation time), T_2 (dephasing time), t (channel transit time).
   - Physical constraint: T_2 <= 2 * T_1 strictly enforced.
   - Combines longitudinal energy relaxation (gamma = 1 - exp(-t / T1))
     and transverse dephasing (coherence factor exp(-t / T2)).
   - Predicts:
     QBER_Z = (1 - exp(-t / T1)) / 2,
     QBER_X = (1 - exp(-t / T2)) / 2.

Compatible with Qiskit 2.6.0, Qiskit Aer 0.18.0, NumPy 2.4.6, SciPy 1.17.1.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from qiskit.quantum_info import Kraus
from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
    thermal_relaxation_error,
)


class NoiseType(str, Enum):
    """Supported NISQ channel noise mechanisms."""

    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    THERMAL_RELAXATION = "thermal_relaxation"
    COMBINED = "combined"


@dataclass(frozen=True)
class NoiseChannelProfile:
    """Container encapsulating noise channel parameters, Kraus operators, and predictions.

    Attributes:
        noise_type: Category of the noise channel.
        parameters: Dictionary of physical parameters (e.g., p, gamma, lambda, T1, T2).
        kraus_operators: List of complex 2x2 Kraus matrices satisfying sum(K_i^dagger K_i) = I.
        noise_model: Configured Qiskit Aer NoiseModel instance.
        theoretical_qber: Asymptotic total QBER across both BB84 conjugate bases.
        theoretical_qber_z: Theoretical QBER conditioned on Z-basis sifted signals.
        theoretical_qber_x: Theoretical QBER conditioned on X-basis sifted signals.
    """

    noise_type: NoiseType
    parameters: Dict[str, float]
    kraus_operators: List[np.ndarray]
    noise_model: NoiseModel
    theoretical_qber: float
    theoretical_qber_z: float
    theoretical_qber_x: float


# ==============================================================================
# 1. Mathematical Kraus Operators & Channel Validation
# ==============================================================================

def get_depolarizing_kraus(p: float) -> List[np.ndarray]:
    """Construct Kraus operators for the single-qubit depolarizing channel.

    Map: E(rho) = (1 - p) * rho + p * (I / 2).
    Kraus decomposition:
        K_0 = sqrt(1 - 3p/4) * I
        K_1 = sqrt(p/4) * X
        K_2 = sqrt(p/4) * Y
        K_3 = sqrt(p/4) * Z

    Args:
        p: Depolarizing probability parameter in [0.0, 1.0].

    Returns:
        List[np.ndarray]: Four 2x2 complex Kraus matrices.

    Raises:
        ValueError: If p is not within [0.0, 1.0].
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Depolarizing parameter p must be in [0.0, 1.0], got: {p}")

    p0 = float(np.sqrt(1.0 - 0.75 * p))
    pk = float(np.sqrt(0.25 * p))

    K0 = p0 * np.eye(2, dtype=np.complex128)
    K1 = pk * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)          # X
    K2 = pk * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)       # Y
    K3 = pk * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)          # Z

    return [K0, K1, K2, K3]


def get_amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
    """Construct Kraus operators for the single-qubit amplitude damping channel.

    Models energy relaxation from |1> to |0>:
        K_0 = [[1, 0], [0, sqrt(1 - gamma)]]
        K_1 = [[0, sqrt(gamma)], [0, 0]]

    Args:
        gamma: Relaxation probability parameter in [0.0, 1.0].

    Returns:
        List[np.ndarray]: Two 2x2 complex Kraus matrices.

    Raises:
        ValueError: If gamma is not within [0.0, 1.0].
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"Amplitude damping parameter gamma must be in [0.0, 1.0], got: {gamma}")

    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)

    return [K0, K1]


def get_phase_damping_kraus(lam: float) -> List[np.ndarray]:
    """Construct Kraus operators for the single-qubit phase damping channel.

    Models pure dephasing (coherence loss) without energy relaxation:
        K_0 = [[1, 0], [0, sqrt(1 - lambda)]]
        K_1 = [[0, 0], [0, sqrt(lambda)]]

    Args:
        lam: Dephasing probability parameter in [0.0, 1.0].

    Returns:
        List[np.ndarray]: Two 2x2 complex Kraus matrices.

    Raises:
        ValueError: If lam is not within [0.0, 1.0].
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"Phase damping parameter lambda must be in [0.0, 1.0], got: {lam}")

    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - lam)]], dtype=np.complex128)
    K1 = np.array([[0.0, 0.0], [0.0, np.sqrt(lam)]], dtype=np.complex128)

    return [K0, K1]


def get_thermal_relaxation_kraus(
    t1: float,
    t2: float,
    time: float,
    p_excited: float = 0.0,
) -> List[np.ndarray]:
    """Extract canonical Kraus operators for thermal relaxation from Qiskit Aer.

    Args:
        t1: Longitudinal relaxation time (seconds) > 0.
        t2: Transverse dephasing time (seconds) > 0.
        time: Channel interaction / gate duration (seconds) >= 0.
        p_excited: Equilibrium excited state population in [0.0, 1.0].

    Returns:
        List[np.ndarray]: Canonical Kraus operators.

    Raises:
        ValueError: If physical constraints (t1 > 0, t2 > 0, t2 <= 2*t1, time >= 0) are violated.
    """
    validate_thermal_parameters(t1, t2, time, p_excited)
    err = thermal_relaxation_error(t1=t1, t2=t2, time=time, excited_state_population=p_excited)
    kraus = Kraus(err)
    return [np.array(K, dtype=np.complex128) for K in kraus.data]


def verify_kraus_completeness(
    kraus_ops: Sequence[np.ndarray],
    atol: float = 1e-10,
) -> bool:
    """Verify trace-preserving completeness condition: sum_i K_i^dagger K_i = I.

    Args:
        kraus_ops: Sequence of complex Kraus matrices.
        atol: Absolute numerical tolerance.

    Returns:
        bool: True if sum equals identity within tolerance, False otherwise.
    """
    if not kraus_ops:
        return False
    dim = kraus_ops[0].shape[0]
    total = sum(K.conj().T @ K for K in kraus_ops)
    identity = np.eye(dim, dtype=np.complex128)
    return bool(np.allclose(total, identity, atol=atol))


def apply_kraus_channel(
    rho: np.ndarray,
    kraus_ops: Sequence[np.ndarray],
) -> np.ndarray:
    """Apply a quantum channel to a density matrix: E(rho) = sum_i K_i rho K_i^dagger.

    Args:
        rho: Input 2x2 complex density matrix.
        kraus_ops: Sequence of 2x2 complex Kraus operators.

    Returns:
        np.ndarray: Output 2x2 complex density matrix.
    """
    rho_out = np.zeros_like(rho, dtype=np.complex128)
    for K in kraus_ops:
        rho_out += K @ rho @ K.conj().T
    return rho_out


def verify_density_matrix(
    rho: np.ndarray,
    atol: float = 1e-10,
) -> Tuple[bool, bool, bool]:
    """Validate mathematical physical properties of a density matrix.

    Args:
        rho: 2x2 complex matrix.
        atol: Numerical tolerance.

    Returns:
        Tuple[bool, bool, bool]:
            - is_hermitian: rho = rho^dagger
            - is_unit_trace: Tr(rho) = 1.0
            - is_positive_semidefinite: all eigenvalues >= -atol
    """
    is_hermitian = bool(np.allclose(rho, rho.conj().T, atol=atol))
    is_unit_trace = bool(np.isclose(np.trace(rho), 1.0 + 0.0j, atol=atol))

    eigenvalues = np.linalg.eigvalsh(rho)
    is_positive_semidefinite = bool(np.all(eigenvalues >= -atol))

    return is_hermitian, is_unit_trace, is_positive_semidefinite


def compute_state_fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Compute state fidelity F(rho1, rho2).

    When rho2 is a pure state |psi><psi|:
        F = Tr(rho1 @ rho2) = <psi| rho1 |psi>.

    Args:
        rho1: First 2x2 complex density matrix.
        rho2: Second 2x2 complex density matrix (or pure state projector).

    Returns:
        float: Fidelity in [0.0, 1.0].
    """
    prod = rho1 @ rho2
    trace_val = float(np.real(np.trace(prod)))
    return float(np.clip(trace_val, 0.0, 1.0))


def compute_coherence(rho: np.ndarray) -> float:
    """Compute l1-norm coherence C_l1(rho) = sum_{i != j} |rho_ij|.

    For single qubit: 2 * |rho_01|.

    Args:
        rho: 2x2 complex density matrix.

    Returns:
        float: l1-norm coherence in [0.0, 1.0].
    """
    return float(2.0 * np.abs(rho[0, 1]))


def compute_channel_fidelity(
    kraus_ops: Sequence[np.ndarray],
    pure_states: Optional[Sequence[np.ndarray]] = None,
) -> float:
    """Compute average fidelity across BB84 states under a Kraus quantum channel.

    BB84 states: {|0>, |1>, |+>, |->}.

    Args:
        kraus_ops: Sequence of 2x2 Kraus matrices.
        pure_states: Optional custom list of 2x2 pure state projectors.

    Returns:
        float: Average state fidelity in [0.0, 1.0].
    """
    if pure_states is None:
        pure_states = [
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),              # |0><0|
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),              # |1><1|
            np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128),              # |+><+|
            np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128),            # |-><-|
        ]

    fidelities = []
    for rho_in in pure_states:
        rho_out = apply_kraus_channel(rho_in, kraus_ops)
        f = compute_state_fidelity(rho_out, rho_in)
        fidelities.append(f)

    return float(np.mean(fidelities))


def compute_basis_qber_from_density_matrices(
    kraus_ops: Sequence[np.ndarray],
    f1: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute exact theoretical QBER directly from Kraus action on BB84 states.

    Args:
        kraus_ops: Sequence of 2x2 Kraus matrices.
        f1: Fraction of |1> states transmitted in Z basis (default 0.5).

    Returns:
        Tuple[float, float, float]: (QBER_total, QBER_Z, QBER_X).
    """
    # |0><0| and |1><1|
    rho_0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho_1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    out_0 = apply_kraus_channel(rho_0, kraus_ops)
    out_1 = apply_kraus_channel(rho_1, kraus_ops)

    # Bob measures in Z basis:
    # Error for |0> is measuring |1>: out_0[1, 1]
    # Error for |1> is measuring |0>: out_1[0, 0]
    err_0 = float(np.real(out_0[1, 1]))
    err_1 = float(np.real(out_1[0, 0]))
    qber_z = float((1.0 - f1) * err_0 + f1 * err_1)

    # |+><+| and |-><-|
    rho_p = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    rho_m = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)
    out_p = apply_kraus_channel(rho_p, kraus_ops)
    out_m = apply_kraus_channel(rho_m, kraus_ops)

    # Bob measures in X basis:
    # Projection onto |->: err_p = < - | out_p | - >
    # Projection onto |+>: err_m = < + | out_m | + >
    v_p = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    v_m = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)

    err_p = float(np.real(v_m.conj().T @ out_p @ v_m))
    err_m = float(np.real(v_p.conj().T @ out_m @ v_p))
    qber_x = float(0.5 * (err_p + err_m))

    qber_tot = float(0.5 * (qber_z + qber_x))
    return qber_tot, qber_z, qber_x


# ==============================================================================
# 2. Analytical QBER Predictions
# ==============================================================================

def predict_depolarizing_qber(p: float) -> Tuple[float, float, float]:
    """Compute theoretical QBER for the depolarizing channel E(rho) = (1-p)rho + p(I/2).

    Symmetric across conjugate bases:
        QBER_Z = p / 2
        QBER_X = p / 2
        QBER_total = p / 2

    Args:
        p: Depolarizing probability parameter in [0.0, 1.0].

    Returns:
        Tuple[float, float, float]: (QBER_total, QBER_Z, QBER_X).
    """
    q = float(p / 2.0)
    return q, q, q


def predict_amplitude_damping_qber(
    gamma: float,
    f1: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute theoretical QBER for the amplitude damping channel.

    Formulas:
        QBER_Z = f1 * gamma
        QBER_X = (1 - sqrt(1 - gamma)) / 2
        QBER_total = 0.5 * (QBER_Z + QBER_X)

    Args:
        gamma: Relaxation probability in [0.0, 1.0].
        f1: Fraction of |1> states transmitted (default 0.5 for unbiased BB84).

    Returns:
        Tuple[float, float, float]: (QBER_total, QBER_Z, QBER_X).
    """
    q_z = float(f1 * gamma)
    q_x = float(0.5 * (1.0 - np.sqrt(max(0.0, 1.0 - gamma))))
    q_tot = float(0.5 * (q_z + q_x))
    return q_tot, q_z, q_x


def predict_phase_damping_qber(lam: float) -> Tuple[float, float, float]:
    """Compute theoretical QBER for the phase damping channel.

    Formulas:
        QBER_Z = 0.0 (no bit-flips in computational basis)
        QBER_X = (1 - sqrt(1 - lambda)) / 2
        QBER_total = 0.5 * QBER_X = (1 - sqrt(1 - lambda)) / 4

    Args:
        lam: Dephasing parameter in [0.0, 1.0].

    Returns:
        Tuple[float, float, float]: (QBER_total, QBER_Z, QBER_X).
    """
    q_z = 0.0
    q_x = float(0.5 * (1.0 - np.sqrt(max(0.0, 1.0 - lam))))
    q_tot = float(0.5 * q_x)
    return q_tot, q_z, q_x


def predict_thermal_relaxation_qber(
    t1: float,
    t2: float,
    time: float,
) -> Tuple[float, float, float]:
    """Compute theoretical QBER for thermal relaxation with T1 and T2 decay.

    Formulas:
        gamma = 1 - exp(-time / T1)
        QBER_Z = 0.5 * gamma = 0.5 * (1 - exp(-time / T1))
        QBER_X = 0.5 * (1 - exp(-time / T2))
        QBER_total = 0.5 * (QBER_Z + QBER_X)

    Args:
        t1: Longitudinal relaxation time > 0.
        t2: Transverse dephasing time > 0.
        time: Channel transit time >= 0.

    Returns:
        Tuple[float, float, float]: (QBER_total, QBER_Z, QBER_X).
    """
    q_z = float(0.5 * (1.0 - np.exp(-time / t1)))
    q_x = float(0.5 * (1.0 - np.exp(-time / t2)))
    q_tot = float(0.5 * (q_z + q_x))
    return q_tot, q_z, q_x


# ==============================================================================
# 3. Parameter Validation
# ==============================================================================

def validate_thermal_parameters(
    t1: float,
    t2: float,
    time: float,
    p_excited: float = 0.0,
) -> None:
    """Enforce physical validity of thermal relaxation parameters.

    Physical constraints:
    - t1 > 0
    - t2 > 0
    - time >= 0
    - t2 <= 2 * t1 (strict quantum positivity boundary)
    - 0.0 <= p_excited <= 1.0

    Raises:
        ValueError: If any constraint is violated.
    """
    if t1 <= 0.0:
        raise ValueError(f"T1 relaxation time must be strictly positive, got: {t1}")
    if t2 <= 0.0:
        raise ValueError(f"T2 dephasing time must be strictly positive, got: {t2}")
    if time < 0.0:
        raise ValueError(f"Channel interaction time must be non-negative, got: {time}")
    if t2 > (2.0 * t1 + 1e-12):
        raise ValueError(
            f"Physical violation: T2 ({t2:.3e} s) cannot exceed 2 * T1 ({2.0 * t1:.3e} s)."
        )
    if not (0.0 <= p_excited <= 1.0):
        raise ValueError(f"p_excited must be in [0.0, 1.0], got: {p_excited}")


# ==============================================================================
# 4. Qiskit Aer NoiseModel Builders
# ==============================================================================

def build_depolarizing_noise_model(
    p: float,
    gate_name: str = "id",
) -> NoiseChannelProfile:
    """Build a Qiskit Aer NoiseModel for single-qubit depolarizing error.

    Noise is applied to the specified carrier transmission gate (default 'id').

    Args:
        p: Depolarizing probability parameter in [0.0, 1.0].
        gate_name: Target instruction name representing channel transit (default 'id').

    Returns:
        NoiseChannelProfile: Profile containing NoiseModel and theoretical metrics.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0.0, 1.0], got: {p}")

    nm = NoiseModel()
    if p > 0.0:
        err = depolarizing_error(p, 1)
        nm.add_all_qubit_quantum_error(err, [gate_name])

    kraus_ops = get_depolarizing_kraus(p)
    q_tot, q_z, q_x = predict_depolarizing_qber(p)

    return NoiseChannelProfile(
        noise_type=NoiseType.DEPOLARIZING,
        parameters={"p": float(p)},
        kraus_operators=kraus_ops,
        noise_model=nm,
        theoretical_qber=q_tot,
        theoretical_qber_z=q_z,
        theoretical_qber_x=q_x,
    )


def build_amplitude_damping_noise_model(
    gamma: float,
    gate_name: str = "id",
) -> NoiseChannelProfile:
    """Build a Qiskit Aer NoiseModel for single-qubit amplitude damping.

    Args:
        gamma: Relaxation probability in [0.0, 1.0].
        gate_name: Target instruction name representing channel transit (default 'id').

    Returns:
        NoiseChannelProfile: Profile containing NoiseModel and theoretical metrics.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0.0, 1.0], got: {gamma}")

    nm = NoiseModel()
    if gamma > 0.0:
        err = amplitude_damping_error(gamma)
        nm.add_all_qubit_quantum_error(err, [gate_name])

    kraus_ops = get_amplitude_damping_kraus(gamma)
    q_tot, q_z, q_x = predict_amplitude_damping_qber(gamma)

    return NoiseChannelProfile(
        noise_type=NoiseType.AMPLITUDE_DAMPING,
        parameters={"gamma": float(gamma)},
        kraus_operators=kraus_ops,
        noise_model=nm,
        theoretical_qber=q_tot,
        theoretical_qber_z=q_z,
        theoretical_qber_x=q_x,
    )


def build_phase_damping_noise_model(
    lam: float,
    gate_name: str = "id",
) -> NoiseChannelProfile:
    """Build a Qiskit Aer NoiseModel for single-qubit phase damping.

    Args:
        lam: Dephasing probability in [0.0, 1.0].
        gate_name: Target instruction name representing channel transit (default 'id').

    Returns:
        NoiseChannelProfile: Profile containing NoiseModel and theoretical metrics.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lambda must be in [0.0, 1.0], got: {lam}")

    nm = NoiseModel()
    if lam > 0.0:
        err = phase_damping_error(lam)
        nm.add_all_qubit_quantum_error(err, [gate_name])

    kraus_ops = get_phase_damping_kraus(lam)
    q_tot, q_z, q_x = predict_phase_damping_qber(lam)

    return NoiseChannelProfile(
        noise_type=NoiseType.PHASE_DAMPING,
        parameters={"lambda": float(lam)},
        kraus_operators=kraus_ops,
        noise_model=nm,
        theoretical_qber=q_tot,
        theoretical_qber_z=q_z,
        theoretical_qber_x=q_x,
    )


def build_thermal_relaxation_noise_model(
    t1: float,
    t2: float,
    time: float,
    p_excited: float = 0.0,
    gate_name: str = "id",
) -> NoiseChannelProfile:
    """Build a Qiskit Aer NoiseModel for physical T1/T2 thermal relaxation.

    Args:
        t1: Longitudinal relaxation time > 0.
        t2: Transverse dephasing time > 0 (must satisfy t2 <= 2*t1).
        time: Channel transit time >= 0.
        p_excited: Equilibrium excited state population in [0.0, 1.0].
        gate_name: Target instruction name representing channel transit (default 'id').

    Returns:
        NoiseChannelProfile: Profile containing NoiseModel and theoretical metrics.
    """
    validate_thermal_parameters(t1, t2, time, p_excited)

    nm = NoiseModel()
    if time > 0.0:
        err = thermal_relaxation_error(
            t1=t1,
            t2=t2,
            time=time,
            excited_state_population=p_excited,
        )
        nm.add_all_qubit_quantum_error(err, [gate_name])

    kraus_ops = get_thermal_relaxation_kraus(t1, t2, time, p_excited)
    q_tot, q_z, q_x = predict_thermal_relaxation_qber(t1, t2, time)

    return NoiseChannelProfile(
        noise_type=NoiseType.THERMAL_RELAXATION,
        parameters={
            "t1": float(t1),
            "t2": float(t2),
            "time": float(time),
            "p_excited": float(p_excited),
        },
        kraus_operators=kraus_ops,
        noise_model=nm,
        theoretical_qber=q_tot,
        theoretical_qber_z=q_z,
        theoretical_qber_x=q_x,
    )


def build_combined_noise_model(
    depol_p: float = 0.0,
    amp_gamma: float = 0.0,
    phase_lam: float = 0.0,
    gate_name: str = "id",
) -> NoiseChannelProfile:
    """Build a combined Qiskit Aer NoiseModel composing depolarizing, amplitude, and phase damping.

    Args:
        depol_p: Depolarizing parameter in [0.0, 1.0].
        amp_gamma: Amplitude damping parameter in [0.0, 1.0].
        phase_lam: Phase damping parameter in [0.0, 1.0].
        gate_name: Target instruction name representing channel transit.

    Returns:
        NoiseChannelProfile: Profile containing composed NoiseModel and theoretical bounds.
    """
    if not (0.0 <= depol_p <= 1.0):
        raise ValueError(f"depol_p must be in [0.0, 1.0], got: {depol_p}")
    if not (0.0 <= amp_gamma <= 1.0):
        raise ValueError(f"amp_gamma must be in [0.0, 1.0], got: {amp_gamma}")
    if not (0.0 <= phase_lam <= 1.0):
        raise ValueError(f"phase_lam must be in [0.0, 1.0], got: {phase_lam}")

    nm = NoiseModel()
    errors_to_compose = []

    if depol_p > 0.0:
        errors_to_compose.append(depolarizing_error(depol_p, 1))
    if amp_gamma > 0.0:
        errors_to_compose.append(amplitude_damping_error(amp_gamma))
    if phase_lam > 0.0:
        errors_to_compose.append(phase_damping_error(phase_lam))

    if errors_to_compose:
        composed_error = errors_to_compose[0]
        for err in errors_to_compose[1:]:
            composed_error = composed_error.compose(err)
        nm.add_all_qubit_quantum_error(composed_error, [gate_name])
        kraus_data = [np.array(K, dtype=np.complex128) for K in Kraus(composed_error).data]
    else:
        kraus_data = [np.eye(2, dtype=np.complex128)]

    # Approximate compounded theoretical QBER
    q_depol = depol_p / 2.0
    q_amp = (amp_gamma + 1.0 - np.sqrt(max(0.0, 1.0 - amp_gamma))) / 4.0
    q_phase = (1.0 - np.sqrt(max(0.0, 1.0 - phase_lam))) / 4.0

    # Compounding formula: Q_tot = Q1 + Q2 - 2*Q1*Q2
    def compound_errors(q1: float, q2: float) -> float:
        return float(q1 + q2 - 2.0 * q1 * q2)

    q_tot = compound_errors(q_depol, compound_errors(q_amp, q_phase))
    q_z = compound_errors(depol_p / 2.0, amp_gamma / 2.0)
    q_x = compound_errors(
        depol_p / 2.0,
        compound_errors(
            (1.0 - np.sqrt(max(0.0, 1.0 - amp_gamma))) / 2.0,
            (1.0 - np.sqrt(max(0.0, 1.0 - phase_lam))) / 2.0,
        ),
    )

    return NoiseChannelProfile(
        noise_type=NoiseType.COMBINED,
        parameters={
            "depol_p": float(depol_p),
            "amp_gamma": float(amp_gamma),
            "phase_lam": float(phase_lam),
        },
        kraus_operators=kraus_data,
        noise_model=nm,
        theoretical_qber=q_tot,
        theoretical_qber_z=q_z,
        theoretical_qber_x=q_x,
    )
