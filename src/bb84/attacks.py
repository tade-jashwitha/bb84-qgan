"""Physical Eavesdropping Attacks for BB84 Quantum Key Distribution.

This module implements physical eavesdropping strategies against the BB84 protocol:
1. Probabilistic Intercept-Resend (PIR) Attack:
   - Parameterized by interception probability alpha in [0.0, 1.0].
   - Parameterized by Eve's basis selection bias beta = P(Eve chooses Z) in [0.0, 1.0].
   - If intercepted: Eve measures in her chosen basis and prepares a replacement BB84 state.
   - If not intercepted: Eve forwards the original carrier pulse unchanged.
   - Theoretical QBER on sifted bits: QBER = alpha / 4 for any beta (symmetric key sifting).
   - Basis-conditioned QBER: QBER_Z = 0.5 * alpha * (1 - beta), QBER_X = 0.5 * alpha * beta.
   - Theoretical mutual information: I_AE(alpha) = 1 - h_2(0.5 - 0.25 * alpha).

2. Fuchs-Peres-Brandt Entangling Unitary Probe:
   - Parameterized by coupling angle theta_E in [0.0, pi / 2].
   - Carrier qubit q0 couples with Eve's probe ancilla q1 via a 2-qubit entangling unitary U_E(theta_E):
     In physical carrier-first basis |A, E> (|carrier, ancilla>):
       U_E |0>_A |0>_E = cos(theta_E) |0>_A |0>_E + sin(theta_E) |1>_A |1>_E
       U_E |1>_A |0>_E = cos(theta_E) |1>_A |0>_E - sin(theta_E) |0>_A |1>_E
     In Qiskit register basis |q1, q0> (|ancilla, carrier>):
       U_E |00>_EA = cos(theta_E) |00>_EA + sin(theta_E) |11>_EA
       U_E |01>_EA = cos(theta_E) |01>_EA - sin(theta_E) |10>_EA
   - Unitarity: U_E^\dagger U_E = I_4.
   - Bob's induced disturbance: D(theta_E) = sin^2(theta_E) across all BB84 states.
   - Analytical theoretical mutual information objective (4-state FPB optimal probe): I_AE(theta_E) = 1 - h_2((1 + sin(2 * theta_E)) / 2).

Compatible with Qiskit 2.6.0, Qiskit Aer 0.18.0, NumPy 2.4.6, SciPy 1.17.1.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from src.bb84.alice import (
    BASIS_X,
    BASIS_Z,
    VALID_BASES,
    prepare_bb84_state,
)
from src.bb84.bob import (
    apply_measurement_basis,
    measure_bb84_circuits,
    measure_pulse,
)


def binary_entropy(p: float) -> float:
    """Compute the Shannon binary entropy h_2(p).

    Args:
        p: Probability in [0.0, 1.0].

    Returns:
        float: Binary entropy -p*log2(p) - (1-p)*log2(1-p), bounded in [0.0, 1.0].
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


@dataclass(frozen=True)
class InterceptResendResult:
    """Result container for Probabilistic Intercept-Resend attack.

    Attributes:
        transmitted_circuits: Replacement and unchanged quantum circuits sent to Bob.
        intercepted_flags: Boolean 1D array indicating whether each pulse was intercepted.
        eve_bases: Array of Eve's measurement basis per pulse (None for non-intercepted).
        eve_bits: Array of Eve's measured bits (-1 for non-intercepted).
        intercept_count: Total number of intercepted pulses.
        intercept_ratio: Ratio of intercepted pulses to total pulses.
        alpha: Configured interception probability.
        beta: Configured probability of selecting Z basis.
    """

    transmitted_circuits: List[QuantumCircuit]
    intercepted_flags: np.ndarray
    eve_bases: np.ndarray
    eve_bits: np.ndarray
    intercept_count: int
    intercept_ratio: float
    alpha: float
    beta: float


@dataclass(frozen=True)
class UnitaryProbeResult:
    """Result container for Fuchs-Peres-Brandt entangling unitary probe attack.

    Attributes:
        probed_circuits: List of 2-qubit circuits containing carrier and entangled ancilla.
        theta_E: Interaction coupling angle in radians.
        unitary_matrix: 4x4 complex unitary operator matrix.
        theoretical_disturbance: Analytical disturbance contribution D(theta_E).
        theoretical_information_gain: Analytical mutual information I_AE(theta_E).
    """

    probed_circuits: List[QuantumCircuit]
    theta_E: float
    unitary_matrix: np.ndarray
    theoretical_disturbance: float
    theoretical_information_gain: float


class BaseAttack(ABC):
    """Abstract base class for physical eavesdropping attacks."""

    @abstractmethod
    def intercept_pulses(
        self,
        circuits: Sequence[QuantumCircuit],
    ) -> Any:
        """Apply the eavesdropping attack to a sequence of quantum pulses.

        Args:
            circuits: Sequence of 1-qubit carrier circuits from Alice.

        Returns:
            Attack-specific result container.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the attack strategy."""
        pass


class InterceptResendAttack(BaseAttack):
    """Probabilistic Intercept-Resend (PIR) attack on BB84 quantum channel.

    Eve intercepts each incoming carrier qubit with probability alpha in [0.0, 1.0].
    If intercepted:
    - Eve chooses basis Z with probability beta, and basis X with probability 1 - beta.
    - Eve measures the carrier state in her chosen basis.
    - Eve prepares a replacement single-qubit BB84 state corresponding to her outcome and basis.
    - Eve forwards this replacement state to Bob.
    If not intercepted:
    - Eve forwards the original carrier state untouched to Bob.

    Attributes:
        alpha: Interception probability in [0.0, 1.0].
        beta: Probability of choosing Z basis in [0.0, 1.0].
        seed: Optional random seed for reproducibility.
        rng: NumPy random Generator.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Intercept-Resend attack parameters.

        Args:
            alpha: Interception probability in [0.0, 1.0].
            beta: Probability of choosing Z basis in [0.0, 1.0].
            seed: Optional integer seed for reproducibility.

        Raises:
            ValueError: If alpha or beta are outside [0.0, 1.0].
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got: {alpha}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0.0, 1.0], got: {beta}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return f"InterceptResend(alpha={self.alpha:.3f}, beta={self.beta:.3f})"

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset the random number generator.

        Args:
            seed: Optional integer seed.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def intercept_pulses(
        self,
        circuits: Sequence[QuantumCircuit],
    ) -> InterceptResendResult:
        """Intercept and optionally resend quantum pulses.

        Args:
            circuits: Sequence of 1-qubit QuantumCircuits prepared by Alice.

        Returns:
            InterceptResendResult: Container with transformed circuits, intercept flags,
                and Eve's measurement records.
        """
        n = len(circuits)
        if n == 0:
            return InterceptResendResult(
                transmitted_circuits=[],
                intercepted_flags=np.array([], dtype=bool),
                eve_bases=np.array([], dtype="<U1"),
                eve_bits=np.array([], dtype=np.int8),
                intercept_count=0,
                intercept_ratio=0.0,
                alpha=self.alpha,
                beta=self.beta,
            )

        transmitted_circuits: List[QuantumCircuit] = []
        intercepted_flags = np.zeros(n, dtype=bool)
        eve_bases = np.full(n, "", dtype="<U1")
        eve_bits = np.full(n, -1, dtype=np.int8)

        # Pre-sample random values for efficiency and reproducibility
        intercept_decisions = self.rng.random(size=n) < self.alpha
        basis_decisions = self.rng.random(size=n) < self.beta

        # Separate indices for intercepted pulses
        intercepted_indices = [i for i in range(n) if intercept_decisions[i]]

        # Prepare measurement for intercepted pulses
        if intercepted_indices:
            intercepted_circuits = [circuits[i] for i in intercepted_indices]
            intercepted_bases = [
                BASIS_Z if basis_decisions[i] else BASIS_X
                for i in intercepted_indices
            ]
            sim_seed = int(self.rng.integers(0, 2**31 - 1)) if self.seed is not None else None
            measured_outcomes = measure_bb84_circuits(
                intercepted_circuits,
                intercepted_bases,
                seed_simulator=sim_seed,
            )
        else:
            intercepted_bases = []
            measured_outcomes = np.array([], dtype=np.int8)

        # Assemble output sequence preserving exact signal order
        inter_idx_counter = 0
        for i in range(n):
            if intercept_decisions[i]:
                intercepted_flags[i] = True
                e_basis = intercepted_bases[inter_idx_counter]
                e_bit = int(measured_outcomes[inter_idx_counter])
                inter_idx_counter += 1

                eve_bases[i] = e_basis
                eve_bits[i] = e_bit

                # Prepare replacement state for Bob
                replacement_qc = prepare_bb84_state(bit=e_bit, basis=e_basis)
                transmitted_circuits.append(replacement_qc)
            else:
                # Forward original state unchanged
                intercepted_flags[i] = False
                eve_bases[i] = ""
                eve_bits[i] = -1
                transmitted_circuits.append(circuits[i].copy())

        intercept_count = int(np.sum(intercepted_flags))
        intercept_ratio = float(intercept_count / n)

        return InterceptResendResult(
            transmitted_circuits=transmitted_circuits,
            intercepted_flags=intercepted_flags,
            eve_bases=eve_bases,
            eve_bits=eve_bits,
            intercept_count=intercept_count,
            intercept_ratio=intercept_ratio,
            alpha=self.alpha,
            beta=self.beta,
        )

    def theoretical_qber(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> float:
        """Compute the theoretical asymptotic QBER on sifted bits.

        Formula:
            QBER = alpha / 4
        Proof:
            QBER_Z = 0.5 * alpha * (1 - beta)
            QBER_X = 0.5 * alpha * beta
            QBER = 0.5 * QBER_Z + 0.5 * QBER_X = 0.25 * alpha

        Args:
            alpha: Optional override for alpha.
            beta: Optional override for beta (does not alter symmetric total QBER).

        Returns:
            float: Expected asymptotic QBER.
        """
        a = self.alpha if alpha is None else float(alpha)
        return float(a / 4.0)

    def theoretical_qber_by_basis(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute theoretical asymptotic QBER conditioned on Z and X sifted subsets.

        Formulas:
            QBER_Z = 0.5 * alpha * (1 - beta)
            QBER_X = 0.5 * alpha * beta

        Args:
            alpha: Optional override for alpha.
            beta: Optional override for beta.

        Returns:
            Tuple[float, float]: (QBER_Z, QBER_X).
        """
        a = self.alpha if alpha is None else float(alpha)
        b = self.beta if beta is None else float(beta)
        qber_z = float(0.5 * a * (1.0 - b))
        qber_x = float(0.5 * a * b)
        return qber_z, qber_x

    def theoretical_information_gain(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> float:
        """Compute Eve's theoretical mutual information I_AE per sifted bit.

        Formula:
            For unbiased basis selection (beta = 0.5):
                P_e = 0.5 - 0.25 * alpha
                I_AE = 1 - h_2(P_e)
            For general beta:
                P_e,Z = 0.5 - 0.5 * alpha * beta
                P_e,X = 0.5 - 0.5 * alpha * (1 - beta)
                I_AE = 0.5 * (1 - h_2(P_e,Z)) + 0.5 * (1 - h_2(P_e,X))

        Args:
            alpha: Optional override for alpha.
            beta: Optional override for beta.

        Returns:
            float: Theoretical mutual information in [0.0, 1.0] bits.
        """
        a = self.alpha if alpha is None else float(alpha)
        b = self.beta if beta is None else float(beta)

        p_e_z = 0.5 - 0.5 * a * b
        p_e_x = 0.5 - 0.5 * a * (1.0 - b)

        i_z = 1.0 - binary_entropy(p_e_z)
        i_x = 1.0 - binary_entropy(p_e_x)

        return float(0.5 * (i_z + i_x))

    def compute_empirical_information_gain(
        self,
        alice_bits: Sequence[int],
        sifted_indices: Sequence[int],
        eve_bits: Sequence[int],
    ) -> float:
        """Compute empirical mutual information I(A; E) on sifted bits.

        Formula:
            I(A; E) = H(A) - H(A | E) = 1.0 - h_2(P_error)
            where P_error = sum(a != e) / |S|

        Args:
            alice_bits: Full array of Alice's bits.
            sifted_indices: Indices of pulses retained in sifting.
            eve_bits: Eve's measured bits (-1 where not intercepted).

        Returns:
            float: Empirical mutual information in bits.
        """
        if len(sifted_indices) == 0:
            return 0.0

        a_bits = np.array(alice_bits, dtype=np.int8)[sifted_indices]
        e_bits = np.array(eve_bits, dtype=np.int8)[sifted_indices]

        # For non-intercepted pulses, Eve has no information (random coin flip error = 0.5)
        # Empirical error rate:
        # If intercepted: error if e_bit != a_bit.
        # If not intercepted: error with probability 0.5.
        intercepted_sifted = (e_bits != -1)
        n_sifted = len(sifted_indices)
        n_inter = np.sum(intercepted_sifted)

        if n_inter > 0:
            inter_errors = np.sum(a_bits[intercepted_sifted] != e_bits[intercepted_sifted])
        else:
            inter_errors = 0

        non_inter_count = n_sifted - n_inter
        expected_total_errors = inter_errors + 0.5 * non_inter_count
        p_error = float(expected_total_errors / n_sifted)

        return float(max(0.0, 1.0 - binary_entropy(p_error)))


def build_probe_unitary(theta_E: float) -> np.ndarray:
    """Construct the 4x4 Fuchs-Peres-Brandt entangling unitary matrix.

    Ordered in Qiskit standard 2-qubit basis |q1, q0> in {|00>, |01>, |10>, |11>}:
    - q0: Carrier qubit (Alice's transmitted pulse).
    - q1: Ancilla probe qubit (Eve's probe initialized in |0>).

    Joint Transformation (for ancilla input |0>_E):
        In physical carrier-first basis |A, E> (|carrier, ancilla>):
            U_E |0>_A |0>_E = cos(theta_E) |0>_A |0>_E + sin(theta_E) |1>_A |1>_E
            U_E |1>_A |0>_E = cos(theta_E) |1>_A |0>_E - sin(theta_E) |0>_A |1>_E
        In Qiskit register basis |q1, q0> (|ancilla, carrier> in {|00>, |01>, |10>, |11>}):
            Col 0 (input |00>_EA):  cos(theta_E) |00>_EA + sin(theta_E) |11>_EA
            Col 1 (input |01>_EA):  cos(theta_E) |01>_EA - sin(theta_E) |10>_EA
            Col 2 (input |10>_EA):  cos(theta_E) |10>_EA + sin(theta_E) |01>_EA
            Col 3 (input |11>_EA):  cos(theta_E) |11>_EA - sin(theta_E) |00>_EA

    In matrix form:
        U_E = [
            [ cos(theta_E),             0,             0, -sin(theta_E) ],
            [            0,  cos(theta_E),  sin(theta_E),             0 ],
            [            0, -sin(theta_E),  cos(theta_E),             0 ],
            [ sin(theta_E),             0,             0,  cos(theta_E) ]
        ]

    Args:
        theta_E: Interaction coupling parameter in [0.0, pi / 2].

    Returns:
        np.ndarray: Complex 4x4 unitary matrix.
    """
    c = float(np.cos(theta_E))
    s = float(np.sin(theta_E))
    return np.array([
        [ c,  0.0,  0.0, -s],
        [0.0,   c,    s, 0.0],
        [0.0,  -s,    c, 0.0],
        [ s,  0.0,  0.0,  c],
    ], dtype=np.complex128)


class FuchsPeresBrandtProbe(BaseAttack):
    """Fuchs-Peres-Brandt entangling unitary probe for BB84 eavesdropping.

    Eve attaches an ancillary probe qubit |0>_E to Alice's carrier qubit |psi>_A
    via a 2-qubit entangling unitary U_E(theta_E):
        U_E(theta_E) |0>_A |0>_E = cos(theta_E) |0>_A |0>_E + sin(theta_E) |1>_A |1>_E
        U_E(theta_E) |1>_A |0>_E = cos(theta_E) |1>_A |0>_E - sin(theta_E) |0>_A |1>_E

    Properties:
    - Carrier q0 is forwarded to Bob (who measures in basis x_B).
    - Ancilla q1 is retained by Eve (who measures after classical basis announcement).
    - Induced disturbance: D(theta_E) = sin^2(theta_E) across all conjugate BB84 bases.
    - Eve mutual information: I_AE(theta_E) = 1 - h_2((1 + sin(2 * theta_E)) / 2).

    Attributes:
        theta_E: Interaction coupling parameter in [0.0, pi / 2].
        unitary_matrix: 4x4 complex unitary matrix U_E.
    """

    def __init__(self, theta_E: float = 0.0) -> None:
        """Initialize the unitary probe with coupling angle theta_E.

        Args:
            theta_E: Coupling angle in radians within [0.0, pi / 2].

        Raises:
            ValueError: If theta_E is outside [0.0, pi / 2].
        """
        if not (0.0 <= theta_E <= (np.pi / 2.0 + 1e-9)):
            raise ValueError(f"theta_E must be in [0.0, pi/2], got: {theta_E}")

        self.theta_E = float(theta_E)
        self.unitary_matrix = build_probe_unitary(self.theta_E)
        self._validate_unitarity()

    @property
    def name(self) -> str:
        return f"FuchsPeresBrandtProbe(theta_E={self.theta_E:.4f} rad)"

    def _validate_unitarity(self) -> None:
        """Verify U^\dagger U = I_4 to high precision."""
        product = self.unitary_matrix.conj().T @ self.unitary_matrix
        diff = np.max(np.abs(product - np.eye(4, dtype=np.complex128)))
        if diff > 1e-12:
            raise ValueError(
                f"Probe matrix is not unitary: max|U^\dagger U - I| = {diff}"
            )

    def theoretical_disturbance(self) -> float:
        """Compute the theoretical average disturbance D(theta_E) across BB84 bases.

        Mathematical Audit Correction (Milestone 2):
            For the Fuchs-Peres-Brandt entangling unitary U_E:
                U_E |0>_A |0>_E = cos(theta_E) |0>_A |0>_E + sin(theta_E) |1>_A |1>_E
                U_E |1>_A |0>_E = cos(theta_E) |1>_A |0>_E - sin(theta_E) |0>_A |1>_E
            The carrier reduced density matrix after partial trace over Eve's ancilla is:
                rho_A(|psi>) = cos^2(theta_E) |psi><psi| + sin^2(theta_E) |psi_perp><psi_perp|
            for every BB84 state |psi> in {|0>, |1>, |+>, |->}.
            On sifted pulses (where Bob measures in the matching basis), Bob observes an
            error with probability P_error = <psi_perp| rho_A |psi_perp> = sin^2(theta_E).
            Thus, the exact disturbance is D(theta_E) = sin^2(theta_E).
        """
        return float(np.sin(self.theta_E) ** 2)

    def theoretical_information_gain(self) -> float:
        """Compute Eve's theoretical mutual information I_AE(theta_E).

        Theoretical Derivation & Scope:
            This formula evaluates the analytical theoretical mutual information objective for
            the symmetric Fuchs-Peres-Brandt optimal probe in a 4-dimensional (2-qubit) ancilla
            space under post-sifting Helstrom state discrimination:
                I_AE(theta_E) = 1 - h_2((1 + sin(2 * theta_E)) / 2)
            where sin(2 * theta_E) = 2 * sqrt(D * (1 - D)) for disturbance D = sin^2(theta_E).
            It represents an analytical theoretical objective under optimal probe and delayed-
            measurement assumptions (not a universal upper bound across arbitrary unconstrained
            attacks).

            Distinction from 2-Qubit Physical Circuit Simulation:
            In the 2-qubit circuit implementation (1 carrier + 1 ancilla), the unitary
            acts on the carrier as a coherent Y-error channel with D = sin^2(theta_E).
            The 1-qubit ancilla state is identical across all four BB84 states:
                rho_E = cos^2(theta_E) |0><0| + sin^2(theta_E) |1><1|
            and serves strictly as an error herald (|0>=no error, |1>=error) with zero
            physically accessible mutual information (I_phys = 0.0). This method provides the
            analytical theoretical proxy quantity used by the cGAN objective and benchmark
            scoring corresponding to the selected disturbance angle theta_E.
        """
        p_succ = float((1.0 + np.sin(2.0 * self.theta_E)) / 2.0)
        return float(max(0.0, 1.0 - binary_entropy(p_succ)))

    def compute_reduced_states(
        self,
        carrier_amplitudes: Sequence[complex],
    ) -> Tuple[DensityMatrix, DensityMatrix, Statevector]:
        """Compute joint state and reduced density matrices for a given pure carrier input.

        Args:
            carrier_amplitudes: 2-element complex amplitude vector [alpha, beta].

        Returns:
            Tuple[DensityMatrix, DensityMatrix, Statevector]:
                - rho_A: Bob's reduced density matrix (tracing out ancilla q1).
                - rho_E: Eve's reduced density matrix (tracing out carrier q0).
                - psi_joint: 4-element joint Statevector |Psi_out>.
        """
        c_arr = np.array(carrier_amplitudes, dtype=np.complex128)
        norm = np.linalg.norm(c_arr)
        if not np.isclose(norm, 1.0, atol=1e-7):
            raise ValueError(f"Input carrier state must be normalized, got norm: {norm}")

        # Joint initial state |q1=0, q0=carrier> = |0>_E \otimes |carrier>_A
        # In Qiskit basis |q1, q0>: [alpha, beta, 0, 0]^T
        psi_in = np.array([c_arr[0], c_arr[1], 0.0, 0.0], dtype=np.complex128)
        psi_out = self.unitary_matrix @ psi_in
        psi_joint = Statevector(psi_out)

        rho_joint = DensityMatrix(psi_joint)
        # Trace out q1 (ancilla) to obtain Bob's carrier state rho_A
        rho_A = partial_trace(rho_joint, [1])
        # Trace out q0 (carrier) to obtain Eve's ancilla state rho_E
        rho_E = partial_trace(rho_joint, [0])

        return rho_A, rho_E, psi_joint

    def apply_probe_to_circuit(
        self,
        carrier_qc: QuantumCircuit,
    ) -> QuantumCircuit:
        """Entangle a single incoming 1-qubit carrier circuit with Eve's probe ancilla.

        Creates a 2-qubit circuit where:
        - Qubit 0: Carrier qubit (forwarded to Bob).
        - Qubit 1: Ancilla probe qubit (Eve's ancilla).

        Args:
            carrier_qc: 1-qubit QuantumCircuit.

        Returns:
            QuantumCircuit: 2-qubit QuantumCircuit with probe gate appended.
        """
        if carrier_qc.num_qubits != 1:
            raise ValueError(
                f"Carrier circuit must have exactly 1 qubit, got: {carrier_qc.num_qubits}"
            )

        probed_qc = QuantumCircuit(2, name=f"{carrier_qc.name}_probed")
        # Compose carrier preparation onto qubit 0
        probed_qc.compose(carrier_qc, [0], inplace=True)

        # Append 2-qubit entangling unitary gate
        probe_gate = UnitaryGate(self.unitary_matrix, label="U_probe")
        probed_qc.append(probe_gate, [0, 1])

        return probed_qc

    def intercept_pulses(
        self,
        circuits: Sequence[QuantumCircuit],
    ) -> UnitaryProbeResult:
        """Entangle all incoming carrier circuits with ancillary probe qubits.

        Args:
            circuits: Sequence of 1-qubit carrier circuits.

        Returns:
            UnitaryProbeResult: Container with 2-qubit probed circuits and analytical metrics.
        """
        probed_circuits = [self.apply_probe_to_circuit(qc) for qc in circuits]

        return UnitaryProbeResult(
            probed_circuits=probed_circuits,
            theta_E=self.theta_E,
            unitary_matrix=self.unitary_matrix,
            theoretical_disturbance=self.theoretical_disturbance(),
            theoretical_information_gain=self.theoretical_information_gain(),
        )
