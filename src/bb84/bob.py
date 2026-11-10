"""Bob Module for BB84 Quantum Key Distribution.

This module implements Bob's measurement process, basis reconciliation (sifting),
and Quantum Bit Error Rate (QBER) estimation in the BB84 protocol:
1. Independent random basis selection x_B in {'Z', 'X'}.
2. Quantum measurement in conjugate bases:
   - Z-basis (computational basis): direct projective measurement.
   - X-basis (Hadamard basis): Hadamard gate H applied prior to measurement.
     H|+> = |0>, H|-> = |1>.
3. Structured recording of measurement outcomes, bases, and signal order.
4. Classical basis reconciliation / sifting comparing public basis announcements
   without comparing private bit values.
5. Sifted QBER calculation with robust handling of edge cases and validation.

Compatible with Qiskit 2.6.0 and Qiskit Aer 0.18.0.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator

# Standard BB84 Basis Identifiers (consistent with Alice module)
BASIS_Z: str = "Z"  # Computational basis: {|0>, |1>}
BASIS_X: str = "X"  # Hadamard / Diagonal basis: {|+>, |->}
VALID_BASES = (BASIS_Z, BASIS_X)
VALID_BITS = (0, 1)


@dataclass(frozen=True)
class SiftingResult:
    """Container for basis reconciliation (sifting) results.

    Attributes:
        alice_sifted_bits: 1D array of Alice's bits at matching basis positions.
        bob_sifted_bits: 1D array of Bob's bits at matching basis positions.
        sifted_indices: 1D array of transmission indices where bases matched.
        total_signals: Total number of transmitted signals compared.
        matching_bases_count: Number of signals with matching bases.
        sifting_efficiency: Ratio of retained signals (matching_bases_count / total_signals).
    """

    alice_sifted_bits: np.ndarray
    bob_sifted_bits: np.ndarray
    sifted_indices: np.ndarray
    total_signals: int
    matching_bases_count: int
    sifting_efficiency: float


@dataclass(frozen=True)
class QBERResult:
    """Container for Quantum Bit Error Rate (QBER) estimation results.

    Attributes:
        qber: Quantum Bit Error Rate in [0.0, 1.0].
        error_count: Number of bit mismatches in evaluated sifted key.
        total_bits: Number of sifted bits evaluated.
    """

    qber: float
    error_count: int
    total_bits: int


def generate_random_bases(
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate an array of uniform random basis choices x_B in {'Z', 'X'}.

    Bob selects his measurement bases independently of Alice.

    Args:
        n: Number of bases to generate (must be a positive integer).
        rng: Optional NumPy random Generator instance for reproducibility.

    Returns:
        np.ndarray: 1D array of strings with shape (n,) containing 'Z' and 'X'.

    Raises:
        ValueError: If n is not a positive integer.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError(f"Number of bases n must be a positive integer, got: {n}")

    if rng is None:
        rng = np.random.default_rng()

    basis_indices = rng.integers(0, 2, size=n, dtype=np.int8)
    basis_lookup = np.array([BASIS_Z, BASIS_X], dtype="<U1")
    return basis_lookup[basis_indices]


def apply_measurement_basis(
    circuit: QuantumCircuit,
    basis: str,
) -> QuantumCircuit:
    """Attach conjugate basis measurement operators to a single-qubit circuit.

    Mathematical Basis Transformation:
    - Z basis (computational): Direct measurement in {|0>, |1>}.
      Outcome 0 corresponds to |0>, outcome 1 corresponds to |1>.
    - X basis (Hadamard): Apply H gate prior to computational measurement.
      H|+> = H [ (|0> + |1>) / sqrt(2) ] = |0> -> measured as 0.
      H|-> = H [ (|0> - |1>) / sqrt(2) ] = |1> -> measured as 1.

    Args:
        circuit: 1-qubit Qiskit QuantumCircuit representing the incoming pulse.
        basis: Measurement basis string ('Z' or 'X', case-insensitive).

    Returns:
        QuantumCircuit: A new circuit with the measurement basis transformation
            and classical measurement appended.

    Raises:
        TypeError: If circuit is not an instance of QuantumCircuit.
        ValueError: If circuit does not have exactly 1 qubit or basis is invalid.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(
            f"Expected circuit to be a QuantumCircuit, got: {type(circuit).__name__}"
        )

    if circuit.num_qubits != 1:
        raise ValueError(
            f"BB84 pulse circuit must have exactly 1 qubit, got: {circuit.num_qubits}"
        )

    basis_normalized = str(basis).strip().upper()
    if basis_normalized not in VALID_BASES:
        raise ValueError(
            f"Invalid basis: '{basis}'. Must be '{BASIS_Z}' or '{BASIS_X}'."
        )

    meas_qc = circuit.copy(name=f"{circuit.name}_meas_{basis_normalized}")

    # Ensure classical register exists for measurement
    if meas_qc.num_clbits == 0:
        meas_qc.add_register(ClassicalRegister(1, "meas"))

    # Apply basis change if measuring in Hadamard basis
    if basis_normalized == BASIS_X:
        meas_qc.h(0)

    # Projective measurement in computational basis
    meas_qc.measure(0, 0)

    return meas_qc


def measure_pulse(
    circuit: QuantumCircuit,
    basis: str,
    backend: Optional[Any] = None,
    seed_simulator: Optional[int] = None,
) -> int:
    """Measure a single incoming BB84 quantum pulse in the specified basis.

    Args:
        circuit: 1-qubit QuantumCircuit.
        basis: Measurement basis ('Z' or 'X').
        backend: Optional Qiskit backend (defaults to AerSimulator).
        seed_simulator: Optional integer seed for simulator reproducibility.

    Returns:
        int: Measured bit value (0 or 1).
    """
    meas_qc = apply_measurement_basis(circuit, basis)
    if backend is None:
        backend = AerSimulator()

    result = backend.run(
        meas_qc,
        shots=1,
        memory=True,
        seed_simulator=seed_simulator,
    ).result()

    memory = result.get_memory()
    return int(memory[0])


def measure_bb84_circuits(
    circuits: Sequence[QuantumCircuit],
    bases: Sequence[str],
    backend: Optional[Any] = None,
    seed_simulator: Optional[int] = None,
) -> np.ndarray:
    """Measure a batch of incoming BB84 quantum circuits using Bob's bases.

    Preserves the exact transmission order of the incoming signals.

    Args:
        circuits: Sequence of 1-qubit QuantumCircuits representing incoming pulses.
        bases: Sequence of measurement bases ('Z' or 'X') of identical length.
        backend: Optional Qiskit backend (defaults to AerSimulator).
        seed_simulator: Optional integer seed for the simulator.

    Returns:
        np.ndarray: 1D array of measured bits b_B in {0, 1} with dtype np.int8.

    Raises:
        ValueError: If lengths of circuits and bases do not match.
    """
    if len(circuits) != len(bases):
        raise ValueError(
            f"Length mismatch: {len(circuits)} circuits provided, but {len(bases)} bases provided."
        )

    n = len(circuits)
    if n == 0:
        return np.array([], dtype=np.int8)

    meas_circuits = [
        apply_measurement_basis(qc, b) for qc, b in zip(circuits, bases)
    ]

    if backend is None:
        backend = AerSimulator()

    result = backend.run(
        meas_circuits,
        shots=1,
        memory=True,
        seed_simulator=seed_simulator,
    ).result()

    # Extract single-shot measurement for each circuit in order
    measured_bits = np.empty(n, dtype=np.int8)
    for i in range(n):
        mem = result.get_memory(i)
        measured_bits[i] = int(mem[0])

    return measured_bits


def reconcile_bases(
    alice_bases: Sequence[str],
    bob_bases: Sequence[str],
    alice_bits: Optional[Sequence[int]] = None,
    bob_bits: Optional[Sequence[int]] = None,
) -> SiftingResult:
    """Perform classical basis reconciliation (sifting).

    Compares Alice's and Bob's public basis announcements without inspecting
    or comparing private bit values. A signal is retained if and only if
    Alice's basis equals Bob's basis (x_A[i] == x_B[i]).

    Args:
        alice_bases: Sequence of Alice's bases ('Z' or 'X').
        bob_bases: Sequence of Bob's bases ('Z' or 'X').
        alice_bits: Optional sequence of Alice's transmitted bits.
        bob_bits: Optional sequence of Bob's measured bits.

    Returns:
        SiftingResult: Data container with sifted bits, indices, counts,
            and sifting efficiency.

    Raises:
        ValueError: If base lengths mismatch, bits lengths mismatch,
            or invalid base labels / bit values are encountered.
    """
    if len(alice_bases) != len(bob_bases):
        raise ValueError(
            f"Length mismatch: Alice announced {len(alice_bases)} bases, "
            f"but Bob has {len(bob_bases)} bases."
        )

    n = len(alice_bases)

    # Validate basis labels
    norm_alice_bases = []
    norm_bob_bases = []
    for i in range(n):
        a_b = str(alice_bases[i]).strip().upper()
        b_b = str(bob_bases[i]).strip().upper()
        if a_b not in VALID_BASES:
            raise ValueError(f"Invalid Alice basis at index {i}: '{alice_bases[i]}'")
        if b_b not in VALID_BASES:
            raise ValueError(f"Invalid Bob basis at index {i}: '{bob_bases[i]}'")
        norm_alice_bases.append(a_b)
        norm_bob_bases.append(b_b)

    # Optional bit validation
    if alice_bits is not None:
        if len(alice_bits) != n:
            raise ValueError(
                f"Length mismatch: Alice bases length is {n}, "
                f"but Alice bits length is {len(alice_bits)}."
            )
        for i, b in enumerate(alice_bits):
            if b not in VALID_BITS:
                raise ValueError(f"Invalid Alice bit value at index {i}: {b}")

    if bob_bits is not None:
        if len(bob_bits) != n:
            raise ValueError(
                f"Length mismatch: Bob bases length is {n}, "
                f"but Bob bits length is {len(bob_bits)}."
            )
        for i, b in enumerate(bob_bits):
            if b not in VALID_BITS:
                raise ValueError(f"Invalid Bob bit value at index {i}: {b}")

    # Sifting rule: Retain signal if bases match
    matching_indices = [
        i for i in range(n) if norm_alice_bases[i] == norm_bob_bases[i]
    ]
    sifted_indices = np.array(matching_indices, dtype=np.int64)
    matching_count = len(matching_indices)
    efficiency = float(matching_count / n) if n > 0 else 0.0

    if alice_bits is not None:
        alice_arr = np.array(alice_bits, dtype=np.int8)
        alice_sifted = alice_arr[sifted_indices] if matching_count > 0 else np.array([], dtype=np.int8)
    else:
        alice_sifted = np.array([], dtype=np.int8)

    if bob_bits is not None:
        bob_arr = np.array(bob_bits, dtype=np.int8)
        bob_sifted = bob_arr[sifted_indices] if matching_count > 0 else np.array([], dtype=np.int8)
    else:
        bob_sifted = np.array([], dtype=np.int8)

    return SiftingResult(
        alice_sifted_bits=alice_sifted,
        bob_sifted_bits=bob_sifted,
        sifted_indices=sifted_indices,
        total_signals=n,
        matching_bases_count=matching_count,
        sifting_efficiency=efficiency,
    )


def compute_qber(
    alice_sifted_bits: Sequence[int],
    bob_sifted_bits: Sequence[int],
) -> QBERResult:
    """Calculate the Quantum Bit Error Rate (QBER) between sifted keys.

    Formula:
        QBER = Number of bit mismatches / Total evaluated sifted bits

    Args:
        alice_sifted_bits: Sequence of Alice's sifted bits in {0, 1}.
        bob_sifted_bits: Sequence of Bob's sifted bits in {0, 1}.

    Returns:
        QBERResult: Contains qber, error_count, and total_bits.
            If total_bits is 0, returns qber=0.0.

    Raises:
        ValueError: If sequence lengths do not match or if elements
            contain invalid bit values not in {0, 1}.
    """
    if len(alice_sifted_bits) != len(bob_sifted_bits):
        raise ValueError(
            f"Length mismatch in sifted keys: Alice has {len(alice_sifted_bits)} bits, "
            f"but Bob has {len(bob_sifted_bits)} bits."
        )

    n = len(alice_sifted_bits)
    if n == 0:
        return QBERResult(qber=0.0, error_count=0, total_bits=0)

    # Validate bit values
    for i in range(n):
        a_b = alice_sifted_bits[i]
        b_b = bob_sifted_bits[i]
        if a_b not in VALID_BITS:
            raise ValueError(
                f"Invalid bit value in Alice's sifted key at index {i}: {a_b}. Must be 0 or 1."
            )
        if b_b not in VALID_BITS:
            raise ValueError(
                f"Invalid bit value in Bob's sifted key at index {i}: {b_b}. Must be 0 or 1."
            )

    a_arr = np.array(alice_sifted_bits, dtype=np.int8)
    b_arr = np.array(bob_sifted_bits, dtype=np.int8)

    error_count = int(np.sum(a_arr != b_arr))
    qber = float(error_count / n)

    return QBERResult(
        qber=qber,
        error_count=error_count,
        total_bits=n,
    )


class Bob:
    """Bob agent class for BB84 measurement and basis reconciliation.

    Encapsulates random basis selection, quantum circuit measurement in
    conjugate bases ({'Z', 'X'}), public basis announcement/sifting,
    and QBER estimation.

    Attributes:
        seed: Optional random seed for reproducible sequence generation.
        rng: NumPy random Generator instance.
        backend: Qiskit backend instance (defaults to AerSimulator).
        bases: Generated basis sequence x_B (populated after generate_bases).
        measured_bits: Measured bit sequence b_B (populated after measure_pulses).
        sifted_bits: Bob's sifted key bits (populated after sift).
        sifted_indices: Indices of matching basis pulses (populated after sift).
        metadata: Dictionary containing execution details and run parameters.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        backend: Optional[Any] = None,
    ) -> None:
        """Initialize Bob with optional seed and backend.

        Args:
            seed: Optional integer seed for reproducibility.
            backend: Optional Qiskit backend (defaults to AerSimulator).
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.backend = backend if backend is not None else AerSimulator()
        self.bases: Optional[np.ndarray] = None
        self.measured_bits: Optional[np.ndarray] = None
        self.sifted_bits: Optional[np.ndarray] = None
        self.sifted_indices: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset the random number generator with a new seed.

        Args:
            seed: Optional integer seed.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_bases(self, n: int) -> np.ndarray:
        """Generate and store n random basis choices ('Z' or 'X').

        Args:
            n: Number of bases to generate.

        Returns:
            np.ndarray: Generated basis sequence x_B.
        """
        self.bases = generate_random_bases(n, self.rng)
        return self.bases

    def measure_pulses(
        self,
        circuits: Sequence[QuantumCircuit],
        bases: Optional[Sequence[str]] = None,
        seed_simulator: Optional[int] = None,
        backend: Optional[Any] = None,
    ) -> np.ndarray:
        """Measure a sequence of incoming quantum circuits using Bob's bases.

        If bases are not provided, uses previously generated bases or
        automatically generates them matching the number of circuits.

        Args:
            circuits: Sequence of incoming 1-qubit QuantumCircuits.
            bases: Optional sequence of measurement bases ('Z' or 'X').
            seed_simulator: Optional integer seed for simulator reproducibility.
            backend: Optional Qiskit backend override.

        Returns:
            np.ndarray: Array of measured bits b_B in {0, 1}.
        """
        n = len(circuits)
        if bases is not None:
            if len(bases) != n:
                raise ValueError(
                    f"Length mismatch: {n} circuits, but {len(bases)} bases provided."
                )
            self.bases = np.array([str(b).strip().upper() for b in bases], dtype="<U1")
        elif self.bases is None or len(self.bases) != n:
            self.generate_bases(n)

        exec_backend = backend if backend is not None else self.backend

        self.measured_bits = measure_bb84_circuits(
            circuits=circuits,
            bases=self.bases,
            backend=exec_backend,
            seed_simulator=seed_simulator,
        )

        self.metadata = {
            "total_pulses": n,
            "backend": getattr(exec_backend, "name", str(exec_backend)),
            "seed": self.seed,
            "seed_simulator": seed_simulator,
        }

        return self.measured_bits

    def sift(
        self,
        alice_bases: Sequence[str],
        alice_bits: Optional[Sequence[int]] = None,
    ) -> SiftingResult:
        """Perform basis reconciliation using Alice's public basis announcement.

        Args:
            alice_bases: Sequence of Alice's announced bases.
            alice_bits: Optional sequence of Alice's bits for key extraction.

        Returns:
            SiftingResult: Result of sifting containing matching indices and keys.

        Raises:
            ValueError: If Bob has not yet generated bases and measured pulses.
        """
        if self.bases is None or self.measured_bits is None:
            raise ValueError(
                "Bob must have generated bases and measured pulses before sifting."
            )

        result = reconcile_bases(
            alice_bases=alice_bases,
            bob_bases=self.bases,
            alice_bits=alice_bits,
            bob_bits=self.measured_bits,
        )

        self.sifted_bits = result.bob_sifted_bits
        self.sifted_indices = result.sifted_indices
        return result

    def compute_qber(
        self,
        alice_sifted_bits: Sequence[int],
    ) -> QBERResult:
        """Estimate QBER between Alice's sifted sample and Bob's sifted key.

        Args:
            alice_sifted_bits: Alice's sifted bit sequence.

        Returns:
            QBERResult: Computed QBER metrics.

        Raises:
            ValueError: If Bob has not performed sifting.
        """
        if self.sifted_bits is None:
            raise ValueError(
                "Bob has no sifted bits. Perform sifting before computing QBER."
            )

        return compute_qber(
            alice_sifted_bits=alice_sifted_bits,
            bob_sifted_bits=self.sifted_bits,
        )

