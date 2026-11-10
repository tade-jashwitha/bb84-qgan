"""Alice Module for BB84 Quantum Key Distribution.

This module implements Alice's state preparation in the BB84 protocol:
1. Generation of random classical bits b_A in {0, 1}.
2. Uniform random selection of conjugate measurement bases x_A in {'Z', 'X'}.
3. Quantum state preparation for the four BB84 states:
   - |0> (bit=0, basis='Z')
   - |1> (bit=1, basis='Z')
   - |+> (bit=0, basis='X')
   - |-> (bit=1, basis='X')

Compatible with Qiskit 2.x.
"""

from typing import List, Optional, Sequence, Union
import numpy as np
from qiskit import QuantumCircuit

# Standard BB84 Basis Identifiers
BASIS_Z: str = "Z"  # Computational basis: {|0>, |1>}
BASIS_X: str = "X"  # Hadamard / Diagonal basis: {|+>, |->}
VALID_BASES = (BASIS_Z, BASIS_X)
VALID_BITS = (0, 1)


def generate_random_bits(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate an array of uniform random binary bits b_A in {0, 1}.

    Args:
        n: Number of bits to generate (must be a positive integer).
        rng: Optional NumPy random Generator instance for reproducibility.

    Returns:
        np.ndarray: 1D array of integers with shape (n,) containing bits 0 and 1.

    Raises:
        ValueError: If n is not a positive integer.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError(f"Number of bits n must be a positive integer, got: {n}")
    
    if rng is None:
        rng = np.random.default_rng()
    
    return rng.integers(0, 2, size=n, dtype=np.int8)


def generate_random_bases(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate an array of uniform random basis choices x_A in {'Z', 'X'}.

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
    
    # 0 maps to BASIS_Z ('Z'), 1 maps to BASIS_X ('X')
    basis_indices = rng.integers(0, 2, size=n, dtype=np.int8)
    basis_lookup = np.array([BASIS_Z, BASIS_X], dtype="<U1")
    return basis_lookup[basis_indices]


def prepare_bb84_state(bit: int, basis: str) -> QuantumCircuit:
    """Prepare a single-qubit QuantumCircuit representing a BB84 state.

    The four BB84 states are prepared from the initial ground state |0> as follows:
    - bit=0, basis='Z' -> |0> = [1, 0]^T
        Circuit: Identity (no gate applied, qubit remains in |0>).
    - bit=1, basis='Z' -> |1> = X|0> = [0, 1]^T
        Circuit: X gate on qubit 0.
    - bit=0, basis='X' -> |+> = H|0> = (|0> + |1>)/sqrt(2) = [1/sqrt(2), 1/sqrt(2)]^T
        Circuit: H gate on qubit 0.
    - bit=1, basis='X' -> |-> = H X|0> = (|0> - |1>)/sqrt(2) = [1/sqrt(2), -1/sqrt(2)]^T
        Circuit: X gate followed by H gate on qubit 0.
        (Relative phase between |0> and |1> is pi, giving amplitudes [1/sqrt(2), -1/sqrt(2)]).

    Args:
        bit: Integer 0 or 1.
        basis: String 'Z' or 'X' (case-insensitive).

    Returns:
        QuantumCircuit: 1-qubit Qiskit circuit preparing the requested state.

    Raises:
        ValueError: If bit is not in {0, 1} or basis is not in {'Z', 'X'}.
    """
    if bit not in VALID_BITS:
        raise ValueError(f"Invalid bit value: {bit}. Must be 0 or 1.")
    
    basis_normalized = str(basis).strip().upper()
    if basis_normalized not in VALID_BASES:
        raise ValueError(f"Invalid basis: '{basis}'. Must be '{BASIS_Z}' or '{BASIS_X}'.")
    
    circuit_name = f"bb84_state_{bit}_{basis_normalized}"
    qc = QuantumCircuit(1, name=circuit_name)
    
    if basis_normalized == BASIS_Z:
        if bit == 1:
            qc.x(0)
        # bit == 0: ground state |0>, no gates required
    elif basis_normalized == BASIS_X:
        if bit == 0:
            qc.h(0)
        else:  # bit == 1
            # |-> = H|1> = H X|0> = (|0> - |1>) / sqrt(2)
            qc.x(0)
            qc.h(0)
            
    return qc


def prepare_bb84_circuits(
    bits: Sequence[int],
    bases: Sequence[str],
) -> List[QuantumCircuit]:
    """Prepare a list of single-qubit QuantumCircuits for sequences of bits and bases.

    Args:
        bits: Sequence of integers (0 or 1) of length N.
        bases: Sequence of strings ('Z' or 'X') of length N.

    Returns:
        List[QuantumCircuit]: List of N QuantumCircuits.

    Raises:
        ValueError: If lengths of bits and bases do not match.
    """
    if len(bits) != len(bases):
        raise ValueError(
            f"Length mismatch: bits length is {len(bits)}, but bases length is {len(bases)}."
        )
    
    return [prepare_bb84_state(b, basis) for b, basis in zip(bits, bases)]


class Alice:
    """Alice agent class for BB84 state preparation and transmission.

    Encapsulates random bit and basis generation, state preparation, and
    transmission recording for reproducible QKD simulation.

    Attributes:
        seed: Optional random seed for reproducible sequence generation.
        rng: NumPy random Generator instance.
        bits: Generated classical bit sequence b_A (populated after generate_bits).
        bases: Generated basis sequence x_A (populated after generate_bases).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize Alice with an optional random seed.

        Args:
            seed: Optional integer seed for reproducible random generation.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.bits: Optional[np.ndarray] = None
        self.bases: Optional[np.ndarray] = None

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset the random number generator with a new seed.

        Args:
            seed: Optional integer seed.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_bits(self, n: int) -> np.ndarray:
        """Generate and store n random binary bits.

        Args:
            n: Number of pulses / bits to generate.

        Returns:
            np.ndarray: Generated bit sequence b_A.
        """
        self.bits = generate_random_bits(n, self.rng)
        return self.bits

    def generate_bases(self, n: int) -> np.ndarray:
        """Generate and store n random basis selections ('Z' or 'X').

        Args:
            n: Number of bases to generate.

        Returns:
            np.ndarray: Generated basis sequence x_A.
        """
        self.bases = generate_random_bases(n, self.rng)
        return self.bases

    def prepare_pulses(self, n: Optional[int] = None) -> List[QuantumCircuit]:
        """Generate bits, bases (if not already generated), and return prepared quantum circuits.

        Args:
            n: Number of pulses to prepare. Required if bits and bases have not been generated.

        Returns:
            List[QuantumCircuit]: List of prepared 1-qubit QuantumCircuits.

        Raises:
            ValueError: If n is not provided and bits/bases have not been pre-generated.
        """
        if n is not None:
            self.generate_bits(n)
            self.generate_bases(n)
        
        if self.bits is None or self.bases is None:
            raise ValueError(
                "Bits and bases must be generated before preparing pulses. Provide n."
            )
            
        return prepare_bb84_circuits(self.bits, self.bases)
