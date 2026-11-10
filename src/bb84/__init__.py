"""BB84 quantum key distribution protocol package."""

from src.bb84.alice import (
    Alice,
    generate_random_bits,
    generate_random_bases as generate_alice_bases,
    prepare_bb84_state,
    prepare_bb84_circuits,
    BASIS_Z,
    BASIS_X,
    VALID_BASES,
    VALID_BITS,
)
from src.bb84.bob import (
    Bob,
    SiftingResult,
    QBERResult,
    generate_random_bases as generate_bob_bases,
    apply_measurement_basis,
    measure_pulse,
    measure_bb84_circuits,
    reconcile_bases,
    compute_qber,
)
from src.bb84.channel import (
    QuantumChannel,
    TransmissionResult,
)
from src.bb84.attacks import (
    BaseAttack,
    InterceptResendAttack,
    InterceptResendResult,
    FuchsPeresBrandtProbe,
    UnitaryProbeResult,
    build_probe_unitary,
    binary_entropy,
)
from src.bb84.bb84_engine import (
    BB84Engine,
    BB84SimulationResult,
)

__all__ = [
    # Alice
    "Alice",
    "generate_random_bits",
    "generate_alice_bases",
    "prepare_bb84_state",
    "prepare_bb84_circuits",
    # Bob
    "Bob",
    "generate_bob_bases",
    "apply_measurement_basis",
    "measure_pulse",
    "measure_bb84_circuits",
    "reconcile_bases",
    "compute_qber",
    "SiftingResult",
    "QBERResult",
    # Channel
    "QuantumChannel",
    "TransmissionResult",
    # Attacks
    "BaseAttack",
    "InterceptResendAttack",
    "InterceptResendResult",
    "FuchsPeresBrandtProbe",
    "UnitaryProbeResult",
    "build_probe_unitary",
    "binary_entropy",
    # Engine
    "BB84Engine",
    "BB84SimulationResult",
    # Constants
    "BASIS_Z",
    "BASIS_X",
    "VALID_BASES",
    "VALID_BITS",
]
