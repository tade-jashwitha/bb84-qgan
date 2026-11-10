"""BB84 Protocol Engine Module.

This module coordinates the complete end-to-end execution of the BB84 Quantum
Key Distribution (QKD) protocol:
1. Quantum state preparation (Alice).
2. Quantum transmission through an optical channel (QuantumChannel).
3. Realistic NISQ channel noise (Depolarizing, Amplitude Damping, Phase Damping, Thermal).
4. Eavesdropping interception and perturbation (Eve).
5. Independent conjugate basis measurement (Bob).
6. Public basis reconciliation (classical sifting).
7. Sifted key extraction and Quantum Bit Error Rate (QBER) estimation.
8. Sub-basis error diagnostics (Z-basis and X-basis QBER).
9. Information gain and attack parameter tracking.

Strict Isolation Guarantee:
- Alice's private bits and bases are never accessible to Eve or Bob during transmission.
- Bob selects measurement bases independently of Alice.
- Basis reconciliation compares only public basis choices, never secret key bits.
- Channel noise acts exclusively on the optical carrier, never corrupting Eve's probe ancilla.

Compatible with Qiskit 2.6.0, Qiskit Aer 0.18.0, NumPy 2.4.6.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.bb84.alice import Alice, BASIS_X, BASIS_Z, prepare_bb84_state
from src.bb84.bob import Bob, SiftingResult, QBERResult, reconcile_bases, compute_qber
from src.bb84.channel import QuantumChannel, TransmissionResult
from src.channel.noise_models import NoiseChannelProfile
from src.bb84.attacks import (
    BaseAttack,
    InterceptResendAttack,
    InterceptResendResult,
    FuchsPeresBrandtProbe,
    UnitaryProbeResult,
)


@dataclass(frozen=True)
class BB84SimulationResult:
    """Comprehensive result of an end-to-end BB84 protocol simulation run.

    Attributes:
        num_signals: Total number of quantum pulses transmitted.
        alice_bits: Alice's generated raw classical bits b_A in {0, 1}^N.
        alice_bases: Alice's chosen preparation bases x_A in {Z, X}^N.
        bob_bases: Bob's chosen measurement bases x_B in {Z, X}^N.
        bob_measured_bits: Bob's recorded measurement outcomes b_B in {0, 1}^N.
        sifted_indices: Indices where bases matched (x_A[i] == x_B[i]).
        sifted_key_length: Total number of sifted bits |S|.
        sifting_efficiency: Fraction of transmitted signals retained in sifting (|S| / N).
        alice_sifted_bits: Alice's secret key bits on matching basis positions.
        bob_sifted_bits: Bob's secret key bits on matching basis positions.
        qber: Overall Quantum Bit Error Rate on the sifted key.
        error_count: Total number of bit discrepancies in the sifted key.
        qber_z: QBER evaluated strictly on Z-basis sifted signals.
        error_count_z: Number of errors in Z-basis sifted signals.
        sifted_count_z: Number of sifted bits in Z basis.
        qber_x: QBER evaluated strictly on X-basis sifted signals.
        error_count_x: Number of errors in X-basis sifted signals.
        sifted_count_x: Number of sifted bits in X basis.
        attack_applied: Boolean indicating if an eavesdropper was active.
        attack_name: String description of the attack (or 'None').
        attack_metadata: Diagnostic dictionary containing attack statistics.
        eve_information_gain: Eve's mutual information I(A; E) on sifted key.
        is_noisy: Boolean indicating whether channel noise was active.
        noise_type: Identifier of the channel noise model (or 'None').
    """

    num_signals: int
    alice_bits: np.ndarray
    alice_bases: np.ndarray
    bob_bases: np.ndarray
    bob_measured_bits: np.ndarray
    sifted_indices: np.ndarray
    sifted_key_length: int
    sifting_efficiency: float
    alice_sifted_bits: np.ndarray
    bob_sifted_bits: np.ndarray
    qber: float
    error_count: int
    qber_z: float
    error_count_z: int
    sifted_count_z: int
    qber_x: float
    error_count_x: int
    sifted_count_x: int
    attack_applied: bool
    attack_name: str
    attack_metadata: Dict[str, Any]
    eve_information_gain: float
    is_noisy: bool = False
    noise_type: Optional[str] = None


class BB84Engine:
    """End-to-end coordinator engine for BB84 Quantum Key Distribution.

    Attributes:
        alice: Alice agent instance for state preparation.
        bob: Bob agent instance for measurement and sifting.
        channel: QuantumChannel instance connecting Alice, Eve, and Bob.
        backend: Qiskit backend for execution (defaults to AerSimulator).
        noise_placement: Position of noise relative to eavesdropping.
    """

    def __init__(
        self,
        alice_seed: Optional[int] = None,
        bob_seed: Optional[int] = None,
        backend: Optional[Any] = None,
        noise: Optional[Union[NoiseChannelProfile, NoiseModel]] = None,
        noise_placement: str = "transit",
    ) -> None:
        """Initialize the BB84 protocol engine.

        Args:
            alice_seed: Optional seed for Alice's random generator.
            bob_seed: Optional seed for Bob's random generator.
            backend: Optional Qiskit backend override.
            noise: Optional NoiseChannelProfile or Qiskit Aer NoiseModel.
            noise_placement: Strategy for noise placement ('transit', 'after_eve', 'before_eve').
        """
        self.noise_placement = noise_placement
        self.channel = QuantumChannel(noise_model=noise, noise_placement=noise_placement)

        if backend is not None:
            self.backend = backend
        elif self.channel.has_noise():
            self.backend = AerSimulator(noise_model=self.channel.get_noise_model())
        else:
            self.backend = AerSimulator()

        self.alice = Alice(seed=alice_seed)
        self.bob = Bob(seed=bob_seed, backend=self.backend)

    def set_noise(
        self,
        noise: Optional[Union[NoiseChannelProfile, NoiseModel]],
        noise_placement: str = "transit",
    ) -> None:
        """Configure or clear channel noise model.

        Args:
            noise: NoiseChannelProfile or Qiskit Aer NoiseModel, or None to clear.
            noise_placement: Strategy for noise placement ('transit', 'after_eve', 'before_eve').
        """
        self.noise_placement = noise_placement
        self.channel.set_noise(noise, noise_placement=noise_placement)
        if self.channel.has_noise():
            self.backend = AerSimulator(noise_model=self.channel.get_noise_model())
        else:
            self.backend = AerSimulator()
        self.bob.backend = self.backend

    def reset_seeds(
        self,
        alice_seed: Optional[int] = None,
        bob_seed: Optional[int] = None,
    ) -> None:
        """Reset Alice's and Bob's random number generators with new seeds.

        Args:
            alice_seed: New integer seed for Alice.
            bob_seed: New integer seed for Bob.
        """
        self.alice.reset_rng(seed=alice_seed)
        self.bob.reset_rng(seed=bob_seed)

    def run(
        self,
        n_signals: int,
        attack: Optional[BaseAttack] = None,
        seed_simulator: Optional[int] = None,
        noise: Optional[Union[NoiseChannelProfile, NoiseModel]] = None,
    ) -> BB84SimulationResult:
        """Execute a full end-to-end simulation of the BB84 protocol.

        Workflow:
        1. Alice generates n random classical bits b_A and bases x_A.
        2. Alice prepares single-qubit quantum circuits for all pulses.
        3. Carrier pulses are transmitted via the QuantumChannel with optional Eve attack and noise.
        4. Bob independently generates bases x_B and measures incoming signals.
        5. Alice and Bob perform public classical sifting on matching bases.
        6. Compute overall and basis-split QBER metrics and attack diagnostics.

        Args:
            n_signals: Number of quantum pulses to transmit (must be > 0).
            attack: Optional BaseAttack instance representing an active eavesdropper.
            seed_simulator: Optional integer seed for simulator reproducibility.
            noise: Optional per-run NoiseChannelProfile or NoiseModel override.

        Returns:
            BB84SimulationResult: Complete protocol execution records.

        Raises:
            ValueError: If n_signals is not a positive integer.
            TypeError: If attack is not None and not an instance of BaseAttack.
        """
        if not isinstance(n_signals, (int, np.integer)) or n_signals <= 0:
            raise ValueError(
                f"Number of signals must be a positive integer, got: {n_signals}"
            )

        if attack is not None and not isinstance(attack, BaseAttack):
            raise TypeError(
                f"attack must be an instance of BaseAttack, got: {type(attack).__name__}"
            )

        # Configure active channel and simulator backend for this run
        if noise is not None:
            active_channel = QuantumChannel(noise_model=noise, noise_placement=self.noise_placement)
            active_backend = AerSimulator(noise_model=active_channel.get_noise_model())
            is_noisy = True
            noise_name = active_channel.get_noise_profile().noise_type.value if active_channel.get_noise_profile() else "AerNoiseModel"
        else:
            active_channel = self.channel
            active_backend = self.backend
            is_noisy = active_channel.has_noise()
            noise_name = active_channel.get_noise_profile().noise_type.value if active_channel.get_noise_profile() else (
                "AerNoiseModel" if is_noisy else None
            )

        # 1. Alice Stage: Generate bits, bases, and prepare quantum circuits
        alice_bits = self.alice.generate_bits(n_signals)
        alice_bases = self.alice.generate_bases(n_signals)
        alice_circuits = self.alice.prepare_pulses()

        # 2. Bob Stage: Independently generate measurement bases
        bob_bases = self.bob.generate_bases(n_signals)

        # 3. Transmission & Eavesdropping Stage
        if attack is None:
            # Transmission without eavesdropper
            trans_res = active_channel.transmit_batch(alice_circuits)
            bob_measured_bits = self.bob.measure_pulses(
                trans_res.received_circuits,
                bases=bob_bases,
                seed_simulator=seed_simulator,
                backend=active_backend,
            )
            attack_applied = False
            attack_name = "None"
            attack_metadata: Dict[str, Any] = {"status": "No active eavesdropper"}
            eve_info_gain = 0.0

        elif isinstance(attack, InterceptResendAttack):
            # Probabilistic Intercept-Resend attack
            active_channel.set_attack(attack)
            trans_res = active_channel.transmit_batch(alice_circuits)
            bob_measured_bits = self.bob.measure_pulses(
                trans_res.received_circuits,
                bases=bob_bases,
                seed_simulator=seed_simulator,
                backend=active_backend,
            )

            attack_applied = True
            attack_name = attack.name
            meta = trans_res.attack_metadata

            attack_metadata = {
                "alpha": attack.alpha,
                "beta": attack.beta,
                "intercept_count": meta.get("intercept_count", 0),
                "intercept_ratio": meta.get("intercept_ratio", 0.0),
                "theoretical_qber": attack.theoretical_qber(),
                "theoretical_qber_z": attack.theoretical_qber_by_basis()[0],
                "theoretical_qber_x": attack.theoretical_qber_by_basis()[1],
                "theoretical_information_gain": attack.theoretical_information_gain(),
                "eve_bases": meta.get("eve_bases"),
                "eve_bits": meta.get("eve_bits"),
            }
            eve_info_gain = attack.theoretical_information_gain()

        elif isinstance(attack, FuchsPeresBrandtProbe):
            # Fuchs-Peres-Brandt entangling unitary probe
            # Bob measures carrier qubit 0 in basis x_B[i]
            # Ancilla qubit 1 is entangled with carrier
            meas_circuits: List[QuantumCircuit] = []
            gate = UnitaryGate(attack.unitary_matrix, label="U_probe")

            for i in range(n_signals):
                qc = QuantumCircuit(2, 1, name=f"probe_meas_{i}")
                # Compose Alice's carrier preparation onto qubit 0
                qc.compose(alice_circuits[i], [0], inplace=True)
                # If channel is noisy, apply transit noise to carrier qubit 0
                if is_noisy:
                    qc.id(0)
                # Entangle carrier q0 with Eve's probe ancilla q1
                qc.append(gate, [0, 1])

                # Bob's measurement on carrier qubit 0
                if bob_bases[i] == BASIS_X:
                    qc.h(0)
                qc.measure(0, 0)
                meas_circuits.append(qc)

            exec_res = active_backend.run(
                meas_circuits,
                shots=1,
                memory=True,
                seed_simulator=seed_simulator,
            ).result()

            bob_measured_bits = np.empty(n_signals, dtype=np.int8)
            for i in range(n_signals):
                mem = exec_res.get_memory(i)
                bob_measured_bits[i] = int(mem[0])

            attack_applied = True
            attack_name = attack.name
            attack_metadata = {
                "theta_E": attack.theta_E,
                "theoretical_disturbance": attack.theoretical_disturbance(),
                "theoretical_information_gain": attack.theoretical_information_gain(),
            }
            eve_info_gain = attack.theoretical_information_gain()

        else:
            # Generic fallback for custom attacks
            active_channel.set_attack(attack)
            trans_res = active_channel.transmit_batch(alice_circuits)
            bob_measured_bits = self.bob.measure_pulses(
                trans_res.received_circuits,
                bases=bob_bases,
                seed_simulator=seed_simulator,
                backend=active_backend,
            )
            attack_applied = True
            attack_name = attack.name
            attack_metadata = trans_res.attack_metadata
            eve_info_gain = 0.0

        # 4. Classical Sifting Stage: Basis reconciliation
        sift_res = reconcile_bases(
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            alice_bits=alice_bits,
            bob_bits=bob_measured_bits,
        )

        # 5. QBER Calculation Stage
        qber_res = compute_qber(
            alice_sifted_bits=sift_res.alice_sifted_bits,
            bob_sifted_bits=sift_res.bob_sifted_bits,
        )

        # 6. Sub-Basis Diagnostics: Separate Z and X sifted errors
        sifted_indices = sift_res.sifted_indices
        if len(sifted_indices) > 0:
            alice_sifted_bases = alice_bases[sifted_indices]
            z_mask = (alice_sifted_bases == BASIS_Z)
            x_mask = (alice_sifted_bases == BASIS_X)

            sifted_count_z = int(np.sum(z_mask))
            sifted_count_x = int(np.sum(x_mask))

            if sifted_count_z > 0:
                errors_z = int(np.sum(
                    sift_res.alice_sifted_bits[z_mask] != sift_res.bob_sifted_bits[z_mask]
                ))
                qber_z = float(errors_z / sifted_count_z)
            else:
                errors_z = 0
                qber_z = 0.0

            if sifted_count_x > 0:
                errors_x = int(np.sum(
                    sift_res.alice_sifted_bits[x_mask] != sift_res.bob_sifted_bits[x_mask]
                ))
                qber_x = float(errors_x / sifted_count_x)
            else:
                errors_x = 0
                qber_x = 0.0
        else:
            sifted_count_z = 0
            sifted_count_x = 0
            errors_z = 0
            errors_x = 0
            qber_z = 0.0
            qber_x = 0.0

        # If Intercept-Resend, update empirical information gain
        if isinstance(attack, InterceptResendAttack) and len(sifted_indices) > 0:
            eve_bits = attack_metadata.get("eve_bits")
            if eve_bits is not None:
                empirical_info = attack.compute_empirical_information_gain(
                    alice_bits=alice_bits,
                    sifted_indices=sifted_indices,
                    eve_bits=eve_bits,
                )
                attack_metadata["empirical_information_gain"] = empirical_info

        return BB84SimulationResult(
            num_signals=n_signals,
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            bob_measured_bits=bob_measured_bits,
            sifted_indices=sifted_indices,
            sifted_key_length=sift_res.matching_bases_count,
            sifting_efficiency=sift_res.sifting_efficiency,
            alice_sifted_bits=sift_res.alice_sifted_bits,
            bob_sifted_bits=sift_res.bob_sifted_bits,
            qber=qber_res.qber,
            error_count=qber_res.error_count,
            qber_z=qber_z,
            error_count_z=errors_z,
            sifted_count_z=sifted_count_z,
            qber_x=qber_x,
            error_count_x=errors_x,
            sifted_count_x=sifted_count_x,
            attack_applied=attack_applied,
            attack_name=attack_name,
            attack_metadata=attack_metadata,
            eve_information_gain=eve_info_gain,
            is_noisy=is_noisy,
            noise_type=noise_name,
        )
