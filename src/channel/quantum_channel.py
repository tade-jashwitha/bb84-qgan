"""Quantum Channel Module for BB84 Protocol.

This module models the physical transmission link connecting Alice, an optional
eavesdropper (Eve), and Bob:
1. Direct Transmission: Qubits traverse undisturbed (noiseless baseline) or subject
   to NISQ channel noise (depolarizing, amplitude damping, phase damping, thermal relaxation).
2. Intercepted Transmission: Carrier qubits are intercepted by an attack operator
   (e.g., Probabilistic Intercept-Resend or Entangling Unitary Probe).
3. Configurable noise placement:
   - "transit" / "after_eve": Carrier pulse experiences channel transit noise.
   - "before_eve": Noise corrupts carrier pulse before Eve intercepts.
   - "distributed": Channel noise is split across link segments.
4. Single-qubit and batch pulse transmission modes.

Compatible with Qiskit 2.6.0, Qiskit Aer 0.18.0.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from src.channel.noise_models import NoiseChannelProfile


@dataclass(frozen=True)
class TransmissionResult:
    """Container for quantum channel transmission outcomes.

    Attributes:
        received_circuits: Quantum circuits delivered to Bob.
        is_eavesdropped: Boolean indicating whether an active attack was applied.
        attack_name: Identifier of the eavesdropping strategy (or 'None').
        attack_metadata: Diagnostic and empirical metrics from the eavesdropper.
        is_noisy: Boolean indicating whether channel noise was active during transmission.
        noise_type: Name/type of active noise channel (or 'None').
    """

    received_circuits: List[QuantumCircuit]
    is_eavesdropped: bool
    attack_name: str
    attack_metadata: Dict[str, Any] = field(default_factory=dict)
    is_noisy: bool = False
    noise_type: Optional[str] = None


class QuantumChannel:
    """Quantum channel representing the physical optical carrier link in BB84.

    Attributes:
        attack: Optional eavesdropping attack attached to the channel.
        noise_profile: Optional NoiseChannelProfile containing Kraus ops and Qiskit NoiseModel.
        noise_placement: Where channel noise acts relative to eavesdropper
            ('transit', 'after_eve', 'before_eve', 'distributed').
    """

    def __init__(
        self,
        attack: Optional[Any] = None,
        noise_model: Optional[Union[NoiseChannelProfile, NoiseModel]] = None,
        noise_placement: str = "transit",
    ) -> None:
        """Initialize the quantum channel with optional attack and noise model.

        Args:
            attack: Optional BaseAttack instance (Eve).
            noise_model: Optional NoiseChannelProfile or Qiskit Aer NoiseModel.
            noise_placement: Noise placement strategy ('transit', 'after_eve', 'before_eve').
        """
        self.attack: Optional[Any] = attack
        self._noise_profile: Optional[NoiseChannelProfile] = None
        self._noise_model: Optional[NoiseModel] = None
        self.noise_placement: str = noise_placement

        if noise_model is not None:
            self.set_noise(noise_model, noise_placement=noise_placement)

    def set_attack(self, attack: Optional[Any]) -> None:
        """Attach or update the channel's eavesdropping attack.

        Args:
            attack: Optional BaseAttack instance.
        """
        self.attack = attack

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
        if noise is None:
            self._noise_profile = None
            self._noise_model = None
        elif isinstance(noise, NoiseChannelProfile):
            self._noise_profile = noise
            self._noise_model = noise.noise_model
        elif isinstance(noise, NoiseModel):
            self._noise_profile = None
            self._noise_model = noise
        else:
            raise TypeError(
                f"Expected NoiseChannelProfile or NoiseModel, got: {type(noise).__name__}"
            )

        self.noise_placement = noise_placement

    def has_noise(self) -> bool:
        """Return True if an active noise model is configured on the channel."""
        return self._noise_model is not None

    def get_noise_model(self) -> Optional[NoiseModel]:
        """Return the active Qiskit Aer NoiseModel, or None."""
        return self._noise_model

    def get_noise_profile(self) -> Optional[NoiseChannelProfile]:
        """Return the active NoiseChannelProfile, or None."""
        return self._noise_profile

    def apply_channel_transit_noise(
        self,
        circuits: Sequence[QuantumCircuit],
    ) -> List[QuantumCircuit]:
        """Insert channel transit marker (id gate on carrier qubit 0) into circuits.

        In Qiskit Aer, quantum errors registered on instruction 'id' for qubit 0
        will take effect during circuit execution on the noisy simulator.

        Args:
            circuits: Sequence of QuantumCircuit objects.

        Returns:
            List[QuantumCircuit]: Copies of circuits with id(0) appended.
        """
        noisy_circuits = []
        for qc in circuits:
            new_qc = qc.copy(name=f"{qc.name}_transit")
            new_qc.id(0)
            noisy_circuits.append(new_qc)
        return noisy_circuits

    def transmit_pulse(
        self,
        circuit: QuantumCircuit,
    ) -> QuantumCircuit:
        """Transmit a single quantum pulse through the channel.

        Args:
            circuit: 1-qubit QuantumCircuit from Alice.

        Returns:
            QuantumCircuit: Transmitted circuit arriving at Bob.
        """
        batch_result = self.transmit_batch([circuit])
        return batch_result.received_circuits[0]

    def transmit_batch(
        self,
        circuits: Sequence[QuantumCircuit],
    ) -> TransmissionResult:
        """Transmit a batch of quantum pulses through the channel.

        If an attack is configured, Eve intercepts and processes pulses according
        to her attack strategy and the noise placement configuration.

        Args:
            circuits: Sequence of QuantumCircuits from Alice.

        Returns:
            TransmissionResult: Result object containing delivered circuits and attack diagnostics.
        """
        n = len(circuits)
        noise_name = self._noise_profile.noise_type.value if self._noise_profile else (
            "AerNoiseModel" if self._noise_model else "None"
        )
        is_noisy = self.has_noise()

        if n == 0:
            return TransmissionResult(
                received_circuits=[],
                is_eavesdropped=False,
                attack_name="None",
                attack_metadata={},
                is_noisy=is_noisy,
                noise_type=noise_name if is_noisy else None,
            )

        if self.attack is None:
            # Direct transmission Alice -> Bob
            if is_noisy:
                received = self.apply_channel_transit_noise(circuits)
            else:
                received = [qc.copy() for qc in circuits]

            return TransmissionResult(
                received_circuits=received,
                is_eavesdropped=False,
                attack_name="None",
                attack_metadata={"loss": 0.0, "interception_count": 0},
                is_noisy=is_noisy,
                noise_type=noise_name if is_noisy else None,
            )

        # Eve is present on the channel - lazy import to avoid circular dependency
        from src.bb84.attacks import InterceptResendAttack, FuchsPeresBrandtProbe

        if isinstance(self.attack, InterceptResendAttack):
            if is_noisy and self.noise_placement == "before_eve":
                # Channel noise acts between Alice and Eve
                intercept_circuits = self.apply_channel_transit_noise(circuits)
                attack_res = self.attack.intercept_pulses(intercept_circuits)
                received = attack_res.transmitted_circuits
            else:
                # Default / transit / after_eve: Alice -> Eve is direct, Eve -> Bob has transit noise
                attack_res = self.attack.intercept_pulses(circuits)
                if is_noisy:
                    received = self.apply_channel_transit_noise(attack_res.transmitted_circuits)
                else:
                    received = attack_res.transmitted_circuits

            metadata = {
                "alpha": attack_res.alpha,
                "beta": attack_res.beta,
                "intercept_count": attack_res.intercept_count,
                "intercept_ratio": attack_res.intercept_ratio,
                "intercepted_flags": attack_res.intercepted_flags,
                "eve_bases": attack_res.eve_bases,
                "eve_bits": attack_res.eve_bits,
            }

        elif isinstance(self.attack, FuchsPeresBrandtProbe):
            # For FuchsPeresBrandtProbe, Bob must measure joint entangled state.
            # BB84Engine directly constructs the joint circuit.
            attack_res = self.attack.intercept_pulses(circuits)
            received = attack_res.probed_circuits
            metadata = {
                "theta_E": attack_res.theta_E,
                "theoretical_disturbance": attack_res.theoretical_disturbance,
                "theoretical_information_gain": attack_res.theoretical_information_gain,
                "unitary_matrix": attack_res.unitary_matrix,
            }

        else:
            # Generic fallback
            attack_res = self.attack.intercept_pulses(circuits)
            received = getattr(attack_res, "transmitted_circuits", [qc.copy() for qc in circuits])
            if is_noisy:
                received = self.apply_channel_transit_noise(received)
            metadata = getattr(attack_res, "__dict__", {})

        return TransmissionResult(
            received_circuits=received,
            is_eavesdropped=True,
            attack_name=self.attack.name,
            attack_metadata=metadata,
            is_noisy=is_noisy,
            noise_type=noise_name if is_noisy else None,
        )
