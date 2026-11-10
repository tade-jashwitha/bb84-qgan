"""Quantum Channel Module for BB84 Protocol.

Re-exports QuantumChannel and TransmissionResult from src.channel.quantum_channel
for backward compatibility with Milestone 1 modules and tests.
"""

from src.channel.quantum_channel import QuantumChannel, TransmissionResult

__all__ = ["QuantumChannel", "TransmissionResult"]
