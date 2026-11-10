"""Dataset schema specification and containers for BB84 attack policies.

This module defines the versioned, validated data structures for collecting,
evaluating, and storing BB84 eavesdropping simulation records under NISQ noise:
1. `AttackFamily`: Enumeration of physical attack families (NULL, UNITARY_PROBE, PIR).
2. `StatisticalFeasibility`: Three-state feasibility category (FEASIBLE, INFEASIBLE, INDETERMINATE).
3. `AttackParameters`: Explicit, validated attack parameters.
4. `EnvironmentalCondition`: Ambient noise and protocol error budget specifications.
5. `RawObservation`: Primary simulation observations directly measured by Alice and Bob.
6. `DerivedEvaluation`: Statistically derived evaluation metrics and feasibility labels.
7. `DatasetRecord`: Comprehensive single-simulation record combining condition, policy, raw outcomes, and derived labels.
8. `DatasetContainer`: Versioned container wrapping a collection of records with dataset-level metadata.

Schema Version: 1.0.0
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import math


class AttackFamily(str, Enum):
    """Categorical enumeration of physical attack families."""

    NULL = "Null"
    UNITARY_PROBE = "Unitary Probe"
    PROBABILISTIC_INTERCEPT_RESEND = "Probabilistic Intercept-Resend"


class StatisticalFeasibility(str, Enum):
    """Three-state statistical feasibility classification under Wilson score interval."""

    FEASIBLE = "Feasible"
    INFEASIBLE = "Infeasible"
    INDETERMINATE = "Indeterminate"


@dataclass(frozen=True)
class AttackParameters:
    """Validated attack parameters for both PIR and Unitary Probe families.

    Attributes:
        alpha: Interception ratio in [0.0, 1.0] for PIR (None for Unitary Probe / Null).
        beta: Basis bias P(Basis = Z) in [0.0, 1.0] for PIR (None for Unitary Probe / Null).
        theta_E_rad: Entangling coupling angle in radians in [0.0, pi/4] for Unitary Probe.
        theta_E_deg: Entangling coupling angle in degrees (derived from theta_E_rad).
    """

    alpha: Optional[float] = None
    beta: Optional[float] = None
    theta_E_rad: Optional[float] = None
    theta_E_deg: Optional[float] = None

    def __post_init__(self):
        if self.alpha is not None and not (0.0 <= self.alpha <= 1.0 + 1e-9):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {self.alpha}")
        if self.beta is not None and not (0.0 <= self.beta <= 1.0 + 1e-9):
            raise ValueError(f"beta must be in [0.0, 1.0], got {self.beta}")
        if self.theta_E_rad is not None and not (0.0 <= self.theta_E_rad <= (math.pi / 4.0 + 1e-9)):
            raise ValueError(f"theta_E_rad must be in [0.0, pi/4], got {self.theta_E_rad}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a clean dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "theta_E_rad": self.theta_E_rad,
            "theta_E_deg": self.theta_E_deg,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackParameters":
        """Reconstruct AttackParameters from dictionary."""
        return cls(
            alpha=data.get("alpha"),
            beta=data.get("beta"),
            theta_E_rad=data.get("theta_E_rad"),
            theta_E_deg=data.get("theta_E_deg"),
        )


@dataclass(frozen=True)
class AttackConfiguration:
    """Specification of an attack configuration in the deterministic lattice.

    Attributes:
        attack_id: Canonical identifier (e.g., 'ATT-00' through 'ATT-15').
        family: Physical attack family.
        parameters: Specific validated parameters.
        theoretical_disturbance: Analytical disturbance D(Attack).
        description: Human-readable description.
    """

    attack_id: str
    family: AttackFamily
    parameters: AttackParameters
    theoretical_disturbance: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "attack_id": self.attack_id,
            "family": self.family.value,
            "parameters": self.parameters.to_dict(),
            "theoretical_disturbance": self.theoretical_disturbance,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackConfiguration":
        """Reconstruct AttackConfiguration from dictionary."""
        return cls(
            attack_id=data["attack_id"],
            family=AttackFamily(data["family"]),
            parameters=AttackParameters.from_dict(data["parameters"]),
            theoretical_disturbance=float(data["theoretical_disturbance"]),
            description=data["description"],
        )


@dataclass(frozen=True)
class EnvironmentalCondition:
    """Environmental channel noise and protocol security threshold condition.

    Attributes:
        condition_id: Unique condition identifier (e.g. 'C01').
        p_depol: Microscopic depolarizing channel error rate in [0.0, 1.0].
        q_env: Theoretical ambient channel QBER (p_depol / 2) for unattacked channel.
        q_budget: Target protocol abort threshold (e.g., 0.05, 0.08, 0.11).
        gamma_noise_hat: Eve's empirical estimate of ambient QBER from calibration pulses.
        sigma_calib: Standard error of Eve's calibration estimate.
        n_calib: Number of public calibration pilot pulses observed.
        k_calib: Observed error count during public calibration pilot pulses.
        seed_calib: Random seed used during calibration pilot sampling.
    """

    condition_id: str
    p_depol: float
    q_env: float
    q_budget: float
    gamma_noise_hat: float
    sigma_calib: float
    n_calib: int = 1000
    k_calib: int = 0
    seed_calib: Optional[int] = None
    partition: Optional[str] = None

    @property
    def sigma_estimated(self) -> float:
        """Observed plug-in standard error: sqrt(gamma_hat * (1 - gamma_hat) / n_calib).

        Collapses to 0.0 at the boundary gamma_hat = 0 (Condition C01).
        """
        if self.gamma_noise_hat <= 0.0 or self.gamma_noise_hat >= 1.0 or self.n_calib <= 0:
            return 0.0
        return float(math.sqrt(self.gamma_noise_hat * (1.0 - self.gamma_noise_hat) / float(self.n_calib)))

    def get_observable_condition(self) -> Tuple[float, float]:
        """Return Eve's observable conditioning vector c = (Q_budget, gamma_noise_hat).

        Strictly excludes ground-truth simulator parameters (p_depol, q_env).
        """
        return (self.q_budget, self.gamma_noise_hat)

    def to_dict(self) -> Dict[str, Any]:
        """Convert condition to dictionary."""
        return {
            "condition_id": self.condition_id,
            "p_depol": self.p_depol,
            "q_env": self.q_env,
            "q_budget": self.q_budget,
            "gamma_noise_hat": self.gamma_noise_hat,
            "sigma_calib": self.sigma_calib,
            "n_calib": self.n_calib,
            "k_calib": self.k_calib,
            "seed_calib": self.seed_calib,
            "partition": self.partition,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentalCondition":
        """Reconstruct EnvironmentalCondition from dictionary."""
        return cls(
            condition_id=data["condition_id"],
            p_depol=float(data["p_depol"]),
            q_env=float(data["q_env"]),
            q_budget=float(data["q_budget"]),
            gamma_noise_hat=float(data["gamma_noise_hat"]),
            sigma_calib=float(data["sigma_calib"]),
            n_calib=int(data.get("n_calib", 1000)),
            k_calib=int(data.get("k_calib", 0)),
            seed_calib=data.get("seed_calib"),
            partition=data.get("partition"),
        )


@dataclass(frozen=True)
class RawObservation:
    """Primary simulation measurements directly recorded from BB84 execution.

    Keeps raw simulation measurements strictly distinct from derived evaluations.

    Attributes:
        n_transmitted_pulses: Total pulses transmitted by Alice (N_pulse).
        n_sifted: Total sifted key bits where bases matched (N_sift).
        sifting_efficiency: Ratio of sifted bits to transmitted pulses (N_sift / N_pulse).
        n_test: Number of sifted bits dedicated to error estimation (N_test).
        n_key: Number of sifted bits reserved for secret key generation (N_key).
        error_count_test: Number of bit errors observed in the test partition.
        qber_test: Empirical sample QBER on test partition (error_count_test / n_test).
        error_count_z_test: Number of Z-basis bit errors in test partition.
        sifted_count_z_test: Number of Z-basis sifted bits in test partition.
        qber_z_test: Empirical Z-basis QBER on test partition.
        error_count_x_test: Number of X-basis bit errors in test partition.
        sifted_count_x_test: Number of X-basis sifted bits in test partition.
        qber_x_test: Empirical X-basis QBER on test partition.
        overall_sifted_qber: Full sifted key error rate across all N_sift bits.
        eve_info_gain: Eve's mutual information I(A; E) on sifted key.
        protocol_abort: Boolean indicating whether Alice and Bob aborted (qber_test > q_budget).
    """

    n_transmitted_pulses: int
    n_sifted: int
    sifting_efficiency: float
    n_test: int
    n_key: int
    error_count_test: int
    qber_test: float
    error_count_z_test: int
    sifted_count_z_test: int
    qber_z_test: float
    error_count_x_test: int
    sifted_count_x_test: int
    qber_x_test: float
    overall_sifted_qber: float
    eve_info_gain: float
    protocol_abort: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert raw observation to dictionary."""
        return {
            "n_transmitted_pulses": self.n_transmitted_pulses,
            "n_sifted": self.n_sifted,
            "sifting_efficiency": self.sifting_efficiency,
            "n_test": self.n_test,
            "n_key": self.n_key,
            "error_count_test": self.error_count_test,
            "qber_test": self.qber_test,
            "error_count_z_test": self.error_count_z_test,
            "sifted_count_z_test": self.sifted_count_z_test,
            "qber_z_test": self.qber_z_test,
            "error_count_x_test": self.error_count_x_test,
            "sifted_count_x_test": self.sifted_count_x_test,
            "qber_x_test": self.qber_x_test,
            "overall_sifted_qber": self.overall_sifted_qber,
            "eve_info_gain": self.eve_info_gain,
            "protocol_abort": self.protocol_abort,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawObservation":
        """Reconstruct RawObservation from dictionary."""
        return cls(
            n_transmitted_pulses=int(data["n_transmitted_pulses"]),
            n_sifted=int(data["n_sifted"]),
            sifting_efficiency=float(data["sifting_efficiency"]),
            n_test=int(data["n_test"]),
            n_key=int(data["n_key"]),
            error_count_test=int(data["error_count_test"]),
            qber_test=float(data["qber_test"]),
            error_count_z_test=int(data["error_count_z_test"]),
            sifted_count_z_test=int(data["sifted_count_z_test"]),
            qber_z_test=float(data["qber_z_test"]),
            error_count_x_test=int(data["error_count_x_test"]),
            sifted_count_x_test=int(data["sifted_count_x_test"]),
            qber_x_test=float(data["qber_x_test"]),
            overall_sifted_qber=float(data["overall_sifted_qber"]),
            eve_info_gain=float(data["eve_info_gain"]),
            protocol_abort=bool(data["protocol_abort"]),
        )


@dataclass(frozen=True)
class DerivedEvaluation:
    """Statistically derived evaluation metrics and feasibility classifications.

    Attributes:
        wilson_q_center: Center point of the two-sided Wilson score interval.
        wilson_q_lcb: Lower confidence bound (Q_LCB) at 95% confidence.
        wilson_q_ucb: Upper confidence bound (Q_UCB) at 95% confidence.
        wilson_margin: Half-width / margin of the Wilson score interval.
        confidence_level: Nominal confidence level (e.g. 0.95).
        feasibility_category: Three-state feasibility category (FEASIBLE, INFEASIBLE, INDETERMINATE).
        margin_to_budget: Security margin Q_budget - Q_UCB (positive implies feasible).
        is_strictly_stealthy: True if Q_UCB <= Q_budget.
    """

    wilson_q_center: float
    wilson_q_lcb: float
    wilson_q_ucb: float
    wilson_margin: float
    confidence_level: float
    feasibility_category: StatisticalFeasibility
    margin_to_budget: float
    is_strictly_stealthy: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert derived evaluation to dictionary."""
        return {
            "wilson_q_center": self.wilson_q_center,
            "wilson_q_lcb": self.wilson_q_lcb,
            "wilson_q_ucb": self.wilson_q_ucb,
            "wilson_margin": self.wilson_margin,
            "confidence_level": self.confidence_level,
            "feasibility_category": self.feasibility_category.value,
            "margin_to_budget": self.margin_to_budget,
            "is_strictly_stealthy": self.is_strictly_stealthy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DerivedEvaluation":
        """Reconstruct DerivedEvaluation from dictionary."""
        return cls(
            wilson_q_center=float(data["wilson_q_center"]),
            wilson_q_lcb=float(data["wilson_q_lcb"]),
            wilson_q_ucb=float(data["wilson_q_ucb"]),
            wilson_margin=float(data["wilson_margin"]),
            confidence_level=float(data["confidence_level"]),
            feasibility_category=StatisticalFeasibility(data["feasibility_category"]),
            margin_to_budget=float(data["margin_to_budget"]),
            is_strictly_stealthy=bool(data["is_strictly_stealthy"]),
        )


@dataclass(frozen=True)
class SimulationMetadata:
    """Provenance and runtime metadata for a single simulation run.

    Attributes:
        seed_simulator: Random seed passed to Qiskit Aer backend.
        seed_alice: Seed used for Alice's state preparation RNG.
        seed_bob: Seed used for Bob's measurement basis RNG.
        generator_version: Dataset generation pipeline version.
        timestamp_utc: ISO 8601 UTC timestamp of execution.
        execution_time_seconds: Elapsed simulation time in seconds.
    """

    seed_simulator: int
    seed_alice: int
    seed_bob: int
    generator_version: str
    timestamp_utc: str
    execution_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "seed_simulator": self.seed_simulator,
            "seed_alice": self.seed_alice,
            "seed_bob": self.seed_bob,
            "generator_version": self.generator_version,
            "timestamp_utc": self.timestamp_utc,
            "execution_time_seconds": self.execution_time_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationMetadata":
        """Reconstruct SimulationMetadata from dictionary."""
        return cls(
            seed_simulator=int(data["seed_simulator"]),
            seed_alice=int(data["seed_alice"]),
            seed_bob=int(data["seed_bob"]),
            generator_version=data["generator_version"],
            timestamp_utc=data["timestamp_utc"],
            execution_time_seconds=float(data["execution_time_seconds"]),
        )


@dataclass(frozen=True)
class DatasetRecord:
    """Comprehensive single-simulation record.

    Encapsulates unique identifiers, environmental conditions, policy specification,
    raw measurements, derived statistical evaluations, and provenance metadata.

    Attributes:
        sample_id: Unique record ID (e.g., 'M3-C01-ATT00-S101').
        condition_id: ID of the environmental condition tuple.
        attack_id: Canonical attack configuration ID ('ATT-00' through 'ATT-15').
        attack_family: Categorical family (Null, Unitary Probe, PIR).
        attack_parameters: Validated physical attack parameters.
        condition: Complete environmental condition specification.
        raw_observation: Directly observed simulation results.
        derived_evaluation: Statistical confidence bounds and feasibility category.
        metadata: Execution provenance and timing metadata.
    """

    sample_id: str
    condition_id: str
    attack_id: str
    attack_family: AttackFamily
    attack_parameters: AttackParameters
    condition: EnvironmentalCondition
    raw_observation: RawObservation
    derived_evaluation: DerivedEvaluation
    metadata: SimulationMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record to dictionary."""
        return {
            "sample_id": self.sample_id,
            "condition_id": self.condition_id,
            "attack_id": self.attack_id,
            "attack_family": self.attack_family.value,
            "attack_parameters": self.attack_parameters.to_dict(),
            "condition": self.condition.to_dict(),
            "raw_observation": self.raw_observation.to_dict(),
            "derived_evaluation": self.derived_evaluation.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRecord":
        """Deserialize record from dictionary."""
        return cls(
            sample_id=data["sample_id"],
            condition_id=data["condition_id"],
            attack_id=data["attack_id"],
            attack_family=AttackFamily(data["attack_family"]),
            attack_parameters=AttackParameters.from_dict(data["attack_parameters"]),
            condition=EnvironmentalCondition.from_dict(data["condition"]),
            raw_observation=RawObservation.from_dict(data["raw_observation"]),
            derived_evaluation=DerivedEvaluation.from_dict(data["derived_evaluation"]),
            metadata=SimulationMetadata.from_dict(data["metadata"]),
        )


@dataclass
class DatasetContainer:
    """Documented, versioned container holding a collection of simulation records.

    Attributes:
        schema_version: Dataset schema version string.
        dataset_type: Dataset classification ('pilot' or 'full').
        created_at_utc: Timestamp of dataset generation.
        total_records: Number of simulation records in container.
        total_conditions: Number of unique environmental conditions.
        total_attacks: Number of unique attack configurations.
        total_seeds: Number of independent simulation seeds per configuration.
        records: List of DatasetRecord instances.
        hardware_info: Hardware specifications and simulator backend details.
        summary_statistics: Aggregate feasibility and abort statistics.
    """

    schema_version: str
    dataset_type: str
    created_at_utc: str
    total_records: int
    total_conditions: int
    total_attacks: int
    total_seeds: int
    records: List[DatasetRecord] = field(default_factory=list)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)

    @property
    def dataset_version(self) -> str:
        """Alias for schema_version."""
        return self.schema_version

    @property
    def feasibility_summary(self) -> Dict[str, int]:
        """Convenience property for feasibility counts."""
        return self.summary_statistics.get("feasibility_counts", {})

    @property
    def protocol_aborts_count(self) -> int:
        """Convenience property for total protocol aborts."""
        return self.summary_statistics.get("empirical_abort_counts", {}).get("Aborted", 0)

    @property
    def family_breakdown(self) -> Dict[str, int]:
        """Convenience property for attack family counts."""
        return self.summary_statistics.get("family_breakdown", {})

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire dataset container to dictionary."""
        return {
            "schema_version": self.schema_version,
            "dataset_type": self.dataset_type,
            "created_at_utc": self.created_at_utc,
            "total_records": self.total_records,
            "total_conditions": self.total_conditions,
            "total_attacks": self.total_attacks,
            "total_seeds": self.total_seeds,
            "hardware_info": self.hardware_info,
            "summary_statistics": self.summary_statistics,
            "records": [rec.to_dict() for rec in self.records],
        }

    def save_json(self, filepath: str, indent: int = 2) -> None:
        """Save container to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    to_json = save_json

    @classmethod
    def load_json(cls, filepath: str) -> "DatasetContainer":
        """Load container from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [DatasetRecord.from_dict(r) for r in data.get("records", [])]
        return cls(
            schema_version=data["schema_version"],
            dataset_type=data["dataset_type"],
            created_at_utc=data["created_at_utc"],
            total_records=data["total_records"],
            total_conditions=data["total_conditions"],
            total_attacks=data["total_attacks"],
            total_seeds=data["total_seeds"],
            records=records,
            hardware_info=data.get("hardware_info", {}),
            summary_statistics=data.get("summary_statistics", {}),
        )

    from_json = load_json
