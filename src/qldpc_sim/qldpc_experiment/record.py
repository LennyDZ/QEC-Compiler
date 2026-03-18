from collections import defaultdict
from enum import Enum
from functools import cached_property
from typing import Dict, List, Set, Tuple
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field

from ..data_structure import TannerNode, LogicalOperator


class MeasurementOutcomes(BaseModel):
    """Class to represent the measurement outcomes within a compiler, map to the corresponding measured nodes."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4, init=False)
    tag: str = Field(
        default="",
        description="Human readable tag to identify the measurement outcomes.",
    )
    size: int = Field(
        default=0, description="Number of measurement outcomes included in this set."
    )
    measured_nodes: List[TannerNode] = Field(
        default_factory=list,
        description="List of nodes that are measured in this compiler, ordered as they appear in the measurement sequence. A node can appear multiple times if it is measured multiple times.",
    )


class EventType(Enum):
    """Enum to represent the type of event in a qLDPC experiment."""

    STAB_MEASUREMENT = "stab_measurement"
    OBSERVABLE = "observable"
    FRAME_CORRECTION = "frame_correction"


class OutcomeSet(BaseModel):
    """Class to refer a set of measurement outcome to be a specific event of interest in a qLDPC experiment."""

    model_config = ConfigDict(frozen=True)
    id: UUID = Field(default_factory=uuid4, init=False)
    tag: str
    type: EventType
    size: int
    measured_nodes: Set[TannerNode] = Field(default_factory=set)
    target: Set[LogicalOperator] | LogicalOperator | None = Field(
        default=None,
        description="The logical operator(s) associated with the event. It is used for frame correction events to specify the logical operator to which the correction applies. Or in observable if it correspond to the measurement of a logical observable.",
    )

    def __hash__(self) -> int:
        # Keep hashing stable and independent of mutable container fields.
        return hash(self.id)


class MeasurementRecord(BaseModel):
    """Class to store the results of a qLDPC experiment and efficiently retrieve the index of measurement outcomes related to a specific event.

    Attributes:
        events (List[OutcomeSet]): List of events that exist as subsets of the recorded measurement outcomes.
        num_measurement_recorded (int): Total number of measurement outcomes recorded.
    """

    events: Dict[OutcomeSet, int] = Field(
        default_factory=dict,
        description="Dictionary mapping each event to the index of the last measurement that exists (including its own measurements) at the time the event is added to the record.",
    )
    num_measurement_recorded: int = 0
    measurements: Dict[TannerNode, List[int]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Dictionary mapping each measured node to a list of its measurement outcomes index.",
    )

    def add_measurements(self, measurements: MeasurementOutcomes) -> None:
        """Add measurement outcomes to the record.

        Args:
            measurements (MeasurementOutcomes): The measurement outcomes to add.
        """
        index_offset = self.num_measurement_recorded
        for node in measurements.measured_nodes:
            self.measurements[node].append(index_offset)
            index_offset += 1
        self.num_measurement_recorded += len(measurements.measured_nodes)

    def add_event(self, event: OutcomeSet) -> None:
        """Add an event to the record.

        Args:
            event (OutcomeSet): The event to add. The index of the last measurement that exists at the time the event is added is recorded in the events dictionary.
        """
        if event in self.events:
            raise ValueError(
                f"OutcomeSet '{event.tag}' already exists in the measurement record."
            )
        self.events[event] = self.num_measurement_recorded

    def get_event_idx(self, event: OutcomeSet) -> Set[int]:
        """Return measurement indices associated with a given event.

        For each measured node in the event, the most recent index before the event
        end is returned.
        """

        if event not in self.events:
            raise KeyError(
                f"OutcomeSet '{event.tag}' is not present in the measurement record."
            )

        end_idx = self.events[event]
        idxs: Set[int] = set()
        for node in event.measured_nodes:
            valid_indices = [
                idx for idx in self.measurements.get(node, []) if idx < end_idx
            ]
            if not valid_indices:
                raise ValueError(
                    f"Event: {event.tag} refer to a node that has never been measured yet."
                )
            idxs.add(valid_indices[-1])
        return idxs

    @cached_property
    def event_to_measurement_idx(self) -> Dict[OutcomeSet, Set[int]]:
        """Return a dictionary mapping each event to measurement indices for its measured nodes.

        For size-0 events (e.g. derived observables), each node is mapped to the most recent
        measurement index that happened before the event end index.
        """

        return {event: self.get_event_idx(event) for event in self.events}
