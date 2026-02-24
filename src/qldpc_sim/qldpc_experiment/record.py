from enum import Enum
from typing import List
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field

from ..data_structure import TannerNode


class EventType(Enum):
    """Enum to represent the type of event in a qLDPC experiment."""

    STAB_MEASUREMENT = "stab_measurement"
    OBSERVABLE = "observable"


class EventTag(BaseModel):
    """Class to represent an event that produces measurement outcomes in a qLDPC experiment."""

    model_config = ConfigDict(frozen=True)
    id: UUID = Field(default_factory=uuid4, init=False)

    type: EventType
    tag: str
    size: int
    measured_nodes: List[TannerNode] = Field(default_factory=list)


class MeasurementRecord(BaseModel):
    """Class to store the results of a qLDPC experiment.

    Attributes:
        data (dict): A dictionary containing the results of the experiment.
    """

    events: List[EventTag] = Field(default_factory=list)

    def add_event(self, event: EventTag):
        """Add an event to the record.

        Args:
            event (EventTag): The event to add.
        """
        self.events.append(event)
