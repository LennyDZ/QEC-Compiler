from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator, model_validator

from qldpc_sim.qldpc_experiment.record import MeasurementOutcomes

from ..data_structure import LogicalOperator, LogicalQubit, PauliChar, TannerNode


class FrameCorrection(BaseModel):
    """Correction to apply to a logical operator after measuring it.

    Attributes:
        target (LogicalOperator): The logical operator to which the correction applies.
        correction_condition (Set[int]): The set of measurement outcomes involved in the correction condition. The correction is applied if the parity of the outcomes in the set is -1.
    """

    target: LogicalQubit
    correction_X_cond: Set[int] = Field(default_factory=set)
    correction_Z_cond: Set[int] = Field(default_factory=set)

    def __add__(self, other: "FrameCorrection") -> "FrameCorrection":
        return FrameCorrection(
            target=self.target,
            correction_X_cond=self.correction_X_cond ^ other.correction_X_cond,
            correction_Z_cond=self.correction_Z_cond ^ other.correction_Z_cond,
        )


class FrameUpdate(BaseModel):
    """Defines how to update the Pauli frame of a set of target nodes after applying a logical gadget.

    Attributes:
        target_size (int): The number of qubits the frame update applies to.
        transformation_matrix (np.ndarray): The matrix defining the frame update.

    """

    model_config = ConfigDict(frozen=True)
    target_size: int = Field(
        default=1, description="The number of qubits the frame update applies to."
    )
    transformation_matrix: List[List[int]] = Field(
        default_factory=lambda: [[1, 0], [0, 1]],
        description="The matrix defining the frame update. It should be a 2*target_size by 2*target_size binary matrix, where the first target_size rows define the update of the X correction conditions and the last target_size rows define the update of the Z correction conditions. The first target_size columns correspond to the X correction conditions of the input frame corrections, and the last target_size columns correspond to the Z correction conditions of the input frame corrections.",
    )

    @model_validator(mode="after")
    def check_transformation_matrix(cls, frame_update: "FrameUpdate") -> "FrameUpdate":
        if frame_update.transformation_matrix.shape != (
            2 * frame_update.target_size,
            2 * frame_update.target_size,
        ):
            raise ValueError(
                f"Transformation matrix must be of shape {(2 * frame_update.target_size, 2 * frame_update.target_size)}"
            )
        return frame_update

    def apply(self, target: List[FrameCorrection]) -> List[FrameCorrection]:
        if len(target) != self.target_size:
            raise ValueError(
                f"Target size must equal {self.target_size}, got {len(target)}"
            )

        updated_frame_corrections = []

        for i in range(self.target_size):
            x_cond: Set[int] = set()
            z_cond: Set[int] = set()

            for j in range(self.target_size):
                # X output bit
                if self.transformation_matrix[2 * i, 2 * j]:
                    x_cond ^= target[j].correction_X_cond
                if self.transformation_matrix[2 * i, 2 * j + 1]:
                    x_cond ^= target[j].correction_Z_cond

                # Z output bit
                if self.transformation_matrix[2 * i + 1, 2 * j]:
                    z_cond ^= target[j].correction_X_cond
                if self.transformation_matrix[2 * i + 1, 2 * j + 1]:
                    z_cond ^= target[j].correction_Z_cond

            updated_frame_corrections.append(
                FrameCorrection(
                    target=target[i].target,
                    correction_X_cond=x_cond,
                    correction_Z_cond=z_cond,
                )
            )

        return updated_frame_corrections


class IdentityFrameUpdate(FrameUpdate):
    transformation_matrix: List[List[int]] = []

    def apply(self, target: List[FrameCorrection]) -> List[FrameCorrection]:
        return target


# class HadamardFrameUpdate(FrameUpdate):
#     def apply(self, target: List[FrameCorrection]) -> List[FrameCorrection]:
#         return [
#             FrameCorrection(
#                 target=target[i].target,
#                 correction_X_cond=target[i].correction_Z_cond,
#                 correction_Z_cond=target[i].correction_X_cond,
#             )
#             for i in range(self.target_size)
#         ]


# class SFrameUpdate(FrameUpdate):
#     target_size: int = 1
#     transformation_matrix: np.ndarray = np.array([[1, 0], [1, 1]])


# class CNOTFrameUpdate(FrameUpdate):
#     target_size: int = 2
#     transformation_matrix: np.ndarray = np.array(
#         [
#             [1, 0, 0, 0],
#             [1, 1, 0, 0],
#             [0, 0, 1, 1],
#             [0, 0, 0, 1],
#         ]
#     )


class FrameState(BaseModel):
    """Class to track the current Pauli frame of logical qubits in a qLDPC simulation."""

    qubits: Set[LogicalQubit] = Field(default_factory=set)
    frame_corrections: Dict[LogicalQubit, FrameCorrection] = Field(
        init=False, default_factory=dict
    )

    def _ensure_correction(self, target: LogicalQubit) -> None:
        if target not in self.frame_corrections:
            self.frame_corrections[target] = FrameCorrection(target=target)

    def get_correction(self, target: LogicalQubit, type: PauliChar) -> Set[int]:
        self._ensure_correction(target)
        if type == PauliChar.X:
            return self.frame_corrections[target].correction_X_cond
        elif type == PauliChar.Z:
            return self.frame_corrections[target].correction_Z_cond
        else:
            raise ValueError(f"Unknown logical operator type: {type}")

    @model_validator(mode="after")
    def initialize_frame_corrections(cls, frame_state: "FrameState") -> "FrameState":
        for qubit in frame_state.qubits:
            frame_state.frame_corrections[qubit] = FrameCorrection(target=qubit)
        return frame_state

    def add_correction(
        self, target: LogicalQubit, type: PauliChar, correction_condition: Set[int]
    ) -> None:
        """Add a frame correction to the tracker.

        Args:
            target (LogicalQubit): The target logical qubit.
            type (PauliChar): The type of Pauli correction (X or Z).
            correction_condition (Set[int]): The set of conditions for the correction.
        """
        self._ensure_correction(target)
        if type == PauliChar.X:
            self.frame_corrections[target].correction_X_cond ^= correction_condition
        elif type == PauliChar.Z:
            self.frame_corrections[target].correction_Z_cond ^= correction_condition
        else:
            raise ValueError(f"Unknown logical operator type: {type}")

    def update_frame(self, targets: List[LogicalQubit], update: FrameUpdate):
        """Update the frame corrections for a list of target nodes using a FrameUpdate.

        Args:
            targets (List[LogicalQubit]): The list of target nodes to update.
            update (FrameUpdate): The FrameUpdate containing the update function.
        """
        for target in targets:
            self._ensure_correction(target)
        target_corrections = [self.frame_corrections[target] for target in targets]
        updated_corrections = update.apply(target_corrections)
        for correction in updated_corrections:
            self.frame_corrections[correction.target] = correction
