"""roomd semantics: furniture and seats from the fused map, tracked across frames.

Entry points: :func:`detect` (one frame -> candidates) and :class:`RoomSemantics`
(frames -> RoomModel v2 objects/seats fragment); :class:`SemanticsSink` /
:func:`create_sink` plug both into roomd on top of the fusion sink. See README.md.
"""
from .classify import Candidate, Detection, detect
from .mapview import HeightMap, MapView
from .params import SemanticsParams
from .seats import generate_seats
from .sink import SemanticsSink, create_sink
from .synthetic import SyntheticRoom
from .tracker import RoomSemantics
from .basics import (
    Kind, Obb, ObjectState, SceneLabel, SeatState, Source, quat_yaw, yaw_diff, yaw_quat)

__all__ = [
    "Candidate", "Detection", "detect", "HeightMap", "MapView", "SemanticsParams",
    "generate_seats", "SyntheticRoom", "RoomSemantics", "Kind", "Obb", "ObjectState",
    "SceneLabel", "SeatState", "Source", "quat_yaw", "yaw_diff", "yaw_quat",
    "SemanticsSink", "create_sink",
]
