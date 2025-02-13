from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

analysed_folder_name = "analysed"

@dataclass
class GpsInfo:
    date: str
    time: str
    latitude: float
    longitude: float
    altitude: float
    homepoint_height: float
    fix: int
    number_of_satellites_visible: int

@dataclass
class GimbalInfo:
    pitch: float
    yaw: float
    mode: str

@dataclass
class CameraInfo:
    exposure_time: int
    fps: int

@dataclass
class Transition:
    class Direction(Enum):
        UNKNOWN = auto()
        LOW_TO_HIGH = auto()
        HIGH_TO_LOW = auto()
    @staticmethod
    def string_to_transition_direction(value: str) -> Direction:
        return Transition.Direction[value.upper().replace("-", "_")]
    direction: Direction
    time: float # seconds since epoch (same as time.time())

@dataclass
class Image:
    class PL_State(Enum):
        HIGH = auto()
        LOW = auto()
        UNKNOWN = auto()
    @staticmethod
    def string_to_pl_state(value: str) -> PL_State:
        return Image.PL_State[value.upper()]
    name: str
    time: float # seconds since epoch (same as time.time())
    pl_state: PL_State
    data: np.ndarray