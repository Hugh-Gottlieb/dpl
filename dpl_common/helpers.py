from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

@dataclass
class GpsInfo:
    time: float
    latitude: float
    longitude: float
    altitude: float
    homepoint_height: float
    fix: int
    number_of_visible_satellies: int

@dataclass
class GimbalInfo:
    pitch: float
    yaw: float
    mode: str

class Transition(Enum):
    UNKNOWN = auto()
    LOW_TO_HIGH = auto()
    HIGH_TO_LOW = auto()

class PL_State(Enum):
    HIGH = auto()
    LOW = auto()
    UNKNOWN = auto()

@dataclass
class Image:
    name: str
    time: float
    pl_state: PL_State
    data: np.ndarray