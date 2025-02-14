from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from cv2 import applyColorMap, COLORMAP_INFERNO

analysed_folder_name = "analysed"

def tif_to_jpeg(img: np.ndarray, clip_percentile: float = 2.5, coloured: bool = True) -> np.ndarray :
    img = img.astype(np.float32)
    preview = img - np.percentile(img, clip_percentile)
    preview = (preview / np.percentile(preview, 100 - clip_percentile))
    preview = (np.clip(preview, 0, 1) * (2**8 - 1)).astype(np.uint8)
    if coloured:
        return applyColorMap(preview, COLORMAP_INFERNO)
    return preview

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