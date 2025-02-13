import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum, auto
import time
import os
import json
from PIL import Image as PIL_Image

from dpl_common.helpers import Image, PL_State, GpsInfo, GimbalInfo
from dpl_common.config import Config

@dataclass
class PL_Image:

    class Status(Enum):
        NO = auto()
        YES = auto()
        STALE = auto() # Processed, but with old settings
        IN_PROGRESS = auto()
        QUEUED = auto()
        ERROR = auto()

    def __init__(self, name):
        self.clear_status()
        self.name = name
        self.data = None

    def create(self, imgs: list[Image], config: Config, gps: GpsInfo, gimbal: GimbalInfo):
        self.data = self.__get_avg_img(imgs, PL_State.HIGH) - self.__get_avg_img(imgs, PL_State.LOW)
        self.process_config = config.get_config()
        self.process_time = time.time()
        self.acq_time = imgs[0].time # Safe as __get_avg_imgs will fail if there are no imgs
        self.gps = gps
        self.gimbal = gimbal

    def __get_avg_img(imgs: list[Image], state: PL_State) -> np.ndarray:
        relevant_imgs = [img for img in imgs if img.pl_state == state]
        assert (len(relevant_imgs) >= 1), f"PL image has no imgs of state {state.name}"
        return np.average(relevant_imgs, axis=0)

    def load(self, processed_folder: str):
        data_path = os.path.join(processed_folder, self.name + ".tif")
        metadata_path = os.path.join(processed_folder, self.name + ".json")
        assert (os.path.exists(data_path) and os.path.exists(metadata_path)), f"PL Image files are missing: {data_path}, {metadata_path}"
        self.data = np.array(PIL_Image.open(data_path), dtype=float)
        with open(metadata_path) as f:
            metadata = json.load(f)
        self.process_config = metadata["process_config"]
        self.process_time = metadata["process_time"]
        self.acq_time = metadata["acq_time"]
        self.gps = GpsInfo(**metadata["gps"])
        self.gimbal = GimbalInfo(**metadata["gimbal"])

    def save(self, folder: str):
        assert (self.data is not None), "Cannot save empty PL Image"
        pil_image = PIL_Image.fromarray(self.data)
        pil_image.save(os.path.join(folder, self.name + ".tif"))
        metadata = {
            "process_config": self.process_config,
            "process_time": self.process_time,
            "acq_time": self.acq_time,
            "gps": asdict(self.gps),
            "gimbal": asdict(self.gimbal),
        }
        with open(os.path.join(folder, self.name + ".json"), "w") as f:
            json.dump(metadata, f, indent=4)

    def set_status(self, status: Status):
        self.temp_status = status

    def clear_status(self):
        self.temp_status = None

    def get_status(self, current_config: Config):
        if self.temp_status is not None:
            return self.temp_status
        elif self.data is None:
            return PL_Image.Status.NO
        elif not current_config.same_hash(self.process_config):
            return PL_Image.Status.STALE
        else:
            return PL_Image.Status.YES