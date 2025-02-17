import numpy as np
from dataclasses import asdict
from enum import Enum, auto
import time
import os
import json
from PIL import Image as PIL_Image

from dpl_common.helpers import Image, GpsInfo, GimbalInfo, CameraInfo
from dpl_common.config import Config

class PL_Image:

    class Status(Enum):
        NO = auto()
        YES = auto()
        STALE = auto() # Yes, but with old settings

    def __init__(self, name):
        self.name = name
        self.data = None

    def create(self, imgs: list[Image], config: Config, gps: GpsInfo, gimbal: GimbalInfo, camera: CameraInfo):
        self.data = self.__get_avg_img(imgs, Image.PL_State.HIGH) - self.__get_avg_img(imgs, Image.PL_State.LOW)
        nan_mask = np.any([np.isnan(img.data) for img in imgs if img.pl_state in [Image.PL_State.HIGH, Image.PL_State.LOW]], axis=0)
        self.data[np.logical_or(nan_mask, self.data < 0)] = 0
        self.process_config = config.get_config()
        self.process_time = time.time()
        self.acq_time = imgs[0].time # Safe as __get_avg_imgs will fail if there are no imgs
        self.gps = gps
        self.gimbal = gimbal
        self.camera = camera

    def __get_avg_img(self, imgs: list[Image], state: Image.PL_State) -> np.ndarray:
        relevant_imgs = [img.data for img in imgs if img.pl_state == state]
        assert (len(relevant_imgs) >= 1), f"PL image has no imgs of state {state.name}"
        return np.average(relevant_imgs, axis=0)

    def load(self, processed_folder: str):
        data_path, metadata_path = self.__get_paths(processed_folder)
        assert (os.path.exists(data_path) and os.path.exists(metadata_path)), f"PL Image files are missing: {data_path}, {metadata_path}"
        self.data = np.array(PIL_Image.open(data_path), dtype=float)
        with open(metadata_path) as f:
            metadata = json.load(f)
        self.process_config = metadata["process_config"]
        self.process_time = metadata["process_time"]
        self.acq_time = metadata["acq_time"]
        self.gps = GpsInfo(**metadata["gps"])
        self.gimbal = GimbalInfo(**metadata["gimbal"])
        self.camera = CameraInfo(**metadata["camera"])

    def __get_paths(self, processed_folder: str) -> tuple[str, str]:
        data_path = os.path.join(processed_folder, self.name + ".tif")
        metadata_path = os.path.join(processed_folder, self.name + ".json")
        return data_path, metadata_path

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
            "camera": asdict(self.camera),
        }
        with open(os.path.join(folder, self.name + ".json"), "w") as f:
            json.dump(metadata, f, indent=4)

    def clear(self, processed_folder: str):
        for path in self.__get_paths(processed_folder):
            if os.path.exists(path):
                os.remove(path)
        self.data = None

    def get_status(self, current_config: Config = None):
        if self.data is None:
            return PL_Image.Status.NO
        elif (current_config is not None) and (not current_config.same_hash(self.process_config)):
            return PL_Image.Status.STALE
        else:
            return PL_Image.Status.YES