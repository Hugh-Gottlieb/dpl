import os
import json
import cv2

from dpl_common.pl_image import PL_Image
from dpl_common.helpers import analysed_folder_name, CameraInfo, GimbalInfo, GpsInfo, Image, Transition

class Acquisition:

    def __init__(self, name: str, root_dir: str):
        self.name = name
        self.root_dir = root_dir
        self.__read_pl_img()
        self.__read_metadata()

    def __read_pl_img(self):
        self.pl_img = PL_Image(self.name)
        if os.path.exists(os.path.join(self.root_dir, analysed_folder_name, self.name + ".tif")):
            self.pl_img.load(os.path.join(self.root_dir, analysed_folder_name))

    def __read_metadata(self):
        metadata_path = os.path.join(self.root_dir, self.name, self.name + "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        self.gps = GpsInfo(**metadata["gps"]) if "gps" in metadata else None
        self.gimbal = GimbalInfo(**metadata["gimbal"]) if "gimbal" in metadata else None
        self.camera = CameraInfo(**metadata["camera"]) if "camera" in metadata else None
        self.imgs = []
        for img in (metadata["images"] if "images" in metadata else []):
            # XXX - catch old metadata json with ms timestamps
            if img["time"] > 1e11:
                raise Exception("Image timestamp probably in milliseconds, please use dpl/scripts/update_metadata_ts.py (or call Gerold / Hugh) to fix this.")
            # XXX - hopefully we can delete this soon?
            self.imgs.append(Image(
                name=img["name"],
                pl_state=Image.string_to_pl_state(img["pl_state"]),
                time=(img["time"]),
                data=None
            ))
        self.transitions = []
        for transition in (metadata["transitions"] if "transitions" in metadata else []):
            self.transitions.append(Transition(
                direction=Transition.string_to_transition_direction(transition["direction"]),
                time=(transition["time"])
            ))

    def get_name(self):
        return self.name

    def get_pl_image(self):
        return self.pl_img

    def get_gps_info(self):
        return self.gps

    def get_camera_info(self):
        return self.camera

    def get_gimbal_info(self):
        return self.gimbal

    def get_imgs(self, load_data):
        if load_data:
            for img in self.imgs:
                if img.data is None:
                    path = os.path.join(self.root_dir, self.name, "tif", img.name + ".tif")
                    img.data = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        return self.imgs

    def clear_imgs(self):
        for img in self.imgs:
            img.data = None

    def get_transitions(self):
        return self.transitions