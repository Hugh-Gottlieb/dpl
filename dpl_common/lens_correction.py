import os
import numpy as np
from dataclasses import dataclass
import cv2

from dpl_common.helpers import Image

# NOTE: looks in the lens_calibration_files folder (child to this folder) for files looking like "LensParameters_x.npz"

class LensCorrection:

    @dataclass
    class LensParameters:
        cameraMatrix: np.ndarray
        distCoeffs: np.ndarray

    def __init__(self, default_lens: str=None):
        lens_calibration_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "lens_calibration_files")
        lens_calibration_files = [file for file in os.listdir(lens_calibration_folder) if (file.startswith("LensParameters") and file.endswith(".npz"))]
        self.lens_calibrations = {}
        for file in lens_calibration_files:
            path = os.path.join(lens_calibration_folder, file)
            params = np.load(path)
            name = file[15:-4] # Omit "LensParameters" and ".npz"
            self.lens_calibrations[name] = self.LensParameters(params["cameraMatrix"], params["distCoeffs"])
        self.set_default_lens(default_lens)

    def set_default_lens(self, default_lens: str):
        if default_lens is not None and default_lens not in self.lens_calibrations:
            raise Exception(f"Unknown default lens {default_lens}")
        self.default_lens = default_lens

    def get_lens_names(self) -> list[str]:
        return sorted(self.lens_calibrations.keys())

    def correct_image(self, image: Image, lens_name: str=None):
        if lens_name is None: lens_name = self.default_lens
        image.data = self.correct_img(image.data, lens_name)

    def correct_images(self, images: list[Image], lens_name: str=None):
        if lens_name is None: lens_name = self.default_lens
        for image in images:
            image.data = self.correct_img(image.data, lens_name)

    def correct_imgs(self, imgs: list[np.ndarray], lens_name: str=None) -> list[np.ndarray]:
        if lens_name is None: lens_name = self.default_lens
        return [self.correct_img(img, lens_name) for img in imgs]

    def correct_img(self, img: np.ndarray, lens_name: str=None) -> np.ndarray:
        if lens_name is None: lens_name = self.default_lens
        assert (lens_name is not None), f"Default lens used, but none provided"
        assert (lens_name in self.lens_calibrations), f"No len calibration with the name {lens_name}"
        calibration = self.lens_calibrations[lens_name]
        return cv2.undistort(img, calibration.cameraMatrix, calibration.distCoeffs)