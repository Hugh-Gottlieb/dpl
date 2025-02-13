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

    def __init__(self):
        lens_calibration_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "lens_calibration_files")
        lens_calibration_files = [file for file in os.listdir(lens_calibration_folder) if (file.startswith("LensParameters") and file.endswith(".npz"))]
        self.lens_calibrations = {}
        for file in lens_calibration_files:
            path = os.path.join(lens_calibration_folder, file)
            params = np.load(path)
            name = file[15:-4] # Omit "LensParameters" and ".npz"
            self.lens_calibrations[name] = self.LensParameters(params["cameraMatrix"], params["distCoeffs"])

    def get_lens_names(self) -> list[str]:
        return sorted(self.lens_calibrations.keys())

    def correct_image(self, image: Image, lens_name: str):
        image.data = self.correct_img(image.data, lens_name)

    def correct_images(self, images: list[Image], lens_name: str):
        for image in images:
            image.data = self.correct_img(image.data, lens_name)

    def correct_imgs(self, imgs: list[np.ndarray], lens_name: str) -> list[np.ndarray]:
        return [self.correct_img(img, lens_name) for img in imgs]

    def correct_img(self, img: np.ndarray, lens_name: str) -> np.ndarray:
        assert (lens_name in self.lens_calibrations), f"No len calibration with the name {lens_name}"
        calibration = self.lens_calibrations[lens_name]
        return cv2.undistort(img, calibration.cameraMatrix, calibration.distCoeffs)