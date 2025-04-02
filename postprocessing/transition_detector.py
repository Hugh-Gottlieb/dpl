from dpl_common.helpers import Image, Transition, tif_to_jpeg
from dpl_common.registration import Registration
from dpl_common.lens_correction import LensCorrection
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter

class TransitionDetector:
    def __init__(self):
        kernel_x = cv2.getGaussianKernel(1280, 1024/2)
        kernel_y = cv2.getGaussianKernel(1024, 1024/2)
        kernel = kernel_y * kernel_x.T
        kernel = kernel - kernel.min()
        kernel = kernel / kernel.max()
        kernel[kernel <= 0] = 1e-9 # Not PERFECTLY zero to prevent division problems
        self.gaussian_kernel = kernel

    def __normalise_signal(self, signal: np.ndarray) -> np.ndarray:
        return (signal - signal.min()) / (signal.max() - signal.min())

    def __get_switched_mask(self, diff_img):
        # XXX Config
        switch_percentile_low, switch_percentile_high = 85, 95
        # XXX Config
        switch_diff_img = diff_img * self.gaussian_kernel # Devalue outside edges
        switch_low, switch_high = np.nanpercentile(switch_diff_img, switch_percentile_low), np.nanpercentile(switch_diff_img, switch_percentile_high)
        switch_mask = np.logical_and(switch_diff_img >= switch_low, switch_diff_img <= switch_high).astype(np.uint8)
        switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,1)),iterations=2) # Remove small noise
        switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations=1) # Close gaps
        switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations=3) # Remove bigger noise
        switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,1)),iterations=3) # Remove more small noise
        switch_mask = switch_mask.astype(bool)
        return switch_mask

    def __get_background_mask(self, diff_img):
        # XXX Config
        background_percentile_low, background_percentile_high = 5, 15
        # XXX Config
        background_diff_img = diff_img / self.gaussian_kernel # Raise value of outside edges so not picked
        background_low, background_high = np.nanpercentile(background_diff_img, background_percentile_low), np.nanpercentile(background_diff_img, background_percentile_high)
        background_mask = np.logical_and(background_diff_img >= background_low, background_diff_img <= background_high).astype(np.uint8)
        background_mask = background_mask.astype(bool)
        return background_mask

    def __check_switched(self, diff_img):
        # XXX Config
        switch_percentile_low, switch_percentile_high = 85, 95
        # XXX Config
        switch_mask = self.__get_switched_mask(diff_img)
        remaining_area = (np.count_nonzero(switch_mask)/switch_mask.size)/((switch_percentile_high - switch_percentile_low)/100)
        return remaining_area > 0.35

    # XXXXXXXXXXXXX
    # TODO - detect switch direction
    # XXXXXXXXXXXXX
    def __get_diff_image(self, first_img, second_img):
        direction = Transition.Direction.LOW_TO_HIGH
        diff_img = (first_img - second_img) if direction == Transition.Direction.HIGH_TO_LOW else (second_img - first_img)
        return diff_img

    # TODO - cleanup variable passed in
    def detect_transitions(self, images: list[Image], lens_name: str, acq_name: str) -> list[Transition]:

        images = images[:15] + images[-15:] # + images[15:30] # TODO - kill!!!!

        # XXX Config params
        dark_offset = 0
        switch_pairs = 5
        # XXX Config

        # TODO - get (and config) as params
        self.registration = Registration(feature_limit=2500)
        self.lens_correction = LensCorrection()
        self.lens_correction.set_default_lens(lens_name)
        self.lens_correction.correct_images(images)
        completion_percent = f"{0} / {len(images)}"
        print(completion_percent) # TODO - omit completion

        # Check if switched, get signal / background masks
        # Use a number of equally spaced pairings, in case multiple switches
        middle_index = round(len(images)/2)
        if switch_pairs < 1 or switch_pairs > len(images):
            print(f"Invalid switch_pairs config: {switch_pairs} not in range [1, {len(images)}]")
            return # TODO - error properly
        switch_indexes = np.linspace(0, len(images)-1, switch_pairs+1).astype(int)
        middle_image = images[middle_index]
        completed_registrations = 0
        for i, index in enumerate(switch_indexes):
            if index != middle_index:
                self.registration.register_image(images[index], middle_image)
            completed_registrations += 1
            completion_percent = f"{completed_registrations} / {len(images)}"
            print(completion_percent) # TODO - omit completion
        diff_imgs = []
        for index in switch_indexes[1:]:
            diff_img = self.__get_diff_image(images[0].data, images[index].data)
            if self.__check_switched(diff_img):
                diff_imgs.append(diff_img)
        if not diff_imgs:
            print("No switches detected") # TODO - error properly
            return
        diff_img = np.mean(diff_imgs, axis=0)
        switch_mask = self.__get_switched_mask(diff_img)
        background_mask = self.__get_background_mask(diff_img)

        # Complete remaining registrations
        for i, image in enumerate(images):
            if i in switch_indexes or i == middle_index:
                continue
            self.registration.register_image(image, middle_image)
            completed_registrations += 1
            completion_percent = f"{completed_registrations} / {len(images)}"
            print(completion_percent) # TODO - omit completion

        # Get PL signal, correcting for changes in background illumination
        background_signal = np.array([np.nanmean(img.data[background_mask]) for img in images]) - dark_offset
        switched_signal = np.array([np.nanmean(img.data[switch_mask]) for img in images]) - dark_offset
        background_ratio = np.average(background_signal) / background_signal
        pl_signal = switched_signal * background_ratio

        # Detect transitions
        savgol_size_percent = 0.8
        while True:
            savgol_size = round(savgol_size_percent * len(pl_signal))
            d_savgol = savgol_filter(pl_signal, savgol_size, 1, 1)
            # TODO - detect switches

            worked = True # TODO - fix this!
            if worked:
                break
            savgol_size_percent *= 0.8

        # TODO: detect switching point / plateau area's based on 0-derivative?
        #   - Normalise (around 0) -> decide if 0 or plateau or unknown. Transition is 0 -> local min (if HIGH_TO_LOW) -> 0
        #   - Or use the straight lines somehow else, derivative very sensitive. Smoothed derivative (over more points than just 2, rather than convolve to smooth)
        # TODO: detect IF high-to-low or low-to-high (prev algo existed?)

        completion_percent = f"{len(images)} / {len(images)}"
        print(completion_percent) # TODO - omit completion

        # Plot
        # TODO - hide behind debug flag
        plt.figure(figsize=(19,9))
        plt.subplot(2,3,1)
        plt.title("Rough PL image")
        diff_img[np.isnan(diff_img)] = 0
        plt.imshow(tif_to_jpeg(diff_img,5,True)[:,:,::-1]) # Flip channels so displayed correctly
        plt.subplot(2,3,2)
        plt.title("Switched area")
        plt.imshow(switch_mask)
        plt.subplot(2,3,3)
        plt.title("Background area")
        plt.imshow(background_mask)
        plt.subplot(2,3,4)
        plt.title("Raw signals")
        ids = list(range(len(pl_signal)))
        all_signal = np.array([np.nanmean(img.data) for img in images]) - dark_offset
        plt.plot(ids, all_signal, label="all")
        plt.plot(ids, background_signal, label="background")
        plt.plot(ids, switched_signal, label="switched")
        plt.plot(ids, pl_signal, label="pl")
        plt.legend()
        plt.subplot(2,3,5)
        plt.title("Normalised signals")
        plt.plot(ids, self.__normalise_signal(all_signal), label="all")
        plt.plot(ids, self.__normalise_signal(background_signal), label="background")
        plt.plot(ids, self.__normalise_signal(switched_signal), label="switched")
        plt.plot(ids, self.__normalise_signal(pl_signal), label="pl", linewidth=1)
        savgol = savgol_filter(pl_signal, savgol_size, 1)
        plt.plot(ids, self.__normalise_signal(savgol), label=f"savgol", linewidth=3)
        plt.legend()
        plt.subplot(2,3,6)
        plt.title("Transition detection")
        plt.plot(ids, d_savgol, label=f"d_savgol", linewidth=1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{acq_name}_switch-debug.png") # TODO - save in debug folder of output dir
        plt.show()