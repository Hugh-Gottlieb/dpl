import matplotlib.patches
from dpl_common.helpers import Image, Transition, tif_to_jpeg
from dpl_common.registration import Registration
from dpl_common.lens_correction import LensCorrection
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter
from itertools import groupby

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
        outlier_percentile, signal_percentile = 5, 10
        # XXX Config
        switch_percentile_low, switch_percentile_high = 100 - outlier_percentile - signal_percentile, 100 - outlier_percentile
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
        outlier_percentile, signal_percentile = 5, 10
        # XXX Config
        background_percentile_low, background_percentile_high = outlier_percentile, outlier_percentile + signal_percentile
        background_diff_img = diff_img / self.gaussian_kernel # Raise value of outside edges so not picked
        background_low, background_high = np.nanpercentile(background_diff_img, background_percentile_low), np.nanpercentile(background_diff_img, background_percentile_high)
        background_mask = np.logical_and(background_diff_img >= background_low, background_diff_img <= background_high).astype(np.uint8)
        background_mask = background_mask.astype(bool)
        return background_mask

    def __check_switched(self, diff_img):
        # XXX Config
        outlier_percentile, signal_percentile = 5, 10
        # XXX Config
        switch_mask = self.__get_switched_mask(diff_img)
        remaining_area = (np.count_nonzero(switch_mask)/switch_mask.size)/(signal_percentile/100)
        return remaining_area > 0.5

    def __get_switch_direction(self, first_img:np.ndarray, second_img:np.ndarray) -> Transition.Direction:
        # XXX Config
        outlier_percentile, signal_percentile = 5, 10
        # XXX Config
        # Clip extremes, to remove outliers
        diff_img = first_img - second_img
        diff_img = np.clip(diff_img, np.nanpercentile(diff_img, outlier_percentile), np.nanpercentile(diff_img, 100 - outlier_percentile))
        # Blur on HUGE scale, to cancel out ground noise
        # - cant median blur as float32 max kernel size is 5
        # - cv2.blur turns the whole things to nan if any present, so temporarily disable (restore for valid percentiles)
        nan_mask = np.isnan(diff_img)
        diff_img[nan_mask] = 0
        diff_img = cv2.blur(diff_img, (51,51))
        diff_img[nan_mask] = np.nan
        # Treat lowest changing as noise, try to shift average to np.nan
        abs_img = np.abs(diff_img)
        noise_cutoff = np.nanpercentile(abs_img, outlier_percentile + signal_percentile)
        noise = np.nanmean(diff_img[np.logical_and(diff_img >= -noise_cutoff, diff_img <= noise_cutoff)])
        diff_img = diff_img - noise
        # Devalue outskirts of images, focus on panels at center
        diff_img = diff_img * self.gaussian_kernel
        # Calculate square sum (accounts for size and strength of signal) to find which one is clearer signal
        positive_cutoff = np.nanpercentile(diff_img, 100 - outlier_percentile - signal_percentile)
        negative_cutoff = np.nanpercentile(diff_img, outlier_percentile + signal_percentile)
        positive_sum = np.sum(np.square(diff_img[diff_img >= positive_cutoff]))
        negative_sum = np.sum(np.square(diff_img[diff_img <= negative_cutoff]))
        direction = Transition.Direction.HIGH_TO_LOW if positive_sum > negative_sum else Transition.Direction.LOW_TO_HIGH
        return direction

    # XXXXXXXXXXXXX
    # TODO - detect switch direction
    # XXXXXXXXXXXXX
    def __get_diff_image(self, first_img:np.ndarray, second_img:np.ndarray) -> tuple[np.ndarray, Transition.Direction]:
        direction = self.__get_switch_direction(first_img, second_img)
        diff_img = (first_img - second_img) if direction == Transition.Direction.HIGH_TO_LOW else (second_img - first_img)
        return diff_img, direction

    # TODO - cleanup variable passed in
    def detect_transitions(self, images: list[Image], lens_name: str, acq_name: str) -> list[Transition]:

        # images = images[:15] + images[-15:] # + images[15:30] # TODO - kill!!!!

        # XXX Config params
        dark_offset = 0
        max_transitions = 5
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
        if max_transitions < 1 or max_transitions > len(images):
            raise Exception(f"Invalid switch_pairs config: {max_transitions} not in range [1, {len(images)}]")
        switch_indexes = np.linspace(0, len(images)-1, max_transitions+1).astype(int)
        middle_image = images[middle_index]
        completed_registrations = 0
        for i, index in enumerate(switch_indexes):
            if index != middle_index:
                self.registration.register_image(images[index], middle_image)
            completed_registrations += 1
            completion_percent = f"{completed_registrations} / {len(images)}"
            print(completion_percent) # TODO - omit completion
        diff_imgs = []
        diff_img_directions = []
        for index in switch_indexes[1:]:
            diff_img, direction = self.__get_diff_image(images[0].data, images[index].data)
            if self.__check_switched(diff_img):
                diff_imgs.append(diff_img)
                diff_img_directions.append(direction)
        expected_transitions = [key for key, _ in groupby(diff_img_directions)]
        print(expected_transitions) # TODO - remove
        if not diff_imgs:
            raise Exception("No switches detected")
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
        savgol_size_percent = 0.25
        while True:
            savgol_size_percent *= 0.8
            savgol_size = round(savgol_size_percent * len(pl_signal))
            dsavgol = savgol_filter(pl_signal, savgol_size, 1, 1)
            norm_dsavgol = dsavgol / np.abs(dsavgol).max()
            abs_norm_dsavgol = np.abs(norm_dsavgol)
            if abs_norm_dsavgol.min() > 0.1: # If never got down to 0, then too savgol coarse
                continue
            index = 0
            plateaus, transitions = [], [] # start, stop, direction
            for is_plateau, vals in groupby(abs_norm_dsavgol < 0.2):
                start = index
                index += len(list(vals))
                stop = index
                if is_plateau:
                    if not transitions:
                        plateau_state = Image.PL_State.UNKNOWN
                    else:
                        plateau_state = Image.PL_State.HIGH if transitions[-1][-1] == Transition.Direction.LOW_TO_HIGH else Image.PL_State.LOW
                    plateaus.append([start, stop, plateau_state])
                else:
                    positive_gradient = np.average(norm_dsavgol[start:stop]) > 0
                    transition_dir = Transition.Direction.LOW_TO_HIGH if positive_gradient else Transition.Direction.HIGH_TO_LOW
                    if transitions:
                        # TODO - this is biting me: what if there are 2 humps but one is super small and just noise?
                        #      - ignore very small transitions?
                        #      - or dont trust the last plateau? But need to still have 2 plateaus!
                        #      - so actually asign all the pl states later, and merge adjacent?
                        # assert(transition_dir != transitions[-1][-1]), "Transitions must go in alternating directions"
                        pass
                    elif plateaus: # Tag state of previous plateau
                        plateaus[-1][-1] = Image.PL_State.LOW if transition_dir == Transition.Direction.LOW_TO_HIGH else Image.PL_State.HIGH
                    transitions.append([start, stop, transition_dir])
            if len(plateaus) < 2: # If dominated by transitions, and cant get 2 plateaus, try more sensitive
                continue
            break
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
        for start, stop, pl_state in plateaus:
            colour = "#5E99DB44" if pl_state == Image.PL_State.LOW else "#B7B22944"
            plt.gca().add_patch(matplotlib.patches.Rectangle((start,0),stop-start-1,1,color=colour))
        plt.legend()
        plt.subplot(2,3,6)
        plt.title("Transition detection")
        plt.plot(ids, norm_dsavgol, label=f"d_savgol", linewidth=1)
        for start, stop, pl_state in plateaus:
            colour = "#5E99DB44" if pl_state == Image.PL_State.LOW else "#B7B22944"
            plt.gca().add_patch(matplotlib.patches.Rectangle((start,-1),stop-start-1,1,color=colour))
        plt.ylim(-1.1,1.1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{acq_name}_switch-debug.png") # TODO - save in debug folder of output dir
        plt.show() # TODO - remove