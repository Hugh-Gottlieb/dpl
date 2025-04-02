import matplotlib.patches
from dpl_common.helpers import Image, Transition, tif_to_jpeg
from dpl_common.registration import Registration
from dpl_common.lens_correction import LensCorrection
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter, argrelextrema
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

    # TODO - for the GAF tests, maybe this should be more sensitive? Depends on zoom level probably? Misses some
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

    def __get_diff_image(self, first_img:np.ndarray, second_img:np.ndarray) -> tuple[np.ndarray, Transition.Direction]:
        direction = self.__get_switch_direction(first_img, second_img)
        diff_img = (first_img - second_img) if direction == Transition.Direction.HIGH_TO_LOW else (second_img - first_img)
        return diff_img, direction

    # TODO - cleanup variable passed in
    def detect_transitions(self, images: list[Image], lens_name: str, acq_name: str) -> list[Transition]:
        # XXX Config params
        dark_offset = 0
        max_transitions = 4
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
            elif diff_img_directions: # If had switched before and no longer switched, then switched back
                diff_img_directions.append(Transition.Direction.HIGH_TO_LOW if diff_img_directions[0] == Transition.Direction.LOW_TO_HIGH else Transition.Direction.LOW_TO_HIGH)
        expected_transitions = [key for key, _ in groupby(diff_img_directions)]
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
        # XXX - fair warning, this gets a bit finnicky!!!
        # NOTE - if the first image is during a transition, things could get weird when considering the expected transitions
        # NOTE - if expected BIG high-to-low but get small high-to-low, small low-to-high, THEN the big high-to-low, we have a problem ...
        savgol_size_percent = 0.25 # TODO - change with number of transitions detected?
        expected_high_to_low = np.sum([1 for dir in expected_transitions if dir == Transition.Direction.HIGH_TO_LOW])
        expected_low_to_high = len(expected_transitions) - expected_high_to_low
        success = False
        for _ in range(10):
            # Calculate savgol
            savgol_size_percent *= 0.8 # NOTE - constant: try more sensitive savgol
            savgol_size = round(savgol_size_percent * len(pl_signal))
            dsavgol = savgol_filter(pl_signal, savgol_size, 1, 1) # Fit 1st-order polynomials, 1 derivative
            norm_dsavgol = dsavgol / np.abs(dsavgol).max()
            abs_norm_dsavgol = np.abs(norm_dsavgol)
            if abs_norm_dsavgol.min() > 0.1: # NOTE - constant: if never got down to 0, then too savgol coarse
                print(f"{acq_name} savgol: no flat regions detected, reducing window size")
                continue
            # Get all transition options
            high_to_low_options = [(index, Transition.Direction.HIGH_TO_LOW) for index in argrelextrema(norm_dsavgol, np.less, mode="clip")[0] if norm_dsavgol[index] < -0.5]
            low_to_high_options = [(index, Transition.Direction.LOW_TO_HIGH) for index in argrelextrema(norm_dsavgol, np.greater, mode="clip")[0] if norm_dsavgol[index] > 0.5]
            if len(high_to_low_options) < expected_high_to_low or len(low_to_high_options) < expected_low_to_high: # If dont get enough transitions, try more sensitive
                print(f"{acq_name} savgol: insufficient transitions detected, reducing window size")
                continue
            transition_options = sorted(high_to_low_options + low_to_high_options, key=lambda x:x[0])
            # Match transitions to expectations
            transitions = []
            error = False
            for expected_transition_dir in expected_transitions:
                option_indices = []
                while transition_options:
                    option_index, option_dir = transition_options[0]
                    if option_dir != expected_transition_dir:
                        break
                    option_indices.append(option_index)
                    transition_options.pop(0)
                print(expected_transition_dir, option_indices)
                if not option_indices: # Error: get to more sensitive savgol
                    error = True
                    print(f"{acq_name} savgol: cannot match expected transitions, reducing window size")
                    break
                option_vals = [abs_norm_dsavgol[index] for index in option_indices]
                best_option_index = option_indices[np.argmax(option_vals)]
                transitions.append((best_option_index, expected_transition_dir))
            if error: # Error occured, try more sensitive
                continue
            # NOTE/TODO - this occuring is my nightmare ... most urgent to fix for GAF!
            if transition_options:
                print(f"{acq_name} savgol: fulfilled all expected transitions, without considering all options ... try increasing starting savgol size?")
                print(transitions, transition_options)
                # continue # TODO - bring this back?
            # Get window around transitions
            index = 0
            transition_windows = [] # start, stop, transtion.direction
            for is_plateau, vals in groupby(abs_norm_dsavgol < 0.2): # NOTE - constant: below what threshold considered plateau
                start = index
                index += len(list(vals))
                stop = index
                if not is_plateau and transitions[0][0] >= start and transitions[0][0] < stop:
                    transition_windows.append([start, stop, transitions[0][1]])
                    transitions.pop(0)
                    if not transitions:
                        break
            assert (not transitions), "Something went wrong, all transitions should have been assigned their respective windows"
            # Create windows around plateaus
            plateau_windows = [] # start, stop, image.pl_state
            prev_stop = 0
            for start, stop, direction in transition_windows:
                if start != 0:
                    state = Image.PL_State.HIGH if direction == Transition.Direction.HIGH_TO_LOW else Image.PL_State.LOW
                    plateau_windows.append([prev_stop, start, state])
                prev_stop = stop
            if prev_stop != len(images):
                state = Image.PL_State.LOW if transition_windows[-1][-1] == Transition.Direction.HIGH_TO_LOW else Image.PL_State.HIGH
                plateau_windows.append([prev_stop, len(images), state])
            success = True
            break
        assert(success), "Failed to find expected transitions, aborting"
        completion_percent = f"{len(images)} / {len(images)}"
        print(completion_percent) # TODO - omit completion

        # Plot
        # TODO - hide behind debug flag
        # TODO - still create when error occurs above (most imporant time!!)
        # TODO - work better from background thread?
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
        for start, stop, pl_state in plateau_windows:
            colour = "#5E99DB44" if pl_state == Image.PL_State.LOW else "#B7B22944"
            plt.gca().add_patch(matplotlib.patches.Rectangle((start,0),stop-start-1,1,color=colour))
        plt.legend()
        plt.subplot(2,3,6)
        plt.title("Transition detection")
        plt.plot(ids, norm_dsavgol, label=f"d_savgol", linewidth=1)
        for start, stop, pl_state in plateau_windows:
            colour = "#5E99DB44" if pl_state == Image.PL_State.LOW else "#B7B22944"
            plt.gca().add_patch(matplotlib.patches.Rectangle((start,-1),stop-start-1,2,color=colour))
        plt.ylim(-1.1,1.1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{acq_name}_switch-debug.png") # TODO - save in debug folder of output dir
        # plt.show() # TODO - remove