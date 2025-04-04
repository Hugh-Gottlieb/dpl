import matplotlib.patches
from typing import Callable
import numpy as np
import cv2
from scipy.signal import savgol_filter, argrelextrema
from itertools import groupby
import os
import shutil
from PIL import Image as PIL_Image
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg") # Non-interactive backend so not a problem from background threads

from dpl_common.helpers import Image, Transition, tif_to_jpeg
from dpl_common.config import Config
from dpl_common.lens_correction import LensCorrection

# TODO - speed optimisation

class TransitionUnknown:
    def __init__(self, config: Config, lens_correction: LensCorrection):
        self.config = config
        self.lens_correction = lens_correction
        kernel_x = cv2.getGaussianKernel(1280, 1024/2)
        kernel_y = cv2.getGaussianKernel(1024, 1024/2)
        kernel = kernel_y * kernel_x.T
        kernel = kernel - kernel.min()
        kernel = kernel / kernel.max()
        kernel[kernel <= 0] = 1e-9 # Not PERFECTLY zero to prevent division problems
        self.gaussian_kernel = kernel

    def tag_and_register_images(self, images: list[Image], register_function: Callable[[Image, Image, str], bool], output_path:str, acq_name: str):
        dark_offset = self.config.get("dark_offset")
        max_transitions = self.config.get("max_transitions")
        debug_name = f"{output_path.split('/')[-2]}:{acq_name}"

        # Register evenly distributed subset of images (rough estimation of switching)
        self.lens_correction.correct_images(images)
        middle_index = round(len(images)/2)
        if max_transitions < 1 or max_transitions > int(len(images)/2)-1:
            raise Exception(f"Invalid switch_pairs config: {max_transitions} not in range [1, {int(len(images)/2)-1}]")
        switch_indexes = np.linspace(0, len(images)-1, (2*max_transitions)+1).astype(int) # Double to ensure found
        middle_image = images[middle_index]
        completed_registrations = 0
        for i, index in enumerate(switch_indexes):
            completed_registrations += 1
            if index != middle_index:
                abort = register_function(images[index], middle_image, f"{completed_registrations} / {len(images)}")
                if abort: return
        if middle_index not in switch_indexes:
            completed_registrations += 1

        # Find switch direction
        # NOTE - always relate to first image, if this is during a transition then things could get weird?
        diff_scores = {Transition.Direction.HIGH_TO_LOW: [], Transition.Direction.LOW_TO_HIGH: []}
        diff_thresholds = {Transition.Direction.HIGH_TO_LOW: [], Transition.Direction.LOW_TO_HIGH: []}
        for index in switch_indexes[1:]:
            direction = self.__get_switch_direction(images[0].data, images[index].data)
            diff_img = self.__get_diff_image(images[0].data, images[index].data, direction)
            did_switch, switch_score = self.__check_if_switched(diff_img)
            if not did_switch: continue
            diff_threshold = (switch_score**2) * np.nanpercentile(diff_img, 100-self.config.get("outlier_percentile"))
            diff_scores[direction].append((switch_score**2) * diff_threshold)
            diff_thresholds[direction].append(diff_threshold)
        # TODO - could exit early HERE if nothing switched (turn debug plot into function that called before exiting)
        score1 = np.average(diff_scores[Transition.Direction.HIGH_TO_LOW]) if diff_scores[Transition.Direction.HIGH_TO_LOW] else 0
        score2 = np.average(diff_scores[Transition.Direction.LOW_TO_HIGH]) if diff_scores[Transition.Direction.LOW_TO_HIGH] else 0
        diff_direction = Transition.Direction.HIGH_TO_LOW if score1 > score2 else Transition.Direction.LOW_TO_HIGH
        diff_threshold = np.average(diff_thresholds[diff_direction]) if diff_thresholds[diff_direction] else 1
        diff_switch_cutoff = diff_threshold * 0.2 # TODO - make this configurable?

        # Get valid diff images, work out expected transitions
        expected_transitions, diff_imgs = [], []
        for index in switch_indexes[1:]:
            diff_img = self.__get_diff_image(images[0].data, images[index].data, diff_direction)
            did_switch, _ = self.__check_if_switched(diff_img, diff_switch_cutoff) # NOTE: constant, remove parts of image that really didn't switch
            if did_switch == True:
                diff_imgs.append(diff_img)
                expected_transitions.append(diff_direction)
            elif did_switch == False and expected_transitions: # If had switched before and no longer switched, then switched back (opposite of first switch)
                expected_transitions.append(Transition.Direction.HIGH_TO_LOW if diff_direction == Transition.Direction.LOW_TO_HIGH else Transition.Direction.LOW_TO_HIGH)
        expected_transitions = [key for key, _ in groupby(expected_transitions)] # Remove consecutive duplicates

        # Create debug plot
        if self.config.get("debug_switching"):
            fig, axes = plt.subplots(2,len(switch_indexes)-1, figsize=(19,9), layout="constrained")
            fig.suptitle([t.name for t in expected_transitions])
            for i, index in enumerate(switch_indexes[1:]):
                _direction = self.__get_switch_direction(images[0].data, images[index].data)
                _diff_img = self.__get_diff_image(images[0].data, images[index].data, diff_direction)
                _did_switch, _switched_score = self.__check_if_switched(_diff_img, diff_switch_cutoff)
                _switch_mask = self.__get_switched_mask(_diff_img, diff_switch_cutoff)
                _diff_img = np.clip(np.nan_to_num(_diff_img), 0, diff_threshold) / diff_threshold # Nans to 0, normalise scaling
                axes[0,i].imshow(tif_to_jpeg(_diff_img,0,True)[:,:,::-1]) # Flip channels so displayed correctly
                axes[0,i].set_title(f"{_direction.name}")
                axes[1,i].imshow(_switch_mask)
                axes[1,i].set_title(f"{_did_switch} ({round(_switched_score,3)})")
            figure_dir = os.path.join(output_path, "debug")
            if not os.path.exists(figure_dir): os.makedirs(figure_dir)
            figure_path = os.path.join(figure_dir, f"{acq_name}_switching-detection.png")
            if os.path.exists(figure_path): os.remove(figure_path)
            fig.savefig(figure_path, format="png")
            time.sleep(1)

        # Get module vs background area
        if not diff_imgs:
            raise Exception("No switches detected")
        diff_img = np.clip(np.mean(diff_imgs, axis=0), 0, None)
        switch_mask = self.__get_switched_mask(diff_img)
        background_mask = self.__get_background_mask(diff_img)

        # Complete remaining registrations
        for i, image in enumerate(images):
            if i in switch_indexes or i == middle_index:
                continue
            completed_registrations += 1
            abort = register_function(image, middle_image, f"{completed_registrations} / {len(images)}")
            if abort: return
        if self.config.get("debug_registration"):
            registered_root = os.path.join(output_path, "debug", acq_name + "_registered")
            if os.path.exists(registered_root):
                shutil.rmtree(registered_root)
            os.makedirs(registered_root)
            for id, image in enumerate(images):
                pil_image = PIL_Image.fromarray(image.data)
                pil_image.save(os.path.join(registered_root, f"{acq_name}_{id:03}.tif"))

        # Get PL signal, correcting for changes in background illumination
        background_signal = np.array([np.nanmean(img.data[background_mask]) for img in images]) - dark_offset
        switched_signal = np.array([np.nanmean(img.data[switch_mask]) for img in images]) - dark_offset
        background_ratio = np.average(background_signal) / background_signal
        background_ratio = 1 # TODO - decide re background signal (just seems to be making things worse???) @Oliver ...
        pl_signal = switched_signal * background_ratio

        # Detect transitions
        # TODO - replace percent with a configurable time window?
        savgol_size_percent = (0.2 / max_transitions) / 0.8 # Divide by 0.8 to pre-empt initial decrease
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
                if self.config.get("debug_switching"): print(f"{debug_name} - savgol no flat regions detected, reducing window size")
                continue
            # Get all transition options
            high_to_low_options = [(index, Transition.Direction.HIGH_TO_LOW) for index in argrelextrema(norm_dsavgol, np.less, mode="clip")[0] if norm_dsavgol[index] < -0.5]
            low_to_high_options = [(index, Transition.Direction.LOW_TO_HIGH) for index in argrelextrema(norm_dsavgol, np.greater, mode="clip")[0] if norm_dsavgol[index] > 0.5]
            if len(high_to_low_options) < expected_high_to_low or len(low_to_high_options) < expected_low_to_high: # If dont get enough transitions, try more sensitive
                if self.config.get("debug_switching"): print(f"{debug_name} savgol insufficient transitions detected, reducing window size")
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
                if not option_indices: # Error: try more sensitive savgol
                    if self.config.get("debug_switching"): print(f"{debug_name} savgol cannot match expected transitions, reducing window size")
                    error = True
                    break
                option_vals = [abs_norm_dsavgol[index] for index in option_indices]
                best_option_index = option_indices[np.argmax(option_vals)]
                transitions.append((best_option_index, expected_transition_dir))
            if error: continue # Error occured, try more sensitive
            # TODO - update algo - work backwards from biggest (find max, remove that and all adjacent of same direction, repeat)
            #                    - potentially some left at the end, but who cares
            #                    - if doesn't line up somehow (ordering wrong?) then repeat with smaller savgol
            #                    - also if have < 5 HIGH or LOW images?
            #                    - put that in a function so easy to quit and restart at any time
            if transition_options:
                if self.config.get("debug_switching"): print(f"{debug_name} savgol fulfilled all expected transitions, with remaining options ({transition_options})")
            success = True
            break
        assert(success), "Failed to find expected transitions, aborting"

        # Get windows around transitions
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
        prev_stop = 0
        plateau_windows = [] # start, stop, image.pl_state
        for start, stop, direction in transition_windows:
            if start != 0:
                state = Image.PL_State.HIGH if direction == Transition.Direction.HIGH_TO_LOW else Image.PL_State.LOW
                plateau_windows.append([prev_stop, start, state])
            prev_stop = stop
        if prev_stop != len(images):
            state = Image.PL_State.LOW if transition_windows[-1][-1] == Transition.Direction.HIGH_TO_LOW else Image.PL_State.HIGH
            plateau_windows.append([prev_stop, len(images), state])

        # Tag the images with their states
        for image in images:
            image.pl_state = Image.PL_State.UNKNOWN
        for start, stop, pl_state in plateau_windows:
            for image in images[start:stop]:
                image.pl_state = pl_state

        # Plot
        # TODO - still create when error occurs above (most imporant time!!) - make into function that called before early return?
        if self.config.get("debug_switching"):
            fig, axes = plt.subplots(2,3, figsize=(19,9), layout="constrained")
            axes[0][0].set_title("Rough PL image")
            diff_img[np.isnan(diff_img)] = 0
            axes[0][0].imshow(tif_to_jpeg(diff_img,5,True)[:,:,::-1]) # Flip channels so displayed correctly
            axes[0][1].set_title("Switched area")
            axes[0][1].imshow(switch_mask)
            axes[0][2].set_title("Background area")
            axes[0][2].imshow(background_mask)
            axes[1][0].set_title("Raw signals")
            ids = list(range(len(pl_signal)))
            all_signal = np.array([np.nanmean(img.data) for img in images]) - dark_offset
            axes[1][0].plot(ids, all_signal, label="all")
            axes[1][0].plot(ids, background_signal, label="background")
            axes[1][0].plot(ids, switched_signal, label="switched")
            axes[1][0].plot(ids, pl_signal, label="pl")
            axes[1][0].legend()
            axes[1][1].set_title("Normalised signals")
            axes[1][1].plot(ids, self.__normalise_signal(all_signal), label="all")
            axes[1][1].plot(ids, self.__normalise_signal(background_signal), label="background")
            axes[1][1].plot(ids, self.__normalise_signal(switched_signal), label="switched")
            axes[1][1].plot(ids, self.__normalise_signal(pl_signal), label="pl", linewidth=1)
            savgol = savgol_filter(pl_signal, savgol_size, 1)
            axes[1][1].plot(ids, self.__normalise_signal(savgol), label=f"savgol", linewidth=3)
            for start, stop, pl_state in plateau_windows:
                colour = "#5E99DB44" if pl_state == Image.PL_State.LOW else "#B7B22944"
                axes[1][1].add_patch(matplotlib.patches.Rectangle((start,0),stop-start-1,1,color=colour))
            axes[1][1].legend()
            axes[1][2].set_title("Transition detection")
            axes[1][2].plot(ids, norm_dsavgol, label=f"d_savgol", linewidth=1)
            for start, stop, pl_state in plateau_windows:
                colour = "#5E99DB44" if pl_state == Image.PL_State.LOW else "#B7B22944"
                axes[1][2].add_patch(matplotlib.patches.Rectangle((start,-1),stop-start-1,2,color=colour))
            axes[1][2].set_ylim(-1.1,1.1)
            axes[1][2].legend()
            figure_dir = os.path.join(output_path, "debug")
            if not os.path.exists(figure_dir): os.makedirs(figure_dir)
            figure_path = os.path.join(figure_dir, f"{acq_name}_state-detection.png")
            if os.path.exists(figure_path): os.remove(figure_path)
            fig.savefig(figure_path, format="png")
            time.sleep(1)

    def __get_diff_image(self, first_img:np.ndarray, second_img:np.ndarray, direction:Transition.Direction) -> np.ndarray:
        diff_img = (first_img - second_img) if direction == Transition.Direction.HIGH_TO_LOW else (second_img - first_img)
        return np.clip(diff_img, 0, None)

    # TODO - make this based on config params? Tune it? Lower vals now appropriate, since remove pixels those that didn't switch?
    def __check_if_switched(self, diff_img: np.ndarray, threshold: float = 0) -> tuple[bool, float]:
        signal_percentile = self.config.get("signal_percentile")
        switch_mask = self.__get_switched_mask(diff_img, threshold)
        remaining_area = (np.count_nonzero(switch_mask)/switch_mask.size)/(signal_percentile/100)
        switched = True if remaining_area > 0.3 else False if remaining_area < 0.1 else None
        return switched, remaining_area

    def __get_switch_direction(self, first_img:np.ndarray, second_img:np.ndarray) -> Transition.Direction:
        outlier_percentile, signal_percentile = self.config.get("outlier_percentile"), self.config.get("signal_percentile")
        # Remove extremes (outliers)
        diff_img = first_img - second_img
        outlier_low, outlier_high = np.nanpercentile(diff_img, outlier_percentile), np.nanpercentile(diff_img, 100 - outlier_percentile)
        diff_img[np.logical_or(diff_img < outlier_low, diff_img > outlier_high)] = np.nan
        # Blur on HUGE scale, to cancel out ground noise
        nan_mask = np.isnan(diff_img)
        diff_img[nan_mask] = 0 # blur turns whole thing to nans if any present
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
        positive_cutoff = np.nanpercentile(diff_img, 100 - signal_percentile)
        negative_cutoff = np.nanpercentile(diff_img, signal_percentile)
        positive_sum = np.sum(np.square(diff_img[diff_img >= positive_cutoff]))
        negative_sum = np.sum(np.square(diff_img[diff_img <= negative_cutoff]))
        direction = Transition.Direction.HIGH_TO_LOW if positive_sum > negative_sum else Transition.Direction.LOW_TO_HIGH
        return direction

    def __get_switched_mask(self, diff_img: np.ndarray, threshold: float = 0) -> np.ndarray:
        size_factor = self.config.get("morphological_size_factor")
        outlier_percentile, signal_percentile = self.config.get("outlier_percentile"), self.config.get("signal_percentile")
        switch_percentile_low, switch_percentile_high = 100 - outlier_percentile - signal_percentile, 100 - outlier_percentile
        switch_diff_img = diff_img * self.gaussian_kernel # Devalue outside edges
        switch_low, switch_high = np.nanpercentile(switch_diff_img, switch_percentile_low), np.nanpercentile(switch_diff_img, switch_percentile_high)
        switch_mask = np.logical_and(switch_diff_img >= switch_low, np.logical_and(switch_diff_img <= switch_high, diff_img > threshold)).astype(np.uint8)
        size1, size2 = round(size_factor), round(2 * size_factor)
        if size1 >= 1:
            switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(size1, size1)),iterations=2) # Remove small noise
        if size2 >= 1:
            switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(size2, size2)),iterations=1) # Close gaps
            switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(size2, size2)),iterations=3) # Remove bigger noise
        if size1 >= 1:
            switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(size1, size1)),iterations=3) # Remove more small noise
        switch_mask = switch_mask.astype(bool)
        return switch_mask

    def __get_background_mask(self, diff_img: np.ndarray) -> np.ndarray:
        outlier_percentile, signal_percentile = self.config.get("outlier_percentile"), self.config.get("signal_percentile")
        background_percentile_low, background_percentile_high = outlier_percentile, outlier_percentile + signal_percentile
        background_diff_img = diff_img / self.gaussian_kernel # Raise value of outside edges so not picked
        background_low, background_high = np.nanpercentile(background_diff_img, background_percentile_low), np.nanpercentile(background_diff_img, background_percentile_high)
        background_mask = np.logical_and(background_diff_img >= background_low, background_diff_img <= background_high).astype(np.uint8)
        background_mask = background_mask.astype(bool)
        return background_mask

    def __normalise_signal(self, signal: np.ndarray) -> np.ndarray:
        return (signal - signal.min()) / (signal.max() - signal.min())