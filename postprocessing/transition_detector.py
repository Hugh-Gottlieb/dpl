from dpl_common.helpers import Image, Transition, tif_to_jpeg
from dpl_common.registration import Registration
from dpl_common.lens_correction import LensCorrection
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter

# NOTE: at the moment this is a shell class, since all processed datasets should have a tagged transition

class TransitionDetector:
    def __init__(self):
        kernel_x = cv2.getGaussianKernel(1280, 1024/2)
        kernel_y = cv2.getGaussianKernel(1024, 1024/2)
        kernel = kernel_y * kernel_x.T
        kernel = kernel - kernel.min()
        kernel = kernel / kernel.max()
        kernel[kernel <= 0] = 1e-9 # Not PERFECTLY zero to prevent division problems
        self.gaussian_kernel = kernel
        self.registration = Registration(feature_limit=2500)
        self.lens_correction = LensCorrection()

    # NOTE: dont love passing in lens name :(
    def detect_transitions(self, images: list[Image], lens_name: str) -> list[Transition]:
        # Config params
        offset = round(len(images) * 0.1) # This assumes ONE transition, which are still valid 10% from start and end
        direction = Transition.Direction.HIGH_TO_LOW
        # direction = Transition.Direction.LOW_TO_HIGH
        switch_percentile_low, switch_percentile_high = 85, 95
        background_percentile_low, background_percentile_high = 5, 15

        # Full registration
        target_image = images[round(len(images)/2)]
        self.lens_correction.correct_images(images, lens_name)
        self.registration.register_images(images, target_image, skip_target=True)

        # Calculate gaussian-kernel-weighted diff images
        before_image = images[offset].data
        after_image = images[-1-offset].data
        raw_diff_img = (before_image - after_image) if direction == Transition.Direction.HIGH_TO_LOW else (after_image - before_image)
        switch_diff_img = raw_diff_img * self.gaussian_kernel # Devalue outside edges
        background_diff_img = raw_diff_img / self.gaussian_kernel # Raise value of outside edges so not picked

        # Switched area mask
        switch_low, switch_high = np.nanpercentile(switch_diff_img, switch_percentile_low), np.nanpercentile(switch_diff_img, switch_percentile_high)
        switch_mask = np.logical_and(switch_diff_img >= switch_low, switch_diff_img <= switch_high).astype(np.uint8)
        switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,1)),iterations=2) # Remove small noise
        switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations=1) # Close gaps
        switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations=3) # Remove bigger noise
        switch_mask = cv2.morphologyEx(switch_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,1)),iterations=3) # Remove more small noise
        switch_mask = switch_mask.astype(bool)

        # Background mask
        background_low, background_high = np.nanpercentile(background_diff_img, background_percentile_low), np.nanpercentile(background_diff_img, background_percentile_high)
        background_mask = np.logical_and(background_diff_img >= background_low, background_diff_img <= background_high).astype(np.uint8)
        background_mask = background_mask.astype(bool)

        # Detect non-switch: switched area all noise so most of it removed
        # NOTE: could detect early, these masks are just based on 2 images which could be registered first
        switched_noise = (np.count_nonzero(switch_mask)/switch_mask.size)/((switch_percentile_high - switch_percentile_low)/100)
        if switched_noise < 0.35:
            print("No switch detected")
            return

        # Correct for changes in background illumination
        all_signal = np.array([np.nanmean(img.data) for img in images])
        background_signal = np.array([np.nanmean(img.data[background_mask]) for img in images])
        switched_signal = np.array([np.nanmean(img.data[switch_mask]) for img in images])
        background_ratio = np.average(background_signal) / background_signal
        pl_signal = switched_signal * background_ratio

        # Setup smoothing convolve window
        median_window_size = round(len(images) * 0.15)
        smooth_filter_size = round(median_window_size * 0.25)
        smooth_filter = np.power(np.concatenate((np.linspace(0,1,smooth_filter_size)[1:], np.linspace(1,0,smooth_filter_size)[:-1])), 0.5)
        smooth_filter /= np.sum(smooth_filter)
        derivative_window_size = 3

        # Calculate median filter, including smoothed version and derivative
        median_filter = []
        for i in range(len(pl_signal) - median_window_size):
            median_filter.append(np.median(pl_signal[i:i+median_window_size]))
        smoothed_median_filter = np.convolve(median_filter, smooth_filter, mode="valid")
        d_smoothed_median_filter = []
        for i in range(len(smoothed_median_filter) - derivative_window_size):
            d_smoothed_median_filter.append(smoothed_median_filter[i+derivative_window_size] - smoothed_median_filter[i])

        # Test: Savgol filter
        raw_savgol = savgol_filter(pl_signal, median_window_size, 1)
        d_raw_savgol = savgol_filter(pl_signal, median_window_size, 1, 1)
        median_savgol = savgol_filter(median_filter, smooth_filter_size, 1)
        d_median_savgol = savgol_filter(median_filter, smooth_filter_size, 1, 1)

        # TODO: detect switching point / plateau area's based on 0-derivative?
        #   - Normalise (around 0) -> decide if 0 or plateau or unknown. Transition is 0 -> local min (if HIGH_TO_LOW) -> 0
        #   - Or use the straight lines somehow else, derivative very sensitive. Smoothed derivative (over more points than just 2, rather than convolve to smooth)
        # TODO: detect IF high-to-low or low-to-high (prev algo existed?)

        # Plot
        plt.figure()
        plt.subplot(2,3,1)
        raw_diff_img[np.isnan(raw_diff_img)] = 0
        plt.imshow(tif_to_jpeg(raw_diff_img,5,True))
        plt.subplot(2,3,2)
        plt.imshow(switch_mask)
        plt.subplot(2,3,3)
        plt.imshow(background_mask)
        plt.subplot(2,3,4)
        ids = list(range(len(pl_signal)))
        # plt.plot(ids, all_signal, label="all")
        # plt.plot(ids, background_signal, label="background")
        # plt.plot(ids, switched_signal, label="switched")
        # plt.plot(ids, pl_signal, label="pl")
        plt.plot(ids, (all_signal-all_signal.min())/(all_signal.max()-all_signal.min()), label="all")
        plt.plot(ids, (background_signal-background_signal.min())/(background_signal.max() - background_signal.min()), label="background")
        plt.plot(ids, (switched_signal - switched_signal.min())/(switched_signal.max() - switched_signal.min()), label="switched")
        plt.plot(ids, (pl_signal - pl_signal.min())/(pl_signal.max() - pl_signal.min()), label="pl")
        plt.legend()
        plt.subplot(2,3,5)
        s = 0
        plt.plot(ids[s:len(raw_savgol)], raw_savgol, label="raw_savgol")
        s = int(median_window_size/2)
        plt.plot(ids[s:s+len(median_savgol)], median_savgol, label="median_savgol")
        plt.plot(ids[s:s+len(median_filter)], median_filter, label=f"median_{median_window_size}")
        s += int(len(smooth_filter)/2) - 1
        plt.plot(ids[s:s+len(smoothed_median_filter)], smoothed_median_filter, label=f"smooth_{len(smooth_filter)}_median_{median_window_size}")
        plt.legend()
        plt.subplot(2,3,6)
        plt.plot(ids[s:s+len(d_smoothed_median_filter)], d_smoothed_median_filter, label=f"d_median_{median_window_size}")
        s = 0
        plt.plot(ids[s:s+len(d_raw_savgol)], d_raw_savgol, label="d_raw_savgol")
        s = int(median_window_size/2)
        plt.plot(ids[s:s+len(d_median_savgol)], d_median_savgol, label="d_median_savgol")
        plt.legend()
        plt.show()