import numpy as np
from typing import Callable

from dpl_common.helpers import Image, Transition
from dpl_common.config import Config
from dpl_common.lens_correction import LensCorrection

class TransitionKnown:
    def __init__(self, config: Config, lens_correction: LensCorrection):
        self.config = config
        self.lens_correction = lens_correction

    def tag_and_register_images(self, images: list[Image], transitions: list[Transition], register_function: Callable[[Image, Image, str], bool], debug_path:str = None):
        # Tag states
        assert (len(transitions) == 1), "StateDetector cannot handle multiple transitions yet"
        min_time = self.config.get("transition_halftime_ms")
        max_time = self.config.get("plateau_time_ms") + min_time
        pre_state, post_state = (Image.PL_State.HIGH, Image.PL_State.LOW) if transitions[0].direction == Transition.Direction.HIGH_TO_LOW else (Image.PL_State.LOW, Image.PL_State.HIGH)
        for image in images:
            dtime_ms = abs(image.time - transitions[0].time) * 1e3
            if dtime_ms < min_time or dtime_ms > max_time:
                image.pl_state = Image.PL_State.UNKNOWN
            else:
                image.pl_state = pre_state if image.time < transitions[0].time else post_state
        # Lens correct and register relevant images
        transition_image = images[np.argmin(np.abs([(image.time - transitions[0].time) for image in images]))]
        transition_image.pl_state = Image.PL_State.UNKNOWN
        self.lens_correction.correct_image(transition_image)
        relevant_images = [image for image in images if image.pl_state != Image.PL_State.UNKNOWN]
        self.lens_correction.correct_images(relevant_images)
        for i, image in enumerate(relevant_images):
            abort = register_function(image, transition_image, f"{i} / {len(relevant_images)}")
            if debug_path is not None:
                # TODO - save registered image to debug path (+- folder names ...)
                analysed_path, acq_name = debug_path
            if abort: return