from dpl_common.helpers import Image, Transition
from dpl_common.config import Config

class StateDetector:
    def __init__(self, config: Config):
        self.config = config

    def tag_states(self, images: list[Image], transitions: list[Transition]):
        assert (len(transitions) == 1), "StateDetector cannot handle multiple transitions yet"
        assert (all([image.pl_state == Image.PL_State.UNKNOWN for image in images])), "StateDetector cannot handle partially tagged images yet"
        min_time = self.config.get("transition_halftime_ms")
        max_time = self.config.get("plateau_time_ms") + min_time
        pre_state, post_state = (Image.PL_State.HIGH, Image.PL_State.LOW) if transitions[0].direction == Transition.Direction.HIGH_TO_LOW else (Image.PL_State.LOW, Image.PL_State.HIGH)
        for image in images:
            dtime_ms = abs(image.time - transitions[0].time) * 1e3
            if dtime_ms < min_time or dtime_ms > max_time:
                continue
            image.pl_state = pre_state if image.time < transitions[0].time else post_state