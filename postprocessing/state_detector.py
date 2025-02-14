from dpl_common.helpers import Image, Transition
from dpl_common.config import Config

class StateDetector:
    def __init__(self):
        pass

    # TODO - implement this properly
    def tag_states(self, images: list[Image], transitions: list[Transition], config: Config):
        assert (len(transitions) == 1), "StateDetector cannot handle multiple transitions yet"
        for image in images:
            if abs(image.time - transitions[0].time) < 0.05 or abs(image.time - transitions[0].time) > 0.5:
                continue
            elif image.time < transitions[0].time:
                image.pl_state = Image.PL_State.HIGH if transitions[0].direction == Transition.Direction.HIGH_TO_LOW else Image.PL_State.LOW
            else:
                image.pl_state = Image.PL_State.LOW if transitions[0].direction == Transition.Direction.HIGH_TO_LOW else Image.PL_State.HIGH