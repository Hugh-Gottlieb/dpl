from dpl_common.helpers import Image, Transition

# NOTE: at the moment this is a shell class, since all processed datasets should have a tagged transition

class TransitionDetector:
    def detect_transitions(self, images: list[Image]) -> list[Transition]:
        raise NotImplementedError("detect_transition has not been implemented")