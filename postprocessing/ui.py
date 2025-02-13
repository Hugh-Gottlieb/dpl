from dpl_common.config import Config, get_config_path
from dpl_common.mission import Mission
from dpl_common.acquisition import Acquisition
from dpl_common.helpers import *

config = Config(get_config_path(__file__))

root = r"C:\Users\z5485746\Documents\test_flight"
mission = Mission(root)
acqs = mission.get_acquisitions()
print(acqs[0].get_name())
print(acqs[0].get_pl_image())
print(acqs[0].get_pl_image().get_status())
print(acqs[0].get_gps_info())
print(acqs[0].get_camera_info())
print(acqs[0].get_gimbal_info())
print(acqs[0].get_transitions())
print(acqs[0].get_imgs(True))

# ---- NOTE -----
# This is all very WIP, was orig in dpl_common but doesn't belong there!

# from enum import Enum, auto
# class ProcessStatus(Enum):
#     NO = auto()
#     YES = auto()
#     STALE = auto() # Processed, but with old settings
#     IN_PROGRESS = auto()
#     QUEUED = auto()
#     ERROR = auto()

# # TODO - fix up to use name-status dict
# def process_remaining(self):
#     for acquisition in self.acquisitions:
#         if acquisition.get_status() == ProcessStatus.NO:
#             acquisition.set_status(ProcessStatus.QUEUED)
#     for acquisition in self.acquisitions:
#         if acquisition.get_status() == ProcessStatus.QUEUED:
#             acquisition.set_status(ProcessStatus.IN_PROGRESS)
#             acquisition.process()

# def clear_all(self):
#     for acquisition in self.acquisitions:
#         acquisition.clear()

# # TODO
# def process(self):
#     # Detect transitions
#     # Detect PL States
#     # Lens correct relevant imgs
#     # Register relevant imgs
#     # Create pl image from relevant imgs
#     pass