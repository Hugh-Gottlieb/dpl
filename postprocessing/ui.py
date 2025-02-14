from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem
import sys
from enum import Enum, auto
import os
from dataclasses import dataclass
import time
import traceback
import numpy as np

from dpl_common.helpers import Image
from dpl_common.config import Config, get_config_path
from dpl_common.mission import Mission
from dpl_common.acquisition import Acquisition
from dpl_common.lens_correction import LensCorrection
from dpl_common.registration import Registration

from postprocessing.postprocessing_ui import Ui_MainWindow
from postprocessing.state_detector import StateDetector
from postprocessing.transition_detector import TransitionDetector

class MainWindow(QMainWindow, Ui_MainWindow):

    @dataclass
    class AcquisitionStatus:
        # NOTE - first three must have this name, to match PL Image. Python should allow enum inheritance!
        class Status(Enum):
            NO = auto()
            YES = auto()
            STALE = auto()
            IN_PROGRESS = auto()
            QUEUED = auto()
            ERROR = auto()
        row: int
        status: Status

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.acquisition_table.setColumnWidth(0, 10) # Set first column (displaying colour) to be very small
        self.__setup_status_text_and_colour()

        self.config = Config(get_config_path(__file__))
        self.lens_correction = LensCorrection()
        self.registration = Registration()
        self.transition_detector = TransitionDetector()
        self.state_detector = StateDetector()
        self.mission = None
        self.acq_status = {}

    def set_mission(self):
        options = QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder', '', options=options)
        if not folder_path:
            return
        self.mission_path.setText(folder_path)
        self.mission = Mission(folder_path)
        self.acquisition_table.setRowCount(0)
        for acq in self.mission.get_acquisitions():
            row = self.acquisition_table.rowCount()
            self.acq_status[acq.get_name()] = self.AcquisitionStatus(row, self.AcquisitionStatus.Status[acq.get_pl_image().get_status(self.config).name])
            self.acquisition_table.insertRow(row)
            self.__update_table_status(acq.get_name())
            self.acquisition_table.setItem(row, 2, QTableWidgetItem(acq.get_name()))
        self.resize(self.sizeHint())
        self.log.appendPlainText(f"Loaded mission {self.mission.get_folder()}")

    def __update_table_status(self, acq_name: str):
        acq_status = self.acq_status[acq_name]
        status_item = QTableWidgetItem()
        status_item.setBackground(self.status_colours[acq_status.status])
        self.acquisition_table.setItem(acq_status.row, 0, status_item)
        self.acquisition_table.setItem(acq_status.row, 1, QTableWidgetItem(self.status_text[acq_status.status]))
        QApplication.processEvents() # NOTE - only needed to repaint if not processing in background thread

    def process_mission(self):
        if self.mission is None:
            self.log.appendPlainText("Error: no mission selected")
            return
        # TODO - check not already running ... or disable button?
        # Update status
        for acq in self.mission.get_acquisitions():
            acq_status = self.acq_status[acq.get_name()]
            if acq_status.status == self.AcquisitionStatus.Status.NO:
                acq_status.status = self.AcquisitionStatus.Status.QUEUED
        self.log.appendPlainText(f"Processing mission {self.mission.get_folder()}")
        # TODO kick up background thread pool and handle there
        # TODO kick up background thread to update status / error msgs in UI?
        # TODO have some way of tracking when processing is ongoing / finished
        # TODO - OR - disable the buttons, go to work in the main thread, with relevant updates, and then re-enable when done.
        # Start processing
        if not os.path.exists(self.mission.get_analysed_folder()):
            os.mkdir(self.mission.get_analysed_folder())
        for acq in self.mission.get_acquisitions():
            if self.acq_status[acq.get_name()].status == self.AcquisitionStatus.Status.QUEUED:
                self.__process_acq(acq)

    def __process_acq(self, acq: Acquisition):
        self.acq_status[acq.get_name()].status = self.AcquisitionStatus.Status.IN_PROGRESS
        start_time = time.time()
        try:
            images = acq.get_imgs(load_data=True)
            transitions = acq.get_transitions()
            if len(transitions) == 0:
                transitions = self.transition_detector.detect_transitions(images)
            assert (len(transitions) == 1), "Multiple transitions cannot be handled yet"
            self.state_detector.tag_states(images, transitions, self.config)
            transition_img = images[np.argmin(np.abs([(image.time - transitions[0].time) for image in images]))]
            relevant_imgs = [image for image in images if image.pl_state != Image.PL_State.UNKNOWN]
            self.lens_correction.correct_images(relevant_imgs, self.lens_correction.get_lens_names()[0]) # TODO - get name from config, which is a drop-down in the UI
            self.registration.register_images(relevant_imgs, transition_img)
            acq.get_pl_image().create(relevant_imgs, self.config, acq.get_gps_info(), acq.get_gimbal_info(), acq.get_camera_info())
            acq.get_pl_image().save(self.mission.get_analysed_folder())
            self.acq_status[acq.get_name()].status = self.AcquisitionStatus.Status.YES
            self.log.appendPlainText(f"Success: {acq.get_name()} ({round(time.time() - start_time, 3)}s)")
        except Exception as e:
            print(traceback.format_exc())
            self.acq_status[acq.get_name()].status = self.AcquisitionStatus.Status.ERROR
            self.log.appendPlainText(f"Error: {acq.get_name()} - {e} ({round(time.time() - start_time, 3)}s)")
        self.__update_table_status(acq.get_name())

    def clear_mission(self):
        if self.mission is None:
            self.log.appendPlainText("Error: no mission selected")
            return
        # TODO - check if processing currently in progress, and prevent? Stop? Disable button?
        for acq in self.mission.get_acquisitions():
            acq.get_pl_image().clear(self.mission.get_analysed_folder())
            self.acq_status[acq.get_name()].status = self.AcquisitionStatus.Status.NO
            self.__update_table_status(acq.get_name())
        self.log.appendPlainText(f"Cleared mission {self.mission.get_folder()}")

    def __setup_status_text_and_colour(self):
        self.status_colours = {
            self.AcquisitionStatus.Status.NO: QBrush(QColor(183, 178, 41)), # yellow
            self.AcquisitionStatus.Status.STALE: QBrush(QColor(183, 116, 45)), # orange
            self.AcquisitionStatus.Status.ERROR: QBrush(QColor(182, 46, 50)), # red
            self.AcquisitionStatus.Status.YES: QBrush(QColor(77, 187, 106)), # green
            self.AcquisitionStatus.Status.QUEUED: QBrush(QColor(94, 153, 219)), # blue
            self.AcquisitionStatus.Status.IN_PROGRESS: QBrush(QColor(133, 68, 183)), # purple
        }
        self.status_text = {
            self.AcquisitionStatus.Status.NO: "Not Done",
            self.AcquisitionStatus.Status.STALE: "Stale",
            self.AcquisitionStatus.Status.ERROR: "Error",
            self.AcquisitionStatus.Status.YES: "Done",
            self.AcquisitionStatus.Status.QUEUED: "Queued",
            self.AcquisitionStatus.Status.IN_PROGRESS: "In Progress",
        }

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())