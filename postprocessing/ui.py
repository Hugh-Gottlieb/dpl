from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QPushButton, QComboBox
from PySide6.QtCore import QTimer
import sys
from enum import Enum, auto
import os
from dataclasses import dataclass
import time
import traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from dpl_common.helpers import Image
from dpl_common.config import Config, get_config_path
from dpl_common.mission import Mission
from dpl_common.acquisition import Acquisition
from dpl_common.lens_correction import LensCorrection
from dpl_common.registration import Registration

from postprocessing.postprocessing_ui import Ui_MainWindow
from postprocessing.state_detector import StateDetector
from postprocessing.transition_detector import TransitionDetector

# TODO: try different registration strategies
#   - bright and dark in sets, then each average together?
#   - to adjacent, and propogate the transform matrix along?

# TODO: add debug flag to config, which saves registered images and state detection info

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
        # Init variables
        self.config = Config(get_config_path(__file__))
        self.lens_correction = LensCorrection()
        self.registration = Registration(feature_limit=2500)
        self.transition_detector = TransitionDetector()
        self.state_detector = StateDetector(self.config)
        self.mission = None
        self.acq_status = {}
        self.processing_status = {}
        self.running_processing = False
        self.abort_processing = False
        self.pending_msgs = []
        self.pending_msgs_lock = Lock()
        self.update_status_thread = QTimer(self)
        self.update_status_thread.timeout.connect(self.__update_status_thread)
        self.thread_pool = None
        # Setup UI
        self.acquisition_table.setColumnWidth(0, 10) # Set first column (displaying colour) to be very small
        self.__setup_status_text_and_colour()
        self.__setup_lens_options()
        self.__enable_buttons()

    def closeEvent(self, event):
        if self.running_processing:
            self.stop_processing()
        super(QMainWindow, self).closeEvent(event)

    def set_mission(self):
        if self.running_processing:
            self.log.appendPlainText("Error: processing already running")
            return
        options = QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder', '', options=options)
        if not folder_path:
            return
        self.mission_path.setText(folder_path)
        try:
            self.mission = Mission(folder_path)
        except Exception as e:
            print(traceback.format_exc())
            self.log.appendPlainText(f"Failed to load mission {folder_path}: {e}")
            return
        self.acquisition_table.setRowCount(0)
        for acq in self.mission.get_acquisitions():
            row = self.acquisition_table.rowCount()
            self.acquisition_table.insertRow(row)
            self.acquisition_table.setItem(row, 2, QTableWidgetItem(acq.get_name()))
            self.acq_status[acq] = self.AcquisitionStatus(row, None) # Init status to None, actually gets set in the line below
            self.__update_acquisition_status(acq, recheck=True)
        self.resize(self.sizeHint())
        self.log.appendPlainText(f"Loaded mission {self.mission.get_folder()}")

    def process_mission(self):
        if self.mission is None:
            self.log.appendPlainText("Error: no mission selected")
            return
        if self.running_processing:
            self.log.appendPlainText("Error: processing already running")
            return
        if self.lens_selection.currentText() not in self.lens_correction.get_lens_names():
            self.log.appendPlainText("Error: unknown lens selected")
            return
        self.lens_correction.set_default_lens(self.lens_selection.currentText())
        self.running_processing = True
        self.__enable_buttons()
        self.thread_pool = ThreadPoolExecutor(max_workers=int(max(1, os.cpu_count()/2)))
        if not os.path.exists(self.mission.get_analysed_folder()):
            os.mkdir(self.mission.get_analysed_folder())
        for acq in self.mission.get_acquisitions():
            if self.acq_status[acq].status == self.AcquisitionStatus.Status.NO:
                self.acq_status[acq].status = self.AcquisitionStatus.Status.QUEUED
                self.processing_status[acq] = None
                self.__process_acq(acq)
                # self.thread_pool.submit(self.__process_acq, acq)
        self.update_status_thread.start((1/10)*1e3) # 10Hz
        self.log.appendPlainText(f"Processing mission {self.mission.get_folder()} ({self.thread_pool._max_workers} threads)")

    def __update_status_thread(self):
        self.__write_pending_msgs()
        finished_acqs = []
        for acq, percent in self.processing_status.items():
            if self.acq_status[acq].status not in [self.AcquisitionStatus.Status.IN_PROGRESS, self.AcquisitionStatus.Status.QUEUED]:
                finished_acqs.append(acq)
                percent = None
            self.__update_acquisition_status(acq, recheck=False, post_text=percent)
        for acq in finished_acqs:
            self.processing_status.pop(acq)
        if not self.processing_status: # Dict now empty, all removed
            self.__conclude_processing()
            self.log.appendPlainText(f"Finished mission {self.mission.get_folder()}")

    def __write_pending_msgs(self):
        with self.pending_msgs_lock:
            for msg in self.pending_msgs:
                self.log.appendPlainText(msg)
            self.pending_msgs = []

    def __process_acq(self, acq: Acquisition):
        self.acq_status[acq].status = self.AcquisitionStatus.Status.IN_PROGRESS
        start_time = time.time()
        try:
            images = acq.get_imgs(load_data=True)
            assert (len(images) > 0), "No image data available"
            transitions = acq.get_transitions()
            if len(transitions) == 0:
                transitions = self.transition_detector.detect_transitions(images, self.lens_selection.currentText(), acq.get_name()) # TODO - cleanup
            assert (len(transitions) == 1), "Multiple transitions cannot be handled yet"
            self.state_detector.tag_states(images, transitions)
            transition_img = images[np.argmin(np.abs([(image.time - transitions[0].time) for image in images]))]
            transition_img.pl_state = Image.PL_State.UNKNOWN
            relevant_imgs = [image for image in images if image.pl_state != Image.PL_State.UNKNOWN]
            self.lens_correction.correct_images(relevant_imgs)
            self.lens_correction.correct_image(transition_img)
            n_imgs = len(relevant_imgs)
            for i, img in enumerate(relevant_imgs):
                if self.abort_processing:
                    acq.clear_imgs() # Remove partially processed images so reloaded from file without lens correction etc.
                    return
                self.processing_status[acq] = f"{i} / {n_imgs}"
                self.registration.register_image(img, transition_img)
            acq.get_pl_image().create(relevant_imgs, self.config, acq.get_gps_info(), acq.get_gimbal_info(), acq.get_camera_info())
            acq.get_pl_image().save(self.mission.get_analysed_folder())
            self.acq_status[acq].status = self.AcquisitionStatus.Status.YES
            with self.pending_msgs_lock:
                self.pending_msgs.append(f"  Success: {acq.get_name()} ({round(time.time() - start_time, 3)}s) ({len([1 for img in relevant_imgs if img.pl_state == Image.PL_State.HIGH])} HIGH, {len([1 for img in relevant_imgs if img.pl_state == Image.PL_State.LOW])} LOW)")
        except Exception as e:
            print(traceback.format_exc())
            self.acq_status[acq].status = self.AcquisitionStatus.Status.ERROR
            with self.pending_msgs_lock:
                self.pending_msgs.append(f"  Error: {acq.get_name()} - {e} ({round(time.time() - start_time, 3)}s)")
        acq.clear_imgs() # Free up memory

    def clear_mission(self):
        if self.mission is None:
            self.log.appendPlainText("Error: no mission selected")
            return
        if self.running_processing:
            self.log.appendPlainText("Error: processing running")
            return
        for acq in self.mission.get_acquisitions():
            acq.clear_imgs()
            acq.get_pl_image().clear(self.mission.get_analysed_folder())
            self.__update_acquisition_status(acq, recheck=True)
        self.log.appendPlainText(f"Cleared mission {self.mission.get_folder()}")

    def stop_processing(self):
        if not self.running_processing:
            self.log.appendPlainText("Error: can't stop processing, nothing running")
            return
        self.abort_processing = True
        self.thread_pool.shutdown(wait=True, cancel_futures=True)
        self.abort_processing = False
        self.__conclude_processing()
        for acq in self.mission.get_acquisitions():
            self.__update_acquisition_status(acq, recheck=True)
        self.log.appendPlainText(f"Aborted mission {self.mission.get_folder()}")

    def __conclude_processing(self):
        self.running_processing = False
        self.update_status_thread.stop()
        self.__enable_buttons()
        self.__write_pending_msgs() # Write any last messages that got written while shutting down

    def __enable_buttons(self):
        for obj in self.central_widget.findChildren(QPushButton) + self.central_widget.findChildren(QComboBox):
            obj.setEnabled(not self.running_processing)
        self.stop_button.setEnabled(self.running_processing)

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

    def __update_acquisition_status(self, acq: Acquisition, recheck: bool, post_text: str=None):
        acq_status = self.acq_status[acq]
        if recheck:
            acq_status.status = self.AcquisitionStatus.Status[acq.get_pl_image().get_status(self.config).name] # Match names of PL_Image.Status amd AcquisitionStatus.Status
        post_text = " " + post_text if post_text is not None else ""
        status_item = QTableWidgetItem()
        status_item.setBackground(self.status_colours[acq_status.status])
        self.acquisition_table.setItem(acq_status.row, 0, status_item)
        self.acquisition_table.setItem(acq_status.row, 1, QTableWidgetItem(self.status_text[acq_status.status] + post_text))

    def update_lens(self):
        if self.lens_selection.currentText() not in self.lens_correction.get_lens_names():
            self.log.appendPlainText("Error: unknown lens selected")
            return
        if self.running_processing:
            self.log.appendPlainText("Error: processing running")
            return
        self.config.set("lens", self.lens_selection.currentText())
        self.config.write()
        if self.mission is not None:
            for acq in self.mission.get_acquisitions():
                self.__update_acquisition_status(acq, recheck=True)

    def __setup_lens_options(self):
        self.lens_selection.blockSignals(True)
        self.lens_selection.clear()
        self.lens_selection.addItems(self.lens_correction.get_lens_names())
        if self.config.get("lens") in self.lens_correction.get_lens_names():
            self.lens_selection.setCurrentText(self.config.get("lens"))
        self.lens_selection.blockSignals(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
