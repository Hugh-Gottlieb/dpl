from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QFileDialog, QGraphicsPixmapItem
from PySide6.QtCore import Qt
import sys
import numpy as np
import cv2

from dpl_common.helpers import tif_to_jpeg
from dpl_common.config import Config, get_config_path
from dpl_common.mission import Mission

from visualisation.visualisation_ui import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.config = Config(get_config_path(__file__))
        self.mission = None
        self.display_scene = QGraphicsScene(self)
        self.display.setScene(self.display_scene)

    def set_mission(self):
        options = QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder', '', options=options)
        if not folder_path:
            return
        self.mission_path.setText(folder_path)
        self.mission = Mission(folder_path)
        self.status.setText(f"Loaded mission - {len(self.mission.get_acquisitions())} images")

    def show_overview(self):
        if self.mission is None:
            self.status.setText("Error: No mission selected")
            return
        n_expected = self.cols.value() * self.rows.value()
        n_actual = len(self.mission.get_acquisitions())
        if n_actual != n_expected:
            self.status.setText(f"Error: Mission has {n_actual} images, but layout requires {n_expected}")
            return
        overview = self.__build_overview()
        self.__display_img(overview)
        self.status.setText(f"Showing overview")

    def __build_overview(self):
        space = 10
        y_size, x_size, = self.mission.get_acquisitions()[0].get_pl_image().data.shape
        rows, cols = self.rows.value(), self.cols.value()
        overview_img = np.zeros((space + rows * (y_size + space), space + cols * (x_size + space)), dtype=np.uint16)
        for i, acq in enumerate(self.mission.get_acquisitions()):
            img = acq.get_pl_image().data
            row, col = self.__get_row_col(i, rows, cols)
            cell_x1 = space + col * (x_size + space)
            cell_y1 = space + row * (y_size + space)
            cell_x2 = cell_x1 + x_size
            cell_y2 = cell_y1 + y_size
            overview_img[cell_y1:cell_y2, cell_x1:cell_x2] = img[:,:]
        overview_img = tif_to_jpeg(overview_img)
        for i in range(len(self.mission.get_acquisitions())):
            row, col = self.__get_row_col(i, rows, cols)
            cell_x1 = space + col * (x_size + space)
            cell_y1 = space + row * (y_size + space) + y_size
            overview_img = cv2.putText(overview_img, str(i), (cell_x1, cell_y1), cv2.FONT_HERSHEY_SIMPLEX, 5, (50,205,50), 5, cv2.LINE_AA)
        return overview_img

    # Snake, starting from bottom left
    def __get_row_col(self, i, rows, cols):
        row = rows - 1 - int(i / cols)
        col = i % cols
        if int(i/cols) % 2 == 1:
            col = cols - col - 1
        return row, col

    def show_cell(self):
        if self.mission is None:
            self.status.setText("Error: No mission selected")
            return
        zoom_id = self.zoom.value()
        if zoom_id >= len(self.mission.get_acquisitions()):
            self.status.setText(f"Error: Mission has no image with ID {zoom_id}")
            return
        acq = self.mission.get_acquisitions()[zoom_id]
        self.__display_img(tif_to_jpeg(acq.get_pl_image().data))
        self.status.setText(f"Showing zoom of {acq.get_name()}")

    def __display_img(self, img):
        height, width, depth = img.shape
        assert (depth == 3)
        bytes_per_line = depth * width
        q_image = QImage(img, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image))
        pixmap_item.setPos(0, 0)
        self.display_scene.setSceneRect(0, 0, width, height)
        self.display_scene.clear()
        self.display_scene.addItem(pixmap_item)
        self.display.fitInView(self.display_scene.sceneRect(), Qt.KeepAspectRatio)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())