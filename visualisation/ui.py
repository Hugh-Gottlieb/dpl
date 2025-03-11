from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QFileDialog, QGraphicsPixmapItem, QSplitter, QWidget, QSizePolicy
from PySide6.QtCore import Qt
import sys
import numpy as np
import cv2
from PIL import Image as PIL_Image
import os

from dpl_common.config import Config, get_config_path
from dpl_common.mission import Mission
from dpl_common.helpers import tif_to_jpeg

from visualisation.visualisation_ui import Ui_MainWindow

### NOTE!!!!
# The single overview is experimental code, which is not fully integrated into the dpl_common Mission.
# If the feature is kept, and integrated into dpl_common, this code MUST be updated.
# Otherwise, you are violating the wishes of Gerold and Hugh. You've been warned ...

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.showMaximized() # Fullscreen
        self.config = Config(get_config_path(__file__))
        self.mission = None
        self.overview_scene = QGraphicsScene(self)
        self.overview.setScene(self.overview_scene)
        self.display_scene = QGraphicsScene(self)
        self.display.setScene(self.display_scene)
        self.zoom_limit_min, self.zoom_limit_max = None, None
        self.__build_splitter()

    def __build_splitter(self) -> None:
        placeholder = self.findChild(QWidget, "placeholderwidget")
        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self.overview)
        splitter.addWidget(self.display)
        splitter.setSizes([300, 700])

        # Replace placeholder's layout with the splitter
        layout = placeholder.layout()
        layout.addWidget(splitter)

        self.overview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_mission(self):
        options = QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder', '', options=options)
        if not folder_path:
            return
        self.mission_path.setText(folder_path)
        self.mission = Mission(folder_path)
        self.status.setText(f"Loaded mission - {len(self.mission.get_acquisitions())} images")
        self.zoom_limit_min, self.zoom_limit_max = None, None # Reset to None so dont use from previous mission

    # TODO rename this function to show_grid_overview
    def show_overview(self):
        if self.mission is None:
            self.status.setText("Error: No mission selected")
            return
        n_expected = self.cols.value() * self.rows.value()
        n_actual = len(self.mission.get_acquisitions())
        if n_actual != n_expected:
            self.status.setText(f"Error: Mission has {n_actual} images, but layout requires {n_expected}")
            return
        overview_grid = self.__build_overview()
        self.__display_img(overview_grid, scene=self.display_scene, view=self.display)
        self.status.setText(f"Showing grid overview")
        self.show_single_overview()

    def show_single_overview(self) -> None:
        overview_single = self.__build_single_overview()
        if overview_single is not None:
            self.__display_img(overview_single, scene=self.overview_scene, view=self.overview)
            self.status.setText(f"Showing overview")
        else:
            self.overview_scene.clear()

    @staticmethod
    def __read_single_overview_img(root, files, postfix):
        file = [file for file in files if file.endswith("_" + postfix + ".tif")]
        if len(file) == 1:
            path = os.path.join(root, file[0])
            return np.array(PIL_Image.open(path), dtype=np.uint16)
        return None

    def __build_single_overview(self) -> np.ndarray:
        if self.zoom_limit_min is None:
            self.status.setText(f"Error: please display an grid overview before showing a single overview")
            return None
        root = os.path.join(self.mission.get_folder(), "overview")
        if not os.path.exists(root):
            self.status.setText("Error: No overview folder found.")
            return None
        files = os.listdir(root)
        ir_data = self.__read_single_overview_img(root, files, "ir")
        pl_data = self.__read_single_overview_img(root, files, "pl")
        if pl_data is None and ir_data is None:
            self.status.setText("Error: No overview images found.")
            return None
        img_width = pl_data.shape[1] if pl_data is not None else ir_data.shape[1]
        space_data = np.ones((10, img_width, 3), dtype=np.uint8) * 255
        pl_data = self.__tif_to_jpeg(pl_data, self.zoom_limit_min, self.zoom_limit_max) if pl_data is not None else space_data
        ir_data = tif_to_jpeg(ir_data) if ir_data is not None else space_data
        overview_single = np.concatenate((ir_data, space_data, pl_data), axis=0)
        return overview_single

    def toggle_overview(self) -> None:
        if self.overview.isVisible():
            self.overview.hide()
        else:
            self.overview.show()

    # TODO rename this function to __build_grid_overview
    def __build_overview(self):
        space = 10
        y_size, x_size, = self.mission.get_acquisitions()[0].get_pl_image().data.shape
        rows, cols = self.rows.value(), self.cols.value()
        overview_img = np.zeros((space + rows * (y_size + space), space + cols * (x_size + space)), dtype=np.float32)
        overview_img[:,:] = np.nan
        for i, acq in enumerate(self.mission.get_acquisitions()):
            img = acq.get_pl_image().data
            row, col = self.__get_row_col(i, rows, cols)
            cell_x1 = space + col * (x_size + space)
            cell_y1 = space + row * (y_size + space)
            cell_x2 = cell_x1 + x_size
            cell_y2 = cell_y1 + y_size
            overview_img[cell_y1:cell_y2, cell_x1:cell_x2] = img[:,:]
        self.zoom_limit_min = np.nanpercentile(overview_img, self.config.get("clip_percent"))
        self.zoom_limit_max = np.nanpercentile(overview_img, 100 - self.config.get("clip_percent"))
        nan_mask = np.isnan(overview_img)
        overview_img[nan_mask] = 0
        overview_img = self.__tif_to_jpeg(overview_img, self.zoom_limit_min, self.zoom_limit_max)
        overview_img[nan_mask] = [255, 255, 255]
        for i in range(len(self.mission.get_acquisitions())):
            row, col = self.__get_row_col(i, rows, cols)
            cell_x1 = space + col * (x_size + space)
            cell_y1 = space + row * (y_size + space) + y_size - 2 * space
            overview_img = cv2.putText(overview_img, str(i), (cell_x1, cell_y1), cv2.FONT_HERSHEY_SIMPLEX, 5, (50,205,50), 5, cv2.LINE_AA)
        return overview_img

    # New tif to jpeg which uses a set limit, instead of a dynamic percentile
    def __tif_to_jpeg(self, img, limit_min, limit_max):
        img = img.astype(np.float32)
        preview = (img - limit_min) / (limit_max - limit_min)
        preview = (np.clip(preview, 0, 1) * (2**8 - 1)).astype(np.uint8)
        return cv2.applyColorMap(preview, cv2.COLORMAP_INFERNO)

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
        if self.zoom_limit_min is None:
            self.status.setText(f"Error: please display an overview before zooming")
            return
        acq = self.mission.get_acquisitions()[zoom_id]
        self.__display_img(self.__tif_to_jpeg(acq.get_pl_image().data, self.zoom_limit_min, self.zoom_limit_max), scene=self.display_scene, view=self.display)
        self.status.setText(f"Showing zoom of {acq.get_name()}")

    @staticmethod
    def __display_img(img, scene, view):
        height, width, depth = img.shape
        assert (depth == 3)
        bytes_per_line = depth * width
        q_image = QImage(img, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image))
        pixmap_item.setPos(0, 0)
        scene.setSceneRect(0, 0, width, height)
        scene.clear()
        scene.addItem(pixmap_item)
        view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())