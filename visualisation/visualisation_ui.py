# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'visualisationHOOfDM.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGraphicsView, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(625, 582)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.show_grid_button = QPushButton(self.centralwidget)
        self.show_grid_button.setObjectName(u"show_grid_button")
        font = QFont()
        font.setPointSize(12)
        self.show_grid_button.setFont(font)

        self.horizontalLayout.addWidget(self.show_grid_button)

        self.show_cell_button = QPushButton(self.centralwidget)
        self.show_cell_button.setObjectName(u"show_cell_button")
        self.show_cell_button.setFont(font)

        self.horizontalLayout.addWidget(self.show_cell_button)

        self.show_overview_button = QPushButton(self.centralwidget)
        self.show_overview_button.setObjectName(u"show_overview_button")
        self.show_overview_button.setFont(font)

        self.horizontalLayout.addWidget(self.show_overview_button)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.placeholderwidget = QWidget(self.centralwidget)
        self.placeholderwidget.setObjectName(u"placeholderwidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.placeholderwidget.sizePolicy().hasHeightForWidth())
        self.placeholderwidget.setSizePolicy(sizePolicy)
        self.horizontalLayout_2 = QHBoxLayout(self.placeholderwidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.overview = QGraphicsView(self.placeholderwidget)
        self.overview.setObjectName(u"display_overview")

        self.horizontalLayout_2.addWidget(self.overview)

        self.display = QGraphicsView(self.placeholderwidget)
        self.display.setObjectName(u"display")

        self.horizontalLayout_2.addWidget(self.display)


        self.verticalLayout.addWidget(self.placeholderwidget)

        self.author_text = QLabel(self.centralwidget)
        self.author_text.setObjectName(u"author_text")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.author_text.sizePolicy().hasHeightForWidth())
        self.author_text.setSizePolicy(sizePolicy1)
        self.author_text.setFont(font)

        self.verticalLayout.addWidget(self.author_text)


        self.gridLayout.addLayout(self.verticalLayout, 1, 0, 1, 1)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.zoom_label = QLabel(self.centralwidget)
        self.zoom_label.setObjectName(u"zoom_label")
        self.zoom_label.setFont(font)

        self.gridLayout_2.addWidget(self.zoom_label, 3, 0, 1, 1)

        self.zoom = QSpinBox(self.centralwidget)
        self.zoom.setObjectName(u"zoom")
        self.zoom.setFont(font)
        self.zoom.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_2.addWidget(self.zoom, 3, 1, 1, 1)

        self.rows_label = QLabel(self.centralwidget)
        self.rows_label.setObjectName(u"rows_label")
        self.rows_label.setFont(font)

        self.gridLayout_2.addWidget(self.rows_label, 1, 0, 1, 1)

        self.cols = QSpinBox(self.centralwidget)
        self.cols.setObjectName(u"cols")
        self.cols.setFont(font)
        self.cols.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.cols.setMinimum(1)

        self.gridLayout_2.addWidget(self.cols, 2, 1, 1, 1)

        self.mission_path = QLineEdit(self.centralwidget)
        self.mission_path.setObjectName(u"mission_path")
        self.mission_path.setFont(font)
        self.mission_path.setReadOnly(True)

        self.gridLayout_2.addWidget(self.mission_path, 0, 1, 1, 1)

        self.set_mission_button = QPushButton(self.centralwidget)
        self.set_mission_button.setObjectName(u"set_mission_button")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.set_mission_button.sizePolicy().hasHeightForWidth())
        self.set_mission_button.setSizePolicy(sizePolicy2)
        self.set_mission_button.setFont(font)

        self.gridLayout_2.addWidget(self.set_mission_button, 0, 0, 1, 1)

        self.rows = QSpinBox(self.centralwidget)
        self.rows.setObjectName(u"rows")
        self.rows.setFont(font)
        self.rows.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.rows.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.rows.setMinimum(1)

        self.gridLayout_2.addWidget(self.rows, 1, 1, 1, 1)

        self.cols_label = QLabel(self.centralwidget)
        self.cols_label.setObjectName(u"cols_label")
        self.cols_label.setFont(font)

        self.gridLayout_2.addWidget(self.cols_label, 2, 0, 1, 1)

        self.status = QLabel(self.centralwidget)
        self.status.setObjectName(u"status")
        self.status.setFont(font)

        self.gridLayout_2.addWidget(self.status, 4, 1, 1, 1)

        self.status_label = QLabel(self.centralwidget)
        self.status_label.setObjectName(u"status_label")
        sizePolicy1.setHeightForWidth(self.status_label.sizePolicy().hasHeightForWidth())
        self.status_label.setSizePolicy(sizePolicy1)
        self.status_label.setFont(font)

        self.gridLayout_2.addWidget(self.status_label, 4, 0, 1, 1)

        self.gridLayout_2.setColumnStretch(1, 1)

        self.gridLayout.addLayout(self.gridLayout_2, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.set_mission_button.clicked.connect(MainWindow.set_mission)
        self.show_grid_button.clicked.connect(MainWindow.show_overview)
        self.show_cell_button.clicked.connect(MainWindow.show_cell)
        self.show_overview_button.clicked.connect(MainWindow.toggle_overview)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"DPL Visualisation", None))
        self.show_grid_button.setText(QCoreApplication.translate("MainWindow", u"Show Grid", None))
        self.show_cell_button.setText(QCoreApplication.translate("MainWindow", u"Zoom Cell", None))
        self.show_overview_button.setText(QCoreApplication.translate("MainWindow", u"Toggle Overview", None))
        self.author_text.setText(QCoreApplication.translate("MainWindow", u"Developed by Hugh Gottlieb, with support from Gerold Kloos", None))
        self.zoom_label.setText(QCoreApplication.translate("MainWindow", u"Zoom cell #", None))
        self.rows_label.setText(QCoreApplication.translate("MainWindow", u"Rows", None))
        self.set_mission_button.setText(QCoreApplication.translate("MainWindow", u"Set Mission", None))
        self.cols_label.setText(QCoreApplication.translate("MainWindow", u"Cols", None))
        self.status.setText("")
        self.status_label.setText(QCoreApplication.translate("MainWindow", u"Status", None))
    # retranslateUi

