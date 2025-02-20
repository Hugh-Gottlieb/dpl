# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'postprocessing.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
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
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QComboBox, QGridLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow,
    QPlainTextEdit, QPushButton, QSizePolicy, QTableWidget,
    QTableWidgetItem, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(620, 346)
        self.central_widget = QWidget(MainWindow)
        self.central_widget.setObjectName(u"central_widget")
        self.gridLayout_2 = QGridLayout(self.central_widget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.grid_layout = QGridLayout()
        self.grid_layout.setObjectName(u"grid_layout")
        self.grid_layout.setVerticalSpacing(6)
        self.clear_button = QPushButton(self.central_widget)
        self.clear_button.setObjectName(u"clear_button")
        font = QFont()
        font.setPointSize(12)
        self.clear_button.setFont(font)

        self.grid_layout.addWidget(self.clear_button, 2, 2, 1, 1)

        self.process_button = QPushButton(self.central_widget)
        self.process_button.setObjectName(u"process_button")
        self.process_button.setFont(font)

        self.grid_layout.addWidget(self.process_button, 2, 0, 1, 1)

        self.select_button = QPushButton(self.central_widget)
        self.select_button.setObjectName(u"select_button")
        self.select_button.setFont(font)

        self.grid_layout.addWidget(self.select_button, 0, 0, 1, 1)

        self.log = QPlainTextEdit(self.central_widget)
        self.log.setObjectName(u"log")
        self.log.setMinimumSize(QSize(600, 0))
        self.log.setFont(font)
        self.log.setReadOnly(True)

        self.grid_layout.addWidget(self.log, 4, 0, 1, 3)

        self.acquisition_table = QTableWidget(self.central_widget)
        if (self.acquisition_table.columnCount() < 3):
            self.acquisition_table.setColumnCount(3)
        __qtablewidgetitem = QTableWidgetItem()
        self.acquisition_table.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.acquisition_table.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.acquisition_table.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        self.acquisition_table.setObjectName(u"acquisition_table")
        self.acquisition_table.setFont(font)
        self.acquisition_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.acquisition_table.setAlternatingRowColors(False)
        self.acquisition_table.setShowGrid(True)
        self.acquisition_table.setSortingEnabled(False)
        self.acquisition_table.horizontalHeader().setVisible(True)
        self.acquisition_table.horizontalHeader().setCascadingSectionResizes(False)
        self.acquisition_table.horizontalHeader().setMinimumSectionSize(10)
        self.acquisition_table.horizontalHeader().setDefaultSectionSize(150)
        self.acquisition_table.horizontalHeader().setHighlightSections(False)
        self.acquisition_table.horizontalHeader().setProperty("showSortIndicator", False)
        self.acquisition_table.horizontalHeader().setStretchLastSection(True)
        self.acquisition_table.verticalHeader().setVisible(False)
        self.acquisition_table.verticalHeader().setStretchLastSection(False)

        self.grid_layout.addWidget(self.acquisition_table, 3, 0, 1, 3)

        self.author_text = QLabel(self.central_widget)
        self.author_text.setObjectName(u"author_text")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.author_text.sizePolicy().hasHeightForWidth())
        self.author_text.setSizePolicy(sizePolicy)
        self.author_text.setFont(font)

        self.grid_layout.addWidget(self.author_text, 5, 0, 1, 3)

        self.mission_path = QLineEdit(self.central_widget)
        self.mission_path.setObjectName(u"mission_path")
        self.mission_path.setFont(font)
        self.mission_path.setReadOnly(True)

        self.grid_layout.addWidget(self.mission_path, 0, 1, 1, 2)

        self.lens_label = QLabel(self.central_widget)
        self.lens_label.setObjectName(u"lens_label")
        self.lens_label.setFont(font)
        self.lens_label.setAlignment(Qt.AlignCenter)

        self.grid_layout.addWidget(self.lens_label, 1, 0, 1, 1)

        self.lens_selection = QComboBox(self.central_widget)
        self.lens_selection.setObjectName(u"lens_selection")
        self.lens_selection.setFont(font)

        self.grid_layout.addWidget(self.lens_selection, 1, 1, 1, 2)

        self.stop_button = QPushButton(self.central_widget)
        self.stop_button.setObjectName(u"stop_button")
        self.stop_button.setFont(font)

        self.grid_layout.addWidget(self.stop_button, 2, 1, 1, 1)

        self.grid_layout.setRowStretch(3, 10)

        self.gridLayout_2.addLayout(self.grid_layout, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.central_widget)

        self.retranslateUi(MainWindow)
        self.select_button.clicked.connect(MainWindow.set_mission)
        self.process_button.clicked.connect(MainWindow.process_mission)
        self.clear_button.clicked.connect(MainWindow.clear_mission)
        self.lens_selection.currentTextChanged.connect(MainWindow.update_lens)
        self.stop_button.clicked.connect(MainWindow.stop_processing)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"DPL Post-processing", None))
        self.clear_button.setText(QCoreApplication.translate("MainWindow", u"Delete Analysed Imgs", None))
        self.process_button.setText(QCoreApplication.translate("MainWindow", u"Process", None))
        self.select_button.setText(QCoreApplication.translate("MainWindow", u"Set Mission", None))
        ___qtablewidgetitem = self.acquisition_table.horizontalHeaderItem(1)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"Status", None));
        ___qtablewidgetitem1 = self.acquisition_table.horizontalHeaderItem(2)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Name", None));
        self.author_text.setText(QCoreApplication.translate("MainWindow", u"Developed by Hugh Gottlieb, with support from Gerold Kloos", None))
        self.lens_label.setText(QCoreApplication.translate("MainWindow", u"Lens", None))
        self.stop_button.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
    # retranslateUi

