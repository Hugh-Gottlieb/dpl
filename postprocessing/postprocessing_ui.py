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
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QGridLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QPlainTextEdit,
    QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(342, 250)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setVerticalSpacing(6)
        self.author_text = QLabel(self.centralwidget)
        self.author_text.setObjectName(u"author_text")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.author_text.sizePolicy().hasHeightForWidth())
        self.author_text.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.author_text, 4, 0, 1, 3)

        self.acquisition_table = QTableWidget(self.centralwidget)
        if (self.acquisition_table.columnCount() < 3):
            self.acquisition_table.setColumnCount(3)
        __qtablewidgetitem = QTableWidgetItem()
        self.acquisition_table.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.acquisition_table.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.acquisition_table.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        self.acquisition_table.setObjectName(u"acquisition_table")
        self.acquisition_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.acquisition_table.setAlternatingRowColors(False)
        self.acquisition_table.horizontalHeader().setVisible(True)
        self.acquisition_table.horizontalHeader().setCascadingSectionResizes(False)
        self.acquisition_table.horizontalHeader().setMinimumSectionSize(10)
        self.acquisition_table.horizontalHeader().setDefaultSectionSize(100)
        self.acquisition_table.horizontalHeader().setHighlightSections(False)
        self.acquisition_table.horizontalHeader().setProperty("showSortIndicator", True)
        self.acquisition_table.horizontalHeader().setStretchLastSection(True)
        self.acquisition_table.verticalHeader().setVisible(False)
        self.acquisition_table.verticalHeader().setStretchLastSection(False)

        self.gridLayout.addWidget(self.acquisition_table, 2, 0, 1, 3)

        self.process_button = QPushButton(self.centralwidget)
        self.process_button.setObjectName(u"process_button")

        self.gridLayout.addWidget(self.process_button, 1, 0, 1, 1)

        self.mission_path = QLineEdit(self.centralwidget)
        self.mission_path.setObjectName(u"mission_path")
        self.mission_path.setReadOnly(True)

        self.gridLayout.addWidget(self.mission_path, 0, 1, 1, 2)

        self.clear_button = QPushButton(self.centralwidget)
        self.clear_button.setObjectName(u"clear_button")

        self.gridLayout.addWidget(self.clear_button, 1, 2, 1, 1)

        self.select_button = QPushButton(self.centralwidget)
        self.select_button.setObjectName(u"select_button")

        self.gridLayout.addWidget(self.select_button, 0, 0, 1, 1)

        self.log = QPlainTextEdit(self.centralwidget)
        self.log.setObjectName(u"log")
        self.log.setReadOnly(True)

        self.gridLayout.addWidget(self.log, 3, 0, 1, 3)


        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.select_button.clicked.connect(MainWindow.set_mission)
        self.process_button.clicked.connect(MainWindow.process_mission)
        self.clear_button.clicked.connect(MainWindow.clear_mission)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"DPL Post-processing", None))
        self.author_text.setText(QCoreApplication.translate("MainWindow", u"Developed by Hugh Gottlieb, with support from Gerold Kloos", None))
        ___qtablewidgetitem = self.acquisition_table.horizontalHeaderItem(1)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"Status", None));
        ___qtablewidgetitem1 = self.acquisition_table.horizontalHeaderItem(2)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Name", None));
        self.process_button.setText(QCoreApplication.translate("MainWindow", u"Process", None))
        self.clear_button.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
        self.select_button.setText(QCoreApplication.translate("MainWindow", u"Set Mission", None))
    # retranslateUi

