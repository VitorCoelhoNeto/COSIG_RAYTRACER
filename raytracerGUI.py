# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'raytracerGUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(992, 716)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.start_btn = QtWidgets.QPushButton(self.centralwidget)
        self.start_btn.setGeometry(QtCore.QRect(880, 630, 93, 28))
        self.start_btn.setObjectName("start_btn")
        self.load_btn = QtWidgets.QPushButton(self.centralwidget)
        self.load_btn.setGeometry(QtCore.QRect(310, 420, 93, 28))
        self.load_btn.setObjectName("load_btn")
        self.image_container = QtWidgets.QGraphicsView(self.centralwidget)
        self.image_container.setGeometry(QtCore.QRect(310, 10, 400, 400))
        self.image_container.setObjectName("image_container")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setGeometry(QtCore.QRect(310, 490, 42, 22))
        self.spinBox.setObjectName("spinBox")
        self.exit_btn_2 = QtWidgets.QPushButton(self.centralwidget)
        self.exit_btn_2.setGeometry(QtCore.QRect(770, 630, 93, 28))
        self.exit_btn_2.setObjectName("exit_btn_2")
        self.exit_btn = QtWidgets.QPushButton(self.centralwidget)
        self.exit_btn.setGeometry(QtCore.QRect(40, 630, 93, 28))
        self.exit_btn.setObjectName("exit_btn")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(590, 420, 118, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(310, 530, 81, 20))
        self.checkBox.setObjectName("checkBox")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Raytracer_COSIG"))
        self.start_btn.setText(_translate("MainWindow", "Start"))
        self.load_btn.setText(_translate("MainWindow", "Load"))
        self.exit_btn_2.setText(_translate("MainWindow", "Save Image"))
        self.exit_btn.setText(_translate("MainWindow", "Exit"))
        self.checkBox.setText(_translate("MainWindow", "CheckBox"))

