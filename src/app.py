from raytracerGUI  import *
from classes import *
import sys

def window():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



def parser(path):
    with open(path) as file:
        for line in file.readlines():
            pass