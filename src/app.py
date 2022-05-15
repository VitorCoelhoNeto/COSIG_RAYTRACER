import sys
import json
from raytracerGUI  import *
from classes import *

def window():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



def parser(path):

    imageContents = {"Images":{}}

    with open(path) as file:

        imgNumber = 0

        IMG_X     = 0
        IMG_Y     = 1
        IMG_RED   = 0
        IMG_GREEN = 1
        IMG_BLUE  = 2
        
        lines = file.readlines()

        for iteration, line in enumerate(lines):

            if "Image" in line:
                imgNumber += 1
                if imgNumber > 1:
                    raise Exception("There may be only one segment of type 'Images'")
                res = lines[iteration + 2].strip().split()
                colors = lines[iteration + 3].strip().split()
                imageContents["Images"].update({imgNumber:{"Resolution":{"X":res[IMG_X], "Y":res[IMG_Y]}, "BG_Color":{"Red":colors[IMG_RED], "Green":colors[IMG_GREEN], "Blue":colors[IMG_BLUE]}}})
    
    with open("temp.txt", "w", encoding="utf-8") as file2:
        file2.write(json.dumps(imageContents, indent=5, separators=(',', ':')))