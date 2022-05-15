import sys
import json
import re
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



    def parse_image(imgNumber, lines, iteration, imageContents):
        IMG_X     = 0
        IMG_Y     = 1
        IMG_RED   = 0
        IMG_GREEN = 1
        IMG_BLUE  = 2
        imgNumber += 1
        if imgNumber > 1:
            raise Exception("There may be only one segment of type 'Images'")
        try:
            res = lines[iteration + 2].strip().split()
            colors = lines[iteration + 3].strip().split()
            imageContents["Images"].update({imgNumber:{"Resolution":{"X":res[IMG_X], "Y":res[IMG_Y]}, "BG_Color":{"Red":colors[IMG_RED], "Green":colors[IMG_GREEN], "Blue":colors[IMG_BLUE]}}})
        except:
            raise Exception("Error parsing image segment")
        return imgNumber

    

    def parse_transform(transformNumber, imageContents, lines, iteration):
        transformNumber += 1
        imageContents["Transformations"].update({transformNumber:{}})
        for transLine in lines[iteration:]:
            if "}" in transLine:
                break
            if re.search(r'\sT\s', transLine):
                imageContents["Transformations"][transformNumber].update({"Translation":{"X":transLine.strip().split()[1], "Y":transLine.strip().split()[2], "Z":transLine.strip().split()[3]}})
            if re.search(r'\sS\s', transLine):
                imageContents["Transformations"][transformNumber].update({"Scale":{"X":transLine.strip().split()[1], "Y":transLine.strip().split()[2], "Z":transLine.strip().split()[3]}})
            if re.search(r'\sR[a-z]\s', transLine):
                if not "Rotations" in imageContents["Transformations"][transformNumber]:
                    imageContents["Transformations"][transformNumber].update({"Rotations":{}})
                imageContents["Transformations"][transformNumber]["Rotations"][transLine.strip().split()[0]] = transLine.strip().split()[1]
        return transformNumber



    imageContents = {"Images":{}, "Transformations":{}}

    with open(path) as file:

        imgNumber = 0
        transformNumber = 0

        lines = file.readlines()

        for iteration, line in enumerate(lines):

            if "Image" in line:
                imgNumber = parse_image(imgNumber, lines, iteration, imageContents)   

            if "Transformation" in line:
                transformNumber = parse_transform(transformNumber, imageContents, lines, iteration)

    with open("temp.txt", "w", encoding="utf-8") as file2:
        file2.write(json.dumps(imageContents, indent=5, separators=(',', ':')))