import sys
import json
import re
from raytracerGUI  import *
from classes import *

def window():
    """
    Initializes the UI using the PyQt5 library.
    """
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



def parser(path):
    """
    Parses the text file containing the scene's objects' properties.
    :param str path: Path to the scene file
    """

    def parse_image(imgNumber, lines, iteration, imageContents):
        """
        Parses image objects.
        :param int imgNumber: Current image object count
        :param list lines: Lines read from the scene file
        :param int iteration: Current lines list iteration
        :param dict imageContents: Dictionary containing objects' properties
        :returns: imgNumber
        :rtype: int
        """
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
            imageContents["Images"].update({imgNumber-1:{"Resolution":{"X":res[IMG_X], "Y":res[IMG_Y]}, "BG_Color":{"Red":colors[IMG_RED], "Green":colors[IMG_GREEN], "Blue":colors[IMG_BLUE]}}})
        except:
            raise Exception("Error parsing image segment")
        return imgNumber

    
    def parse_transform(transformNumber, imageContents, lines, iteration):
        """
        Parses transformation objects.
        :param int transformNumber: Current transformation object count
        :param list lines: Lines read from the scene file
        :param int iteration: Current lines list iteration
        :param dict imageContents: Dictionary containing objects' properties
        :returns: transformNumber
        :rtype: int
        """
        X = 1
        Y = 2
        Z = 3
        R_TYPE = 0
        R_VALUE = 1
        transformNumber += 1
        imageContents["Transformations"].update({transformNumber-1:{}})
        for transLine in lines[iteration:]:
            if "}" in transLine:
                break
            if re.search(r'\sT\s', transLine):
                imageContents["Transformations"][transformNumber-1].update({"Translation":{"X":transLine.strip().split()[X], "Y":transLine.strip().split()[Y], "Z":transLine.strip().split()[Z]}})
            if re.search(r'\sS\s', transLine):
                imageContents["Transformations"][transformNumber-1].update({"Scale":{"X":transLine.strip().split()[X], "Y":transLine.strip().split()[Y], "Z":transLine.strip().split()[Z]}})
            if re.search(r'\sR[a-z]\s', transLine):
                if not "Rotations" in imageContents["Transformations"][transformNumber-1]:
                    imageContents["Transformations"][transformNumber-1].update({"Rotations":{}})
                imageContents["Transformations"][transformNumber-1]["Rotations"][transLine.strip().split()[R_TYPE]] = transLine.strip().split()[R_VALUE]
        return transformNumber


    def parse_material(materialNumber, imageContents, lines, iteration):
        """
        Parses transformation objects.
        :param int materialNumber: Current material object count
        :param list lines: Lines read from the scene file
        :param int iteration: Current lines list iteration
        :param dict imageContents: Dictionary containing objects' properties
        :returns: materialNumber
        :rtype: int
        """
        MAT_R = 0
        MAT_G = 1
        MAT_B = 2
        AMBIENT       = 0
        DIFFUSE       = 1
        SPECULAR      = 2
        REFRACT       = 3
        REFRACT_INDEX = 4
        materialNumber += 1
        matColors = lines[iteration + 2].strip().split()
        matProperties = lines[iteration + 3].strip().split()
        imageContents["Materials"].update({materialNumber-1:{"Color":{"Red":matColors[MAT_R], "Green":matColors[MAT_G], "Blue":matColors[MAT_B]}, "Properties":{"Ambient":matProperties[AMBIENT], "Diffuse":matProperties[DIFFUSE], "Specular":matProperties[SPECULAR], "Refraction":matProperties[REFRACT], "Refraction_Index":matProperties[REFRACT_INDEX]}}}) 
        return materialNumber


    def parse_cameras(cameraNumber, imageContents, lines, iteration):
        """
        Parses camera objects.
        :param int cameraNumber: Current camera object count
        :param list lines: Lines read from the scene file
        :param int iteration: Current lines list iteration
        :param dict imageContents: Dictionary containing objects' properties
        :returns: cameraNumber
        :rtype: int
        """
        CAM_TRAN = 2
        CAM_DIST = 3
        CAM_FOV  = 4
        cameraNumber += 1
        if cameraNumber > 1:
            raise Exception("There may be only one camera")
        transformIndex = lines[iteration + CAM_TRAN].strip()
        cameraDistance = lines[iteration + CAM_DIST].strip()
        fieldOfView    = lines[iteration + CAM_FOV].strip()
        imageContents["Cameras"].update({cameraNumber-1:{"Transformation":transformIndex, "Distance":cameraDistance, "FOV":fieldOfView}})
        return cameraNumber

    
    def parse_lights(lightNumber, imageContents, lines, iteration):
        """
        Parses light objects.
        :param int lightNumber: Current light object count
        :param list lines: Lines read from the scene file
        :param int iteration: Current lines list iteration
        :param dict imageContents: Dictionary containing objects' properties
        :returns: lightNumber
        :rtype: int
        """
        LIGHT_TRAN = 2
        LIGHT_COLOR = 3
        LIGHT_R = 0
        LIGHT_G = 1
        LIGHT_B = 2
        lightNumber += 1
        transformIndex = lines[iteration + LIGHT_TRAN].strip()
        lightColors = lines[iteration + LIGHT_COLOR].strip().split()
        imageContents["Lights"].update({lightNumber-1:{"Transformation":transformIndex, "Color":{"Red":lightColors[LIGHT_R], "Green":lightColors[LIGHT_G], "Blue":lightColors[LIGHT_B]}}}) 
        return lightNumber


    def parse_spheres(sphereNumber, imageContents, lines, iteration):
        """
        Parses sphere objects.
        :param int sphereNumber: Current sphere object count
        :param list lines: Lines read from the scene file
        :param int iteration: Current lines list iteration
        :param dict imageContents: Dictionary containing objects' properties
        :returns: sphereNumber
        :rtype: int
        """
        SPHERE_TRAN = 2
        SPHERE_MAT  = 3
        sphereNumber += 1
        transformIndex = lines[iteration + SPHERE_TRAN].strip()
        matIndex       = lines[iteration + SPHERE_MAT].strip()
        imageContents["Spheres"].update({sphereNumber-1:{"Transformation":transformIndex, "Material":matIndex}})
        return sphereNumber
    

    def parse_boxes(boxNumber, imageContents, lines, iteration):
        """
        Parses box objects.
        :param int boxNumber: Current box object count
        :param list lines: Lines read from the scene file
        :param int iteration: Current lines list iteration
        :param dict imageContents: Dictionary containing objects' properties
        :returns: boxNumber
        :rtype: int
        """
        BOX_TRAN = 2
        BOX_MAT  = 3
        boxNumber += 1
        transformIndex = lines[iteration + BOX_TRAN].strip()
        matIndex       = lines[iteration + BOX_MAT].strip()
        imageContents["Boxes"].update({boxNumber-1:{"Transformation":transformIndex, "Material":matIndex}})
        return boxNumber


    imageContents = {"Images":{}, "Transformations":{}, "Materials":{}, "Cameras":{}, "Lights":{}, "Spheres":{}, "Boxes":{}}

    with open(path) as file:

        imgNumber       = 0
        transformNumber = 0
        materialNumber  = 0
        cameraNumber    = 0
        lightNumber     = 0
        sphereNumber    = 0
        boxNumber       = 0

        lines = file.readlines()

        for iteration, line in enumerate(lines):

            if "Image" in line:
                imgNumber = parse_image(imgNumber, lines, iteration, imageContents)   

            if "Transformation" in line:
                transformNumber = parse_transform(transformNumber, imageContents, lines, iteration)

            if "Material" in line:
                materialNumber = parse_material(materialNumber, imageContents, lines, iteration)

            if "Camera" in line:
                cameraNumber = parse_cameras(cameraNumber, imageContents, lines, iteration)

            if "Light" in line:
                lightNumber = parse_lights(lightNumber, imageContents, lines, iteration)

            if "Sphere" in line:
                sphereNumber = parse_spheres(sphereNumber, imageContents, lines, iteration)

            if "Box" in line:
                boxNumber = parse_boxes(boxNumber, imageContents, lines, iteration)

    with open("temp.txt", "w", encoding="utf-8") as file2:
        file2.write(json.dumps(imageContents, indent=5, separators=(',', ':')))