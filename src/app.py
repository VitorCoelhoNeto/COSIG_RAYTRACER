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
    

    def parse_triangles(meshNumber, imageContents, lines, iteration):
        """
        Parses box objects.
        :param int meshNumber: Current triangle meshes object count
        :param list lines: Lines read from the scene file
        :param int iteration: Current lines list iteration
        :param dict imageContents: Dictionary containing objects' properties
        :returns: meshNumber
        :rtype: int
        """
        V_X = 0
        V_Y = 1
        V_Z = 2
        TRANSFORM_LINE = iteration + 2
        FIRST_TRIANG_LINE = iteration + 3
        meshNumber += 1
        # Add mesh and transformation
        imageContents["TriangleMeshes"].update({meshNumber-1:{"Transformation":lines[TRANSFORM_LINE].strip(), "Triangles":{}}})

        triangNumber = -1 #So it becomes 1 when we add the first triangle
        for meshLineCount, meshLine in enumerate(lines[FIRST_TRIANG_LINE:]):
            if "}" in meshLine:
                break
            # Add the triangles and their materials
            if len(meshLine.strip().split()) == 1:
                triangNumber += 1
                imageContents["TriangleMeshes"][meshNumber-1]["Triangles"].update({triangNumber:{
                "Material":lines[iteration+3+meshLineCount].strip(),
                "(0,0)":lines[FIRST_TRIANG_LINE+meshLineCount + 1].strip().split()[V_X], "(0,1)":lines[FIRST_TRIANG_LINE+meshLineCount + 1].strip().split()[V_Y], "(0,2)":lines[FIRST_TRIANG_LINE+meshLineCount + 1].strip().split()[V_Z], 
                "(1,0)":lines[FIRST_TRIANG_LINE+meshLineCount + 2].strip().split()[V_X], "(1,1)":lines[FIRST_TRIANG_LINE+meshLineCount + 2].strip().split()[V_Y], "(1,2)":lines[FIRST_TRIANG_LINE+meshLineCount + 2].strip().split()[V_Z], 
                "(2,0)":lines[FIRST_TRIANG_LINE+meshLineCount + 3].strip().split()[V_X], "(2,1)":lines[FIRST_TRIANG_LINE+meshLineCount + 3].strip().split()[V_Y], "(2,2)":lines[FIRST_TRIANG_LINE+meshLineCount + 3].strip().split()[V_Z]}})
        return meshNumber


    imageContents = {"Images":{}, "Transformations":{}, "Materials":{}, "Cameras":{}, "Lights":{}, "Spheres":{}, "Boxes":{}, "TriangleMeshes":{}}

    with open(path) as file:

        imgNumber       = 0
        transformNumber = 0
        materialNumber  = 0
        cameraNumber    = 0
        lightNumber     = 0
        sphereNumber    = 0
        boxNumber       = 0
        meshNumber      = 0

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
            
            if "Triangles" in line:
                meshNumber = parse_triangles(meshNumber, imageContents, lines, iteration)

    return imageContents



def normal_calculation(imageContents, mesh):
    """
    Calculates triangle normals.
    :param dict imageContents: Dictionary containing objects' properties
    :param int mesh: Mesh index (0 is the floor, 1 is the pyramid and 2 is the donut)
    :returns: normalDict
    :rtype: dict
    """
    triangleList = []

    for key, triangle in imageContents["TriangleMeshes"][mesh]["Triangles"].items():
        transformation = Transformation([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]])
        material = Material(Color3(     float(imageContents["Materials"][float(triangle["Material"])]["Color"]["Red"]),
                                        float(imageContents["Materials"][float(triangle["Material"])]["Color"]["Green"]),
                                        float(imageContents["Materials"][float(triangle["Material"])]["Color"]["Blue"])),
                                        float(imageContents["Materials"][float(triangle["Material"])]["Properties"]["Ambient"]),
                                        float(imageContents["Materials"][float(triangle["Material"])]["Properties"]["Diffuse"]),
                                        float(imageContents["Materials"][float(triangle["Material"])]["Properties"]["Specular"]), 
                                        float(imageContents["Materials"][float(triangle["Material"])]["Properties"]["Refraction"]), 
                                        float(imageContents["Materials"][float(triangle["Material"])]["Properties"]["Refraction_Index"]))
        vec1 = Vector3(float(triangle["(0,0)"]), float(triangle["(0,1)"]), float(triangle["(0,2)"]))
        vec2 = Vector3(float(triangle["(1,0)"]), float(triangle["(1,1)"]), float(triangle["(1,2)"]))
        vec3 = Vector3(float(triangle["(2,0)"]), float(triangle["(2,1)"]), float(triangle["(2,2)"]))
        triangleList.append(Triangle(transformation, material, vec1, vec2, vec3))

    normalDict = {}
    for iteration, triangle in enumerate(triangleList):
        triangleNormal = triangle.calculate_normal()
        #print(triangle.print_transformation())
        normalDict[iteration] = triangleNormal
    return normalDict



def generate_scene_objects(imageContents: dict) -> list:
    """
    Generates the objects present on the scene
    :param dict imageContents: Dictionary containing objects' properties
    :returns: List with scene objects -> sceneObjects
    :rtype: list
    """
    sceneObjects = list()

    # Camera
    camTransformation = Transformation()
    camTranslation = imageContents["Transformations"][int(imageContents["Cameras"][0]["Transformation"])]["Translation"]
    camTransformation.translate(float(camTranslation["X"]), float(camTranslation["Y"]), float(camTranslation["Z"]))
    camTransformation.rotateX(float(imageContents["Transformations"][int(imageContents["Cameras"][0]["Transformation"])]["Rotations"]["Rx"]))
    camTransformation.rotateZ(float(imageContents["Transformations"][int(imageContents["Cameras"][0]["Transformation"])]["Rotations"]["Rz"]))
    camDistance = float(imageContents["Cameras"][0]["Distance"])
    camFOV = float(imageContents["Cameras"][0]["FOV"])
    camera = Camera(camTransformation, camDistance, camFOV)
    sceneObjects.append(camera)

    # Image
    imageColor = Color3(float(imageContents["Images"][0]["BG_Color"]["Red"]), 
                        float(imageContents["Images"][0]["BG_Color"]["Green"]), 
                        float(imageContents["Images"][0]["BG_Color"]["Blue"]))
    image = Image(imageColor, int(imageContents["Images"][0]["Resolution"]["X"]), int(imageContents["Images"][0]["Resolution"]["Y"]))
    sceneObjects.append(image)

    return sceneObjects



def trace_rays(ray: Ray, rec: int) -> Color3:
    """
    Traces the rays, recursively following the path the ray takes and returns a color.
    :param Ray ray: Ray to be traced.
    :param int rec: Recursivity level.
    :returns: A Color3 object
    :rtype: Color3
    """
    return Color3(0.4, 0.5, 0.6)



def preliminar_calculations(camera: Camera, image: Image) -> list:
    """
    Preliminar calculations
    :param Camera camera: Scene's camera object.
    :param Image image: Scene's image object.
    :returns: pixelList, a list of colors of each pixel
    :rtype: list
    """
    camFOV = camera.fov
    camDistance = camera.distance.z

    # Convert FOV to radians (originally it's 30 degrees)
    radianCamFOV = (camFOV * np.pi) / 180.0

    # Calculate pixel size (Image is square 200x200)
    height = 2.0 * camDistance * np.tan(radianCamFOV / 2.0)
    width = height * image.resolutionX / image.resolutionY
    pixelSize = width / image.resolutionY

    # Primary rays
    origin  = Vector3(0, 0, camDistance)

    pixelList = list()

    for j in range(image.resolutionY + 1):
        for i in range(image.resolutionX + 1):
            pixelX = (i + 0.5) * pixelSize - width / 2.0
            pixelY = -(j + 0.5) * pixelSize + height / 2.0
            pixelZ = 0
            direction = Vector3(float(pixelX), float(pixelY), -float(camDistance))
            direction = direction.normalize_vector()
            directionVector = Vector3(float(direction[0]), float(direction[1]), float(direction[2]))
            ray = Ray(origin, directionVector)
            rec = 2
            color = trace_rays(ray, rec)
            color.check_range()
            pixelList.append(Color3(float(int(255.0 * color.red)), float(int(255.0 * color.green)), float(int(255.0 * color.blue))))

    return pixelList
