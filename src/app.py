import sys
import copy
from tqdm import tqdm
from raytracerGUI  import *
from sphere import *
from box import *
from triangles import *
from parser import *

C = 0
S = 1
B = 2
T = 3
L = 4
FINAL= 0
INV  = 1
TRAN = 2

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



def get_mesh_triangle_list(imageContents, mesh):
    """
    Gets the triangles of a specific mesh into a triangle list
    :param dict imageContents: Dictionary containing image objects
    :para int mesh: Mesh index  (0 for floor, 1 for pyramid, 2 for donut)
    :returns: triangleList
    :rtype: dict
    """
    triangleList = []

    for key, triangle in imageContents["TriangleMeshes"][mesh]["Triangles"].items():
        transformation = Transformation() #Because all triangles have transformation 0 and it is the identity matrix, no more calculations are needed
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

    return triangleList



def calculate_mesh_normals(triangleList):
    """
    Calculates triangle normals.
    :param dict triangleList: Dictionary containing triangles
    :returns: normalDict
    :rtype: dict
    """
    normalDict = {}
    for iteration, triangle in enumerate(triangleList):
        triangleNormal = triangle.calculate_normal()
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

    # Light source
    lightTransformation = Transformation()
    lightTranslation = imageContents["Transformations"][int(imageContents["Lights"][0]["Transformation"])]["Translation"]
    lightTransformation.translate(float(lightTranslation["X"]), float(lightTranslation["Y"]), float(lightTranslation["Z"]))
    lightColor = Color3(float(imageContents["Lights"][0]["Color"]["Red"]), float(imageContents["Lights"][0]["Color"]["Green"]), float(imageContents["Lights"][0]["Color"]["Blue"]))
    light = Light(lightTransformation, lightColor)
    sceneObjects.append([light])

    # Sphere
    sphereTransformation = Transformation()
    sphereTranslation = imageContents["Transformations"][int(imageContents["Spheres"][0]["Transformation"])]["Translation"]
    sphereTransformation.translate(float(sphereTranslation["X"]), float(sphereTranslation["Y"]), float(sphereTranslation["Z"]))
    sphereScale = imageContents["Transformations"][int(imageContents["Spheres"][0]["Transformation"])]["Scale"]
    sphereTransformation.scale(float(sphereScale["X"]), float(sphereScale["Y"]), float(sphereScale["Z"]))
    sphereMatColor = imageContents["Materials"][int(imageContents["Spheres"][0]["Material"])]["Color"]
    sphereColor = Color3(float(sphereMatColor["Red"]), float(sphereMatColor["Green"]), float(sphereMatColor["Blue"]))
    sphereMatProps = imageContents["Materials"][int(imageContents["Spheres"][0]["Material"])]["Properties"]
    sphereMat = Material(sphereColor, 
                        float(sphereMatProps["Ambient"]), 
                        float(sphereMatProps["Diffuse"]), 
                        float(sphereMatProps["Specular"]), 
                        float(sphereMatProps["Refraction"]),
                        float(sphereMatProps["Refraction_Index"]))
    sphere = Sphere(sphereTransformation, sphereMat)
    sceneObjects.append(sphere)

    # Box
    boxTransformation = Transformation()
    boxTranslation = imageContents["Transformations"][int(imageContents["Boxes"][0]["Transformation"])]["Translation"]
    boxTransformation.translate(float(boxTranslation["X"]), float(boxTranslation["Y"]), float(boxTranslation["Z"]))
    boxScale = imageContents["Transformations"][int(imageContents["Boxes"][0]["Transformation"])]["Scale"]
    boxTransformation.scale(float(boxScale["X"]), float(boxScale["Y"]), float(boxScale["Z"]))
    boxMatColor = imageContents["Materials"][int(imageContents["Boxes"][0]["Material"])]["Color"]
    boxColor = Color3(float(boxMatColor["Red"]), float(boxMatColor["Green"]), float(boxMatColor["Blue"]))
    boxMatProps = imageContents["Materials"][int(imageContents["Boxes"][0]["Material"])]["Properties"]
    boxMat = Material(boxColor, 
                        float(boxMatProps["Ambient"]), 
                        float(boxMatProps["Diffuse"]), 
                        float(boxMatProps["Specular"]), 
                        float(boxMatProps["Refraction"]),
                        float(boxMatProps["Refraction_Index"]))
    box = Box(boxTransformation, boxMat)
    sceneObjects.append(box)

    # Meshes, appliable to all three meshes
    meshMat = Material(Color3(0.0, 0.0, 0.0), 0.5, 0.5, 0.5, 0.5, 1.5) # Dummy material as it is irrelevant because each triangle has its own material
    meshTransformation = Transformation() # All meshes have transformation 0, which is equivalent to the identity matrix

    # Pyramid mesh
    pyramidTriangleList = get_mesh_triangle_list(imageContents, 1)
    pyramidMesh = TrianglesMesh(meshTransformation, meshMat, pyramidTriangleList)
    sceneObjects.append(pyramidMesh)

    # Donut mesh
    donutTriangleList = get_mesh_triangle_list(imageContents, 2)
    donutMesh = TrianglesMesh(meshTransformation, meshMat, donutTriangleList)
    sceneObjects.append(donutMesh)

    # Floor mesh
    floorTriangleList = get_mesh_triangle_list(imageContents, 0)
    floorMesh = TrianglesMesh(meshTransformation, meshMat, floorTriangleList)
    sceneObjects.append(floorMesh)

    return sceneObjects



def trace_rays(ray: Ray, rec: int, sceneObjects: list, transformList: list) -> Color3:
    """
    Traces the rays, recursively following the path the ray takes and returns a color based on the object intersection.
    :param Ray ray: Ray to be traced.
    :param int rec: Recursivity level.
    :returns: A Color3 object
    :rtype: Color3
    """

    hit = Hit(False, Material(Color3(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 1.0), Vector3(0, 0, 0), Vector3(0, 0, 0), 0.0, float(1 * pow(10, 12)))
    for objecto in sceneObjects: # TODO add them all together
        if isinstance(objecto, TrianglesMesh):
            if len(objecto.triangleList) == 128:
                for triangle in objecto.triangleList:
                    triangle.intersect(ray, hit, transformList, True)
                pass
            else:
                if len(objecto.triangleList) == 6: # or len(objecto.triangleList) == 512:
                    for triangle in objecto.triangleList:
                        triangle.intersect(ray, hit, transformList, False)
                pass
            pass
        if isinstance(objecto, Box):
            objecto.intersect(ray, hit, transformList)
            pass
        if isinstance(objecto, Sphere):
            objecto.intersect(ray, hit, transformList)
            pass

    if hit.found:
        color = Color3(0.0, 0.0, 0.0)
        # Go through all scene lights
        for light in sceneObjects[2]:
            # Calculate color with light interference
            color = color + (light.color * hit.material.ambientColor)

            # Calculate distance from intersection point and light source
            lightPosition = Vector3(0, 0, 0)
            lightPosition = lightPosition.convert_point3_vector4()
            lightPosition = transformList[L][FINAL] * lightPosition
            lightPosition = lightPosition.convert_point4_vector3()

            l = lightPosition - hit.point
            lLength = l.calculate_distance()
            l = l.normalize_vector()

            l2 = copy.deepcopy(l)
            pointCopy = copy.deepcopy(hit.point)

            # Calculate light incidence angle
            cosTheta = hit.normal.calculate_scalar_product(l)

            # If light is being incided on the object's normal (less than 90º), add the diffuse coefficient
            if cosTheta > 0.0:
                shadowRay = Ray(pointCopy, l2)
                shadowHit = Hit(False, Material(Color3(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 1.0), Vector3(0, 0, 0), Vector3(0, 0, 0), 0.0, float(1 * pow(10, 12)))
                shadowHit.t_min = lLength

                for item in sceneObjects:
                    if isinstance(item, TrianglesMesh):
                        if len(item.triangleList) == 6: # or len(item.triangleList) == 512: # TODO Add them all together
                            for triangle2 in item.triangleList:
                                triangle2.intersect_shadow(shadowRay, shadowHit, transformList)
                        pass
                    if isinstance(item, Box):
                        item.intersect_shadow(shadowRay, shadowHit, transformList)
                        pass
                    if isinstance(item, Sphere):
                        item.intersect_shadow(shadowRay, shadowHit, transformList)
                        pass

                    if shadowHit.found:
                        break
                    
                if not shadowHit.found or not hit.is_floor:
                    color = color + ((light.color * hit.material.diffuseColor) * cosTheta)
        
        # Refraction and reflection
        #if rec > 0:
        #    rec = rec - 1
        #    cosThetaV = -(float(copy.deepcopy(ray.direction.calculate_scalar_product(hit.normal))))
        #
        #    # Reflection
        #    if float(hit.material.specular) > 0.0:
        #        
        #        r = copy.deepcopy(ray.direction) + (copy.deepcopy(hit.normal) * (2.0 * cosThetaV))
        #        r = r.normalize_vector()
        #    
        #        reflectedOrigin = copy.deepcopy(hit.point) + (copy.deepcopy(r) * 1.0E-6)
        #    
        #        reflectedRayTemp = Ray(reflectedOrigin, r)
        #        reflectedRay = copy.deepcopy(reflectedRayTemp)
        #        
        #        color = color + (hit.material.specularColor * trace_rays(reflectedRay, rec, sceneObjects, transformList))
        #    
        #    # Refraction
        #    if float(hit.material.refraction) > 0.0:
        #        eta = 1.0 / hit.material.refractionIndex
        #        cosThetaR = np.sqrt(1.0 - eta * eta * (1.0 - cosThetaV * cosThetaV))
        #
        #        if cosThetaV < 0.0:
        #            eta = copy.deepcopy(hit.material.refractionIndex)
        #            cosThetaR = -cosThetaR
        #
        #        direction2 = (copy.deepcopy(ray.direction) * eta) + (hit.normal * ((eta * cosThetaV) - cosThetaR))
        #        direction2 = direction2.normalize_vector()
        #
        #        refractedRay = Ray(copy.deepcopy(hit.point), direction2)
        #        color = color + (hit.material.refractionColor * trace_rays(refractedRay, rec, sceneObjects, transformList))

        # If the ray intersects an object, paint the pixel with the nearest scene object material color with the light interference
        return color / len(sceneObjects[2]) 
    else:
        # If the ray hits no scene objects, paint the pixel with the background color (Black)
        return Color3(0.2, 0.2, 0.2)



def preliminar_calculations(camera: Camera, image: Image, sceneObjects: list) -> list:
    """
    Preliminar calculations
    :param Camera camera: Scene's camera object.
    :param Image image: Scene's image object.
    :param list sceneObjects: List with all of the scene objects, including camera, light and image.
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
    #rayList = list()

    # Transformations
    # Camera
    cameraTransformation = Transformation()
    cameraTransformation.translate(0, 0, -74)
    cameraTransformation.rotateX(-60)
    cameraTransformation.rotateZ(45)
    # Sphere
    sphereTransformation = Transformation()
    sphereTransformation.translate(0, -24, 0)
    sphereTransformation.scale(6, 6, 6)
    finalSphereTransformation = cameraTransformation * sphereTransformation
    inversefinalSphereTransformation = copy.deepcopy(finalSphereTransformation)
    inversefinalSphereTransformation.inverse_matrix()
    transposedFinalSphereTransformation = copy.deepcopy(inversefinalSphereTransformation)
    transposedFinalSphereTransformation.transpose_matrix()
    # Box
    boxTransformation = Transformation()
    boxTransformation.translate(24, 0, 0)
    boxTransformation.scale(12, 12, 12)
    finalBoxTransformation = cameraTransformation * boxTransformation
    inversefinalBoxTransformation = copy.deepcopy(finalBoxTransformation)
    inversefinalBoxTransformation.inverse_matrix()
    transposedFinalBoxTransformation = copy.deepcopy(inversefinalBoxTransformation)
    transposedFinalBoxTransformation.transpose_matrix()
    # Triangles
    trianglesTransformation = Transformation()
    finalTrianglesTransformation = cameraTransformation * trianglesTransformation
    inversefinalTrianglesTransformation = copy.deepcopy(finalTrianglesTransformation)
    inversefinalTrianglesTransformation.inverse_matrix()
    transposedFinalTrianglesTransformation = copy.deepcopy(inversefinalTrianglesTransformation)
    transposedFinalTrianglesTransformation.transpose_matrix()
    # Lights
    lightsTransformation = Transformation()
    lightsTransformation.translate(-7, 5, 66)
    finalLightsTransformation = cameraTransformation * lightsTransformation
    inversefinalLightsTransformation = copy.deepcopy(finalLightsTransformation)
    inversefinalLightsTransformation.inverse_matrix()
    transposedFinalLightsTransformation = copy.deepcopy(inversefinalLightsTransformation)
    transposedFinalLightsTransformation.transpose_matrix()

    transformList = list()
    transformList.append([cameraTransformation])
    transformList.append([finalSphereTransformation, inversefinalSphereTransformation, transposedFinalSphereTransformation])
    transformList.append([finalBoxTransformation, inversefinalBoxTransformation, transposedFinalBoxTransformation])
    transformList.append([finalTrianglesTransformation, inversefinalTrianglesTransformation, transposedFinalTrianglesTransformation])
    transformList.append([finalLightsTransformation, inversefinalLightsTransformation, transposedFinalLightsTransformation])

    # For each pixel in the image, generate a ray from the camera to the back of the scene to check if the ray intersects with any scene objects.
    # If it does, return the color of the intersection. With that list of colors (40k), an image will be generated with the calculated colors.

    for j in tqdm(range(image.resolutionY)):
        for i in range(image.resolutionX):
            pixelX = (i + 0.5) * pixelSize - width / 2.0
            pixelY = -(j + 0.5) * pixelSize + height / 2.0
            pixelZ = 0
            direction = Vector3(float(pixelX), float(pixelY), -float(camDistance))
            direction = direction.normalize_vector()
            ray = Ray(origin, direction)
            rec = 2
            color = trace_rays(ray, rec, sceneObjects, transformList)
            color.check_range()
            #rayList.append(ray)
            pixelList.append(Color3(float(int(255.0 * color.red)), float(int(255.0 * color.green)), float(int(255.0 * color.blue))))

    return pixelList #, rayList
