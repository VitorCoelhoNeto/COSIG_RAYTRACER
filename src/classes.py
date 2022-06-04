#from this import d
import numpy as np

class Color3:
    """
    Class that represents colors in RGB format
    :param float red: Represents the RGB value red (R). Must be a float between 0.0 and 1.0.
    :param float green:  Represents the RGB value green (G). Must be a float between 0.0 and 1.0.
    :param float blue: Represents the RGB value red (B). Must be a float between 0.0 and 1.0.
    """

    def __init__(self, red, green, blue):
        if not isinstance(red, float) or type(green) != float or type(blue) != float:
            raise TypeError(
                "Color values must be of float type between 0.0 and 1.0")
        # Deprecated
        #if red < 0.0 or red > 1.0 or green < 0.0 or green > 1.0 or blue < 0.0 or blue > 1.0:
        #    raise TypeError("Color values must be between 0.0 and 1.0")
        self.red = red
        self.green = green
        self.blue = blue


    def print_colors(self):
        print(self.red, self.green, self.blue)
    
    
    def check_range(self):
        """
        Checks the range of the color elements and limits their value
        """
        if self.red > 1.0:
            self.red = 1.0
        if self.green > 1.0:
            self.green = 1.0
        if self.blue > 1.0:
            self.blue = 1.0
        if self.red < 0.0:
            self.red = 0.0
        if self.green < 0.0:
            self.green = 0.0
        if self.blue < 0.0:
            self.blue = 0.0



class Vector3:
    """
    Represents a vector with cartesian 3D coordinates (x, y, z)
    :param int/float x: Represents the value of x on the cartesian 3D coordinate model. Must be of type int or float
    :param int/float y: Represents the value of y on the cartesian 3D coordinate model. Must be of type int or float
    :param int/float z: Represents the value of z on the cartesian 3D coordinate model. Must be of type int or float
    """

    def __init__(self, x, y, z):
        if (type(x) != float and type(x) != int) or (type(y) != float and type(y) != int) or (type(z) != float and type(z) != int):
            raise TypeError("Vector coordinates must be of type float or int")
        self.x = x
        self.y = y
        self.z = z

    def print_coordinates(self):
        print(self.x, self.y, self.z)

    # Deprecated
    # def sum_vectors(vec1,vec2):
    #    x = (vec1.x + vec2.x)
    #    y = (vec1.y + vec2.y)
    #    z = (vec1.z + vec2.z)
    #
    #    vec_result=Vector3(x, y, z)
    #    return vec_result
    #
    # def subtract_vectors(vec1,vec2):
    #    x = (vec1.x - vec2.x)
    #    y = (vec1.y - vec2.y)
    #    z = (vec1.z - vec2.z)
    #
    #    vec_result=Vector3(x, y, z)
    #    return vec_result

    def normalize_vector(self):
        """
        Normalizes a vector.
        :param Vector3 self: Vector to be normalized
        :returns: self
        :rtype: Vector3
        """
        mainArray = [self.x, self.y, self.z]
        # normalized x = x/sqrt(x^2 + y^2 + z^2) and the same goes for y and z
        return np.array(mainArray) / np.sqrt(np.sum(np.array(mainArray)**2))

    def calculate_distance(self):
        """
        Calculates the distance of a vector.
        :returns: float
        :rtype: float
        """
        return np.sqrt(pow(self.x, 2) + pow(self.y, 2) + pow(self.z, 2))

    def calculate_scalar_product(self, vector2):
        """
        Calculates the dot product of two vectors.
        :param Vector3 vector2: Vector to be calculated with the original vector.
        :returns: float
        :rtype: float
        """
        mainArray = [self.x, self.y, self.z]
        secondArray = [vector2.x, vector2.y, vector2.z]
        return np.dot(np.array(mainArray), np.array(secondArray))

    def calculate_vectorial_product(self, vector2):
        """
        Calculates the cross product of two vectors.
        :param Vector3 vector2: Vector to be calculated with the original vector.
        :returns: float
        :rtype: float
        """
        mainArray = [self.x, self.y, self.z]
        secondArray = [vector2.x, vector2.y, vector2.z]
        return np.cross(np.array(mainArray), np.array(secondArray))

    def calculate_distance_two(self, vector2):
        """
        Calculates the distance between two vectors.
        :param Vector3 vector2: Other vector from which the distance is going to be calculated, starting in self.
        :returns: float
        :rtype: float
        """
        if not isinstance(vector2, Vector3):
            raise TypeError("vector2 needs to be of type Vector3")
        return np.sqrt(pow((vector2.x - self.x), 2) + pow((vector2.y - self.y), 2) + pow((vector2.z - self.z), 2))

    def __add__(self, vec3):
        """
        Overload of the '+' operation
        :param Vector3 vec3: Vector to be added to the original one
        :return: Vector3(self.x + vec3.x, self.y + vec3.y, self.z + vec3.z)
        :rtype: Vector3
        """
        if isinstance(vec3, Vector3):
            return Vector3(self.x + vec3.x, self.y + vec3.y, self.z + vec3.z)

    def __mul__(self, t):
        """
        Overload of the '*' operation
        :param int/float t: Variable to be multiplied by the vector
        :return: Vector3(self.x * t, self.y * t, self.z * t)
        :rtype: Vector3
        """
        if isinstance(t, float) or isinstance(t, int):
            return Vector3(self.x * t, self.y * t, self.z * t)

    def __sub__(self, vec3):
        """
        Overload of the '-' operation
        :param Vector3 vec3: Vector to be added to the original one
        :return: Vector3(self.x + vec3.x, self.y + vec3.y, self.z + vec3.z)
        :rtype: Vector3
        """
        if isinstance(vec3, Vector3):
            return Vector3(self.x - vec3.x, self.y - vec3.y, self.z - vec3.z)


class Vector4:
    """
    Represents homogeneous points and coordinates
    :param int/float x: Represents the value of x on the homogeneous coordinates. Must be of type int or float
    :param int/float y: Represents the value of y on the homogeneous coordinates. Must be of type int or float
    :param int/float z: Represents the value of z on the homogeneous coordinates. Must be of type int or float
    :param int/float w: Represents the value of w on the homogeneous coordinates. Must be of type int or float
    """

    def __init__(self, x, y, z, w):
        if (type(x) != float and type(x) != int) or (type(y) != float and type(y) != int) or (type(z) != float and type(z) != int) or (type(w) != float and type(w) != int):
            raise TypeError("Vector coordinates must be of type float or int")
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def print_coordinates(self):
        print(self.x, self.y, self.z, self.w)

    def convertVectorHomoToCar(self):
        """
        For a vector, converts Homogeneous coordinates to Cartesian coordinates
        :param Vector4 self: Vector to be converted 
        """
        return Vector3(self.x, self.y, self.z)

    def convertVectorCarToHomo(vec3):
        """
        For a vector, converts Cartesian coordinates to Homogeneous coordinates 
        :param Vector3 vec3: Vector to be converted
        """
        return Vector4(vec3.x, vec3.x, vec3.x, 0)

    #---------------------------   VERIFICAR CONVERSÃO DOS PONTOS ---------------------
       
    def convertPointHomoToCar(self):
        """
        For a point, converts Homogeneous coordinates to Cartesian coordinates
        """
        self.x = self.x/self.w
        self.y = self.y/self.w
        self.z = self.z/self.w

    def convertPointCarToHomo(self):
        """
        For a point, converts Cartesian coordinates to Homogeneous coordinates
        """
        self.x = self.x
        self.y = self.y
        self.z = self.z
        self.w = 1


class Ray:
    """
    Represents light rays with an origin point and a direction
    :param Vector3 origin: Represents the ray's origin point
    :param Vector3 direction: Represents the ray's direction vector
    """

    def __init__(self, origin, direction):
        if type(origin) != Vector3:
            raise TypeError("Origin value must be of Vector3 type")
        if type(direction) != Vector3:
            raise TypeError("direction value must be of Vector3 type")
        self.origin = origin
        self.direction = direction

    def point_at_parameter(self, t):
        """
        ??? -> Ver 'TR_01.pdf' -> pág. 57
        :param int/float t: Variable to be multiplied by the direction vector
        :return: self.origin + self.direction*t
        :rtype: Vector3
        """
        if isinstance(t, float) or isinstance(t, int):
            return self.origin + self.direction*t


    def print_ray_direction(self):
        print("X: ", self.direction.x, " Y: ", self.direction.y, " Z: ", self.direction.z)


    def print_ray_origin(self):
        print("X: ", self.origin.x, " Y: ", self.origin.y, " Z: ", self.origin.z)



class Hit:
    """
    Represents information obtained by the intersection of a light ray and an object.
    :param bool found: Boolean that stores if the ray has struck an object or not
    :param Material material: Intersected object material
    :param Vector3 point: Intersection point
    :param Vector3 normal: Normal to the tangent plane of the surface where the ray intersected the point
    :param float t: Distance from the origin of the cast ray to the intersection point
    :param float t_min: Minimum t distance found until now
    """

    def __init__(self, found, material, point, normal, t, t_min):
        if not isinstance(found, bool) or not isinstance(material, Material) or not isinstance(point, Vector3) or not isinstance(normal, Vector3) or not isinstance(normal, float) or not isinstance(normal, float):
            raise TypeError("Wrong data type(s) on class constructor")
        self.found = found
        self.material = material
        self.point = point
        self.normal = normal
        self.t = t
        self.t_min = t_min


class Transformation:

    # Deprecated
    # def __init__(self, matrix):
    #    if not isinstance(matrix, list):
    #        raise TypeError("Wrong matrix data type, needs to be a list of lists")
    #    if not len(matrix) == 4:
    #        raise TypeError("Wrong matrix, needs to be a 4x4 dimension matrix (list)")
    #    for row in matrix:
    #        if not isinstance(row, list) or len(row) != 4:
    #            raise TypeError("Wrong matrix, needs to be a 4x4 dimension matrix (list)")
    #    for row in matrix:
    #        for column in row:
    #            if not isinstance(column, int) and not isinstance(column, float):
    #                raise TypeError("Wrong matrix, values need to be of type int or float")
    #    self.matrix = matrix

    def __init__(self):
        self.matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def get_matrix(self):
        """
        Returns the matrix associated with the object.
        """
        return self.matrix

    def print_matrix_items(self):
        """
        Helper function that prints the current status of the matrix, value by value (from left to right, top to bottom).
        """
        for row in self.matrix:
            for column in row:
                print(column)

    def translate(self, x: float, y: float, z: float):
        """
        :param float x: x translation value.
        :param float y: y translation value.
        :param float z: z translation value.
        :returns: the matrix with the new values after the translation.
        :rtype: list
        """
        matrix = np.array(self.matrix)
        matrix2 = np.array([[1, 0, 0, x], [0, 1, 0, y],
                           [0, 0, 1, z], [0, 0, 0, 1]])
        #self.matrix = list(matrix.dot(matrix2))
        self.matrix = matrix@matrix2
        return matrix@matrix2 

    # TODO Find a workaround to get the exact values of the sines and cosines
    def rotateX(self, angle: float):
        """
        :param float angle: rotation angle in degrees (function converts to radians).
        :returns: the matrix with the new values after the rotation.
        :rtype: list
        """
        matrix = np.array(self.matrix)
        matrix2 = np.array([
            [1, 0,                         0,                            0],
            [0, np.cos(np.radians(angle)), -(np.sin(np.radians(angle))), 0],
            [0, np.sin(np.radians(angle)), np.cos(np.radians(angle)),    0],
            [0, 0,                         0,                            1]
        ])
        self.matrix = matrix@matrix2
        return matrix@matrix2

    def rotateY(self, angle: float):
        """
        :param float angle: rotation angle in degrees (function converts to radians).
        :returns: the matrix with the new values after the rotation.
        :rtype: list
        """
        matrix = np.array(self.matrix)
        matrix2 = np.array([
            [np.cos(np.radians(angle)),    0, np.sin(np.radians(angle)), 0],
            [0,                            1, 0,                         0],
            [-(np.sin(np.radians(angle))), 0, np.cos(np.radians(angle)), 0],
            [0,                            0, 0,                         1]
        ])
        self.matrix = matrix@matrix2
        return matrix@matrix2

    def rotateZ(self, angle: float):
        """
        :param float angle: rotation angle in degrees (function converts to radians).
        :returns: the matrix with the new values after the rotation.
        :rtype: list
        """
        matrix = np.array(self.matrix)
        matrix2 = np.array([
            [np.cos(np.radians(angle)), -(np.sin(np.radians(angle))), 0, 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)),    0, 0],
            [0,                         0,                            1, 0],
            [0,                         0,                            0, 1]
        ])
        self.matrix = matrix@matrix2
        return matrix@matrix2

    def scale(self, x: float, y: float, z: float):
        """
        :param float x: x scale value.
        :param float y: y scale value.
        :param float z: z scale value.
        :returns: the matrix with the new values after the scale.
        :rtype: list
        """
        matrix = np.array(self.matrix)
        matrix2 = np.array([[x, 0, 0, 0], [0, y, 0, 0],
                           [0, 0, z, 0], [0, 0, 0, 1]])
        self.matrix = matrix@matrix2
        return matrix@matrix2

    def transpose_matrix(self):
        """
        Returns a transposed version of the matrix. (DOES NOT AFFECT THE MATRIX ITSELF)
        """
        return np.array(self.matrix).transpose()

    def inverse_matrix(self):  # TODO
        pass


class Image:
    """
    Image class that contains image properties, namely, resolution and background color.
    :param int resolutionX: Image resolution on the X axis (horizontal resolution).
    :param int resolutionY: Image resolution on the Y axis (vertical resolution).
    :param Color3 backColor: Background color primary components (R, G and B).
    """

    def __init__(self, backColor, resolutionX, resolutionY):
        if not isinstance(backColor, Color3):
            raise TypeError("Wrong color type. Needs to be of type Color3")
        if not isinstance(resolutionX, int) or not isinstance(resolutionY, int):
            raise TypeError("Wrong resolution. 2 integers are required")
        self.resolutionX = resolutionX
        self.resolutionY = resolutionY
        self.backColor = backColor


class Camera:
    """
    Image class that contains camera properties, namely, transformation, distance and field of view (FOV).
    :param Transformation transformation: Transformation index associated with the camera's transformation.
    :param float distance: Distance to the projection plane (the plane Z = 0.0).
    :param float fov: Vertical field of view in degrees.
    """

    def __init__(self, transformation, distance, fov):
        if not isinstance(transformation, Transformation):
            raise TypeError(
                "Transformation needs to be of type Transformation")
        if not isinstance(distance, float):
            raise TypeError("Distance needs to be of type float")
        if not isinstance(fov, float):
            raise TypeError("Field of view needs to be of type float")
        self.transformation = transformation
        self.distance = Vector3(0, 0, distance)
        self.fov = fov


class Light:
    """
    Light class that contains light source properties, namely, transformation and color.
    :param Transformation transformation: Transformation index associated with the light's transformation.
    :param Color3 color: Light color.
    """

    def __init__(self, transformation, color):
        if not isinstance(transformation, Transformation):
            raise TypeError(
                "Transformation needs to be of type Transformation")
        if not isinstance(color, Color3):
            raise TypeError("Color needs to be of type Color3")
        self.transformation = transformation
        self.color = color


class Material:
    """
    Represents materials that will be applied to Object3D objects.
    :param Color3 color: Material color.
    :param float ambient: Material ambient coefficient (0.0 <= ambient <= 1.0).
    :param float diffuse: Material diffuse coefficient (0.0 <= diffuse <= 1.0).
    :param float specular: Material specular coefficient (0.0 <= specular <= 1.0).
    :param float refraction: Material refraction coefficient (0.0 <= refraction <= 1.0).
    :param float refractionIndex: Material refraction index (1.0 <= refractionIndex).
    """

    def __init__(self, color, ambient, diffuse, specular, refraction, refractionIndex):
        if not isinstance(color, Color3):
            raise TypeError("Color needs to be of type Color3")
        if not isinstance(ambient, float) or ambient < 0.0 or ambient > 1.0:
            raise TypeError(
                "Ambient coefficient needs to be of type float (0.0 <= ambient <= 1.0")
        if not isinstance(diffuse, float) or diffuse < 0.0 or diffuse > 1.0:
            raise TypeError(
                "Diffuse coefficient needs to be of type float (0.0 <= diffuse <= 1.0")
        if not isinstance(specular, float) or specular < 0.0 or specular > 1.0:
            raise TypeError(
                "Specular coefficient needs to be of type float (0.0 <= specular <= 1.0")
        if not isinstance(refraction, float) or refraction < 0.0 or refraction > 1.0:
            raise TypeError(
                "Refraction coefficient needs to be of type float (0.0 <= refraction <= 1.0")
        if not isinstance(refractionIndex, float) or refractionIndex < 1.0:
            raise TypeError(
                "RefractionIndex needs to be of type float (1.0 <= refractionIndex")
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.refraction = refraction
        self.refractionIndex = refractionIndex
    
    
    def print_material(self):
        print(self.color.print_colors(), "\n", self.ambient, self.diffuse, self.specular, self.refraction, self.refractionIndex)


class Object3D:
    """
    Base class for 3D objects (Spheres, Boxes and Triangles).
    :param Transformation transformation: Applied object transformation
    :param Material material: Object material
    """

    def __init__(self, transformation, material):
        if not isinstance(transformation, Transformation):
            raise TypeError("Wrong transformation type")
        if not isinstance(material, Material):
            raise TypeError("Wrong material type")
        self.transformation = transformation
        self.material = material
    
    
    def intersect(self, ray: Ray, hit: Hit, tmin: float) -> bool:
        epsilon = 1 * pow(10, -6)
        return True



class Triangle(Object3D):
    """
    Represents a triangle 3D object. Inherits from base super class "Object3D"
    :param Transformation transformation: Applied object transformation
    :param Material material: Object material
    """

    def __init__(self, transformation, material, vertex1, vertex2, vertex3):
        super().__init__(transformation, material)
        if not isinstance(vertex1, Vector3):
            raise TypeError("Vertices must be of type Vector3")
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.vertex3 = vertex3


    def print_vertices(self):
        """
        Prints triangle's vertices coordinates.
        """
        print("Vertex 1: ", self.vertex1.x, self.vertex1.y, self.vertex1.z)
        print("Vertex 2: ", self.vertex2.x, self.vertex2.y, self.vertex2.z)
        print("Vertex 3: ", self.vertex3.x, self.vertex3.y, self.vertex3.z)


    def print_material_properties(self):
        """
        Prints material properties.
        """
        print("Material color: Red:", self.material.color.red, " Green:",
              self.material.color.green, " Blue:",  self.material.color.blue)
        print("Material properties: Ambient:", self.material.ambient,
              " Diffuse:", self.material.diffuse,
              " Specular:", self.material.specular,
              " Refraction:", self.material.refraction,
              " Refraction Index:", self.material.refractionIndex)


    def print_transformation(self):
        """
        Prints triangle's transformation
        """
        print(self.transformation.print_matrix_items())


    def calculate_normal(self):
        """
        Calculates triangle normal vector.
        :returns:
        :rtype: Vector3
        """
        edgeAB = self.vertex1 - self.vertex2
        edgeBC = self.vertex2 - self.vertex3
        normal = edgeAB.calculate_vectorial_product(edgeBC)
        tempVec = Vector3(float(normal[0]), float(normal[1]), float(normal[2]))
        return tempVec.normalize_vector()


class TrianglesMesh(Object3D):
    """
    Represents a triangle mesh 3D object. Inherits from base super class "Object3D"
    :param Transformation transformation: Applied object transformation
    :param Material material: Object material
    """

    def __init__(self, transformation: Transformation, material: Material, triangleList: list):
        super().__init__(transformation, material)
        self.triangleList = triangleList

    
    def print_triangles_vertices(self):
        for triangle in self.triangleList:
            triangle.print_vertices()


class Box(Object3D):
    """
    Represents a box 3D object. Inherits from base super class "Object3D"
    :param Transformation transformation: Applied object transformation
    :param Material material: Object material
    """

    def __init__(self, transformation, material):
        super().__init__(transformation, material)


class Sphere(Object3D):
    """
    Represents a Sphere 3D object. Inherits from base super class "Object3D"
    :param Transformation transformation: Applied object transformation
    :param Material material: Object material
    """

    def __init__(self, transformation, material):
        super().__init__(transformation, material)
