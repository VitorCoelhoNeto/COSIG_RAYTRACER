class Color3:
    """
    Class that represents colors in RGB format
    :param float red: Represents the RGB value red (R). Must be a float between 0.0 and 1.0.
    :param float green:  Represents the RGB value green (G). Must be a float between 0.0 and 1.0.
    :param float blue: Represents the RGB value red (B). Must be a float between 0.0 and 1.0.
    """
    def __init__(self, red, green, blue):
        if not isinstance(red, float) or type(green) != float or type(blue) != float:
            raise TypeError("Color values must be of float type between 0.0 and 1.0")
        if red < 0.0 or red > 1.0 or green < 0.0 or green > 1.0 or blue < 0.0 or blue > 1.0:
            raise TypeError("Color values must be between 0.0 and 1.0")
        self.red = red
        self.green = green
        self.blue = blue
    
    def print_colors(self):
        print(self.red, self.green, self.blue)



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

    def sum_vectors(vec1,vec2):
        x = (vec1.x + vec2.x)
        y = (vec1.y + vec2.y)
        z = (vec1.z + vec2.z)

        vec_result=Vector3(x, y, z)
        return vec_result

    def subtract_vectors(vec1,vec2):
        x = (vec1.x - vec2.x)
        y = (vec1.y - vec2.y)
        z = (vec1.z - vec2.z)

        vec_result=Vector3(x, y, z)
        return vec_result
    
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

    def __init__(self, matrix):
        if not isinstance(matrix, list):
            raise TypeError("Wrong matrix data type, needs to be a list of lists")
        if not len(matrix) == 4:
            raise TypeError("Wrong matrix, needs to be a 4x4 dimension matrix (list)")
        for row in matrix:
            if not isinstance(row, list) or len(row) != 4:
                raise TypeError("Wrong matrix, needs to be a 4x4 dimension matrix (list)")
        for row in matrix:
            for column in row:
                if not isinstance(column, int) and not isinstance(column, float):
                    raise TypeError("Wrong matrix, values need to be of type int or float")
        self.matrix = matrix
    
    def get_matrix(self):
        return self.matrix



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
            raise TypeError("Transformation needs to be of type Transformation")
        if not isinstance(distance, float):
            raise TypeError("Distance needs to be of type float")
        if not isinstance(fov, float):
            raise TypeError("Field of view needs to be of type float")
        self.transformation = transformation
        self.distance = distance
        self.fov = fov



class Light:
    """
    Light class that contains light source properties, namely, transformation and color.
    :param Transformation transformation: Transformation index associated with the light's transformation.
    :param Color3 color: Light color.
    """
    def __init__(self, transformation, color):
        if not isinstance(transformation, Transformation):
            raise TypeError("Transformation needs to be of type Transformation")
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
            raise TypeError("Ambient coefficient needs to be of type float (0.0 <= ambient <= 1.0")
        if not isinstance(diffuse, float) or diffuse < 0.0 or diffuse > 1.0:
            raise TypeError("Diffuse coefficient needs to be of type float (0.0 <= diffuse <= 1.0")
        if not isinstance(specular, float) or specular < 0.0 or specular > 1.0:
            raise TypeError("Specular coefficient needs to be of type float (0.0 <= specular <= 1.0")
        if not isinstance(refraction, float) or refraction < 0.0 or refraction > 1.0:
            raise TypeError("Refraction coefficient needs to be of type float (0.0 <= refraction <= 1.0")
        if not isinstance(refractionIndex, float) or refractionIndex < 1.0:
            raise TypeError("RefractionIndex needs to be of type float (1.0 <= refractionIndex")
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.refraction = refraction
        self.refractionIndex = refractionIndex



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



class Triangle(Object3D):
    """
    Represents a triangle 3D object. Inherits from base super class "Object3D"
    :param Transformation transformation: Applied object transformation
    :param Material material: Object material
    """
    def __init__(self, transformation, material):
        super().__init__(transformation, material)



class TrianglesMesh(Object3D):
    """
    Represents a triangle mesh 3D object. Inherits from base super class "Object3D"
    :param Transformation transformation: Applied object transformation
    :param Material material: Object material
    """
    def __init__(self, transformation, material):
        super().__init__(transformation, material)



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