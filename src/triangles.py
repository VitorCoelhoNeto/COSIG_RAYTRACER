from classes import *

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
        #tempVec = Vector3(float(normal[0]), float(normal[1]), float(normal[2]))
        return normal.normalize_vector()
    

    def intersect(self, ray: Ray, hit: Hit, transformList: list, isFloor: bool) -> bool:
        """
        Checks if the ray hits the object or not.
        :param Ray ray: Current ray being analyzed.
        :param Hit hit: Information about the ray intersection with the current object. If no intersection, hit.found = False, rest is irrelevant.
        :returns: True if ray intersects current object, False if not
        :rtype: bool
        """
        epsilon = 1.0E-5

        # (Step 1) Ray copy for the triangles. Transforming the ray
        triangleRay = copy.deepcopy(ray)
        triangleRay.origin = triangleRay.origin.convert_point3_vector4()
        triangleRay.origin = transformList[T][INV] * triangleRay.origin
        triangleRay.origin = triangleRay.origin.convert_point4_vector3()
        triangleRay.direction = triangleRay.direction.convert_vector3_vector4()
        triangleRay.direction = transformList[T][INV] * triangleRay.direction
        triangleRay.direction = triangleRay.direction.convert_vector4_vector3()
        triangleRay.direction = triangleRay.direction.normalize_vector()

        # (Step 2) Calculate point if there is an intersection point between the triangleRay and the current object
        beta = (self.calculate_determinant([ [(self.vertex1.x - triangleRay.origin.x),    (self.vertex1.x - self.vertex3.x), triangleRay.direction.x, 0], 
                                             [(self.vertex1.y - triangleRay.origin.y),    (self.vertex1.y - self.vertex3.y), triangleRay.direction.y, 0], 
                                             [(self.vertex1.z - (triangleRay.origin.z )), (self.vertex1.z - self.vertex3.z), triangleRay.direction.z, 0],
                                             [0,                                  0,                                 0,               1] ])) / (self.calculate_determinant([
                                                
                                             [(self.vertex1.x - self.vertex2.x), (self.vertex1.x - self.vertex3.x), triangleRay.direction.x, 0], 
                                             [(self.vertex1.y - self.vertex2.y), (self.vertex1.y - self.vertex3.y), triangleRay.direction.y, 0], 
                                             [(self.vertex1.z - self.vertex2.z), (self.vertex1.z - self.vertex3.z), triangleRay.direction.z, 0],
                                             [0,                                 0,                                 0,               1] ]))

        if beta >= 0:
            gamma = (self.calculate_determinant([ [(self.vertex1.x - self.vertex2.x), (self.vertex1.x - triangleRay.origin.x), triangleRay.direction.x, 0], 
                                                  [(self.vertex1.y - self.vertex2.y), (self.vertex1.y - triangleRay.origin.y), triangleRay.direction.y, 0],
                                                  [(self.vertex1.z - self.vertex2.z), (self.vertex1.z - triangleRay.origin.z), triangleRay.direction.z, 0],
                                                  [0,                                 0,                               0,               1] ])) / (self.calculate_determinant([
                                                  
                                                  [(self.vertex1.x - self.vertex2.x), (self.vertex1.x - self.vertex3.x), triangleRay.direction.x, 0], 
                                                  [(self.vertex1.y - self.vertex2.y), (self.vertex1.y - self.vertex3.y), triangleRay.direction.y, 0], 
                                                  [(self.vertex1.z - self.vertex2.z), (self.vertex1.z - self.vertex3.z), triangleRay.direction.z, 0],
                                                  [0,                                 0,                                 0,               1] ]))
            # alpha = 1.0 - beta - gamma # Not necessary because we already know that if β > 0.0, γ > 0.0 and β + γ < 1.0 the ray intersects the object
            
            # Check if ray intersects
            if beta >= -epsilon and gamma >= -epsilon and (beta + gamma < 1.0 + epsilon):
                point = self.vertex1 + (self.vertex2 - self.vertex1) * beta + (self.vertex3 - self.vertex1) * gamma

                # (Step 3) Convert point to homogenoeus coordinates (object coordinates), transform it, and bring it back to world coordinates (cartesian coordinates)
                # Calculate normal and apply transformation before applying the transformation to the intersection point
                normal =  self.calculate_normal()#Calcular a normal N’ à superfície do objecto i no ponto P’ de intersecção

                point = point.convert_point3_vector4()
                point = transformList[T][FINAL] * point
                point = point.convert_point4_vector3()

                v = point - triangleRay.origin
                hit.t = v.calculate_distance()
                
                if hit.t >= epsilon and hit.t < hit.t_min:
                    
                    # Transform normal
                    normal = normal.convert_vector3_vector4()
                    normal = transformList[T][TRAN] * normal
                    normal = normal.convert_vector4_vector3()
                    normal = normal.normalize_vector()

                    hit.found = True
                    hit.t_min = hit.t
                    hit.material = self.material
                    hit.point = point
                    hit.normal = normal
                    if isFloor:
                        hit.is_floor = True
                    else:
                        hit.is_floor = False
                    return True

        return False

    
    def intersect_shadow(self, shadowRay: Ray, hit: Hit, transformList: list) -> bool:
        """
        Checks if the ray hits the object or not.
        :param Ray ray: Current ray being analyzed.
        :param Hit hit: Information about the ray intersection with the current object. If no intersection, hit.found = False, rest is irrelevant.
        :returns: True if ray intersects current object, False if not
        :rtype: bool
        """
        epsilon = 1 * pow(10, -5)

        # (Step 1) Ray copy for the triangles. Transforming the ray
        triangleRay = None
        del triangleRay
        triangleRay = copy.deepcopy(shadowRay)
        triangleRay.direction = triangleRay.direction.convert_vector3_vector4()
        triangleRay.direction = transformList[T][INV] * triangleRay.direction
        triangleRay.direction = triangleRay.direction.convert_vector4_vector3()
        triangleRay.direction = triangleRay.direction.normalize_vector()

        triangleRay.origin = triangleRay.origin.convert_point3_vector4()
        triangleRay.origin = transformList[T][INV] * triangleRay.origin
        triangleRay.origin = triangleRay.origin.convert_point4_vector3()
    
        # (Step 2) Calculate point if there is an intersection point between the triangleRay and the current object
        beta = (self.calculate_determinant([ [(self.vertex1.x - triangleRay.origin.x),    (self.vertex1.x - self.vertex3.x), triangleRay.direction.x, 0], 
                                             [(self.vertex1.y - triangleRay.origin.y),    (self.vertex1.y - self.vertex3.y), triangleRay.direction.y, 0], 
                                             [(self.vertex1.z - (triangleRay.origin.z )), (self.vertex1.z - self.vertex3.z), triangleRay.direction.z, 0],
                                             [0,                                  0,                                 0,               1] ])) / (self.calculate_determinant([
                                                
                                             [(self.vertex1.x - self.vertex2.x), (self.vertex1.x - self.vertex3.x), triangleRay.direction.x, 0], 
                                             [(self.vertex1.y - self.vertex2.y), (self.vertex1.y - self.vertex3.y), triangleRay.direction.y, 0], 
                                             [(self.vertex1.z - self.vertex2.z), (self.vertex1.z - self.vertex3.z), triangleRay.direction.z, 0],
                                             [0,                                 0,                                 0,               1] ]))
    
        if beta >= 0:
            gamma = (self.calculate_determinant([ [(self.vertex1.x - self.vertex2.x), (self.vertex1.x - triangleRay.origin.x), triangleRay.direction.x, 0], 
                                                  [(self.vertex1.y - self.vertex2.y), (self.vertex1.y - triangleRay.origin.y), triangleRay.direction.y, 0],
                                                  [(self.vertex1.z - self.vertex2.z), (self.vertex1.z - triangleRay.origin.z), triangleRay.direction.z, 0],
                                                  [0,                                 0,                               0,               1] ])) / (self.calculate_determinant([
                                                  
                                                  [(self.vertex1.x - self.vertex2.x), (self.vertex1.x - self.vertex3.x), triangleRay.direction.x, 0], 
                                                  [(self.vertex1.y - self.vertex2.y), (self.vertex1.y - self.vertex3.y), triangleRay.direction.y, 0], 
                                                  [(self.vertex1.z - self.vertex2.z), (self.vertex1.z - self.vertex3.z), triangleRay.direction.z, 0],
                                                  [0,                                 0,                                 0,               1] ]))
            # alpha = 1.0 - beta - gamma # Not necessary because we already know that if β > 0.0, γ > 0.0 and β + γ < 1.0 the ray intersects the object
            
            # Check if ray intersects
            if beta >= -epsilon and gamma >= -epsilon and (beta + gamma < 1.0 + epsilon):
                point = self.vertex1 + (self.vertex2 - self.vertex1) * beta + (self.vertex3 - self.vertex1) * gamma

                # Calculate normal and apply transformation before applying the transformation to the intersection point
                normal =  self.calculate_normal()#Calcular a normal N’ à superfície do objecto i no ponto P’ de intersecção

                # (Step 3) Convert point to homogenoeus coordinates (object coordinates), transform it, and bring it back to world coordinates (cartesian coordinates)
                #point = point.convert_point3_vector4()
                #point = transformList[T][FINAL] * point
                #point = point.convert_point4_vector3()
    
                v = point - triangleRay.origin
                hit.t = v.calculate_distance()
                if hit.t >= epsilon and hit.t < hit.t_min:
                    # Transform normal
                    normal = normal.convert_vector3_vector4()
                    normal = transformList[T][TRAN] * normal
                    normal = normal.convert_vector4_vector3()
                    normal = normal.normalize_vector()
    
                    hit.found = True
                    hit.t_min = hit.t
                    hit.material = self.material
                    hit.point = point
                    hit.normal = normal
                    return True
    
        return False

    
    def calculate_determinant(self, matrix):
        """
        CAlculates a matrix determinant.
        :param list/np.array matrix: Matrix to be determined.
        :returns: The matrix's determinant. 
        :rtype: float
        """
        return np.linalg.det(np.array(matrix))



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