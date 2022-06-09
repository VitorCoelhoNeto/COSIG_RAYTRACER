#from this import d
#def intersect_old(self, ray: Ray, hit: Hit) -> bool: #Deprecated Sphere intersection
#    """
#    Checks if the ray hits the object or not.
#    :param Ray ray: Current ray being analyzed.
#    :param Hit hit: Information about the ray intersection with the current object. If no intersection, hit.found = False, rest is irrelevant.
#    :returns: True if ray intersects current object, False if not
#    :rtype: bool
#    """
#    epsilon = 1 * pow(10, -6)
#
#    # Transformations
#    sphereTransformation = Transformation()
#    sphereTransformation.translate(0, -24, 0)
#    sphereTransformation.scale(6, 6, 6)
#
#    cameraTransformation = Transformation()
#    #cameraTransformation.translate(0, 0, 10)
#    #cameraTransformation.rotateX(-60)
#    #cameraTransformation.rotateZ(45)
#    #cameraTransformation.transpose_matrix()
#
#
#    # Sphere center
#    sphereCenter = Vector4(0, 0, 0, 1)
#    transformedSphereCenter = sphereTransformation * sphereCenter
#    transformedSphereVector = Vector4(float(transformedSphereCenter[0]), float(transformedSphereCenter[1]), float(transformedSphereCenter[2]), float(transformedSphereCenter[3]))
#
#    # Ray origin
#    origin = Vector4(ray.origin.x, ray.origin.y, ray.origin.z, 1.0)
#    transformedOrigin = cameraTransformation * origin
#    transformedOriginVector = Vector4(float(transformedOrigin[0]), float(transformedOrigin[1]), float(transformedOrigin[2]), float(transformedOrigin[3]))
#
#    # Distance from ray origin to sphere center
#    distanceToCenter = transformedSphereVector - transformedOriginVector
#    distanceToCenter = np.array([distanceToCenter.x, distanceToCenter.y, distanceToCenter.z, 0.0])
#
#    # Ray Direction
#    transformedDirection = cameraTransformation * Vector4(ray.direction.x, ray.direction.y, ray.direction.z, 0.0)
#
#    v = np.dot(distanceToCenter, transformedDirection)
#    discriminant = np.dot(distanceToCenter, distanceToCenter) - v * v
#
#    if discriminant < 1: 
#
#        thc = np.sqrt(1 - discriminant)
#        t0 = v - thc
#        t1 = v + thc
#
#        if t1 < epsilon:
#            t0, t1 = [t1, t0]
#
#            if t0 < epsilon: 
#                return False
#
#        t = t0
#
#        # Check if ray intersects sphere
#        if t >= epsilon and t < hit.t_min:
#            # Intersection point
#            #point = Vector4(float(ray.origin.x + ray.direction.x * t), float(ray.origin.y + ray.direction.y * t), float(ray.origin.z + ray.direction.z * t), 0.0)
#            point = Vector4(float(transformedOriginVector.x + float(transformedDirection[0]) * t), float(transformedOriginVector.y + float(transformedDirection[1]) * t), float(transformedOriginVector.z + float(transformedDirection[2]) * t), 1.0)
#            hit.t = t
#            hit.point = point        
#            hit.t_min = hit.t
#            hit.found = True
#            hit.material = self.material
#            #hit.normal = 
#            return True
#
#    return False


# Deprecated Transformation init
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



# Deprecated Vector 3 functions
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

# Deprecated color init
#if red < 0.0 or red > 1.0 or green < 0.0 or green > 1.0 or blue < 0.0 or blue > 1.0:
#    raise TypeError("Color values must be between 0.0 and 1.0")