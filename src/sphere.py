from classes import *

class Sphere(Object3D):
    """
    Represents a Sphere 3D object. Inherits from base super class "Object3D"
    :param Transformation transformation: Applied object transformation
    :param Material material: Object material
    """

    def __init__(self, transformation, material):
        super().__init__(transformation, material)

    
    def calculate_normal(self, vec3: Vector3):
        """
        Calculates sphere's normal on intersection point.
        :returns: Sphere's normal in intersection point.
        :rtype: Vector3
        """
        point = copy.deepcopy(vec3)
        scalar = float(np.sqrt(point.calculate_scalar_product(point)))
        normal = Vector3(float(point.x / scalar), float(point.y / scalar), float(point.z / scalar))
        return normal.normalize_vector()
    

    def intersect(self, ray: Ray, hit: Hit, transformList:list) -> bool:
        """
        Calculates if a ray hits a sphere or not.
        :param Ray ray: Current ray being cast.
        :param Hit hit: Class to store hit properties if the ray intersects the sphere
        :returns: Wether the ray intersects the sphere or not.
        :rtype: bool
        """
        epsilon = 1.0E-5

        # Sphere center and radius
        sphereCenter = Vector3(0, 0, 0)
        radius = 1

        # (Step 1) Ray copy for the Sphere. Transforming the ray
        sphereRay = None
        del sphereRay
        sphereRay = copy.deepcopy(ray)
        sphereRay.origin = sphereRay.origin.convert_point3_vector4()
        sphereRay.origin = transformList[S][INV] * sphereRay.origin
        sphereRay.origin = sphereRay.origin.convert_point4_vector3()
        sphereRay.direction = sphereRay.direction.convert_vector3_vector4()
        sphereRay.direction = transformList[S][INV] * sphereRay.direction
        sphereRay.direction = sphereRay.direction.convert_vector4_vector3()
        sphereRay.direction = sphereRay.direction.normalize_vector()
        
        # (Step 2) Calculate point if there is an intersection point between the sphereRay and the current object

        # Distance from ray origin to sphere center
        originToCenter = sphereCenter - sphereRay.origin

        # Dot product between distance from ray origin to sphere center and ray direction
        v = originToCenter.calculate_scalar_product(sphereRay.direction)

        # Discriminant
        discriminant = pow(radius, 2) - (originToCenter.calculate_scalar_product(originToCenter) - (v * v))

        if discriminant < 0:
            # No intersection
            return False
        else:

            # If it intersects, we calculate the intersection point
            d = np.sqrt(discriminant)
            point = sphereRay.origin + (sphereRay.direction * (v - d))
            
            # Calculate normal and apply transformation before transforming the intersection point
            normal = self.calculate_normal(point)

            # (Step 3) Convert point to homogenoeus coordinates (object coordinates), transform it, and bring it back to world coordinates (cartesian coordinates)
            point = point.convert_point3_vector4()
            point = transformList[S][FINAL] * point
            point = point.convert_point4_vector3()

            # Then the distance from the camera to the intersection point
            distance = point - sphereRay.origin
            hit.t = distance.calculate_distance()

            # And check if it is smaller than the lowest distance and also epsilon
            if hit.t >= epsilon and hit.t < hit.t_min:
                # Transform normal
                normal = normal.convert_vector3_vector4()
                normal = transformList[T][TRAN] * normal
                normal = normal.convert_vector4_vector3()
                normal = normal.normalize_vector()

                hit.found = True
                hit.t_min = hit.t
                hit.point = point        
                hit.material = self.material
                hit.normal = normal
                hit.is_floor = False
                return True

            return False


    def intersect_shadow(self, shadowRay: Ray, shadowHit: Hit, transformList:list) -> bool:
        """
        Calculates if a ray hits a sphere or not.
        :param Ray ray: Current ray being cast.
        :param Hit hit: Class to store hit properties if the ray intersects the sphere
        :returns: Wether the ray intersects the sphere or not.
        :rtype: bool
        """
        epsilon = 1 * pow(10, -5)

        # (Step 1) Ray copy for the sphere. Transforming the ray
        sphereRay = None
        del sphereRay
        sphereRay = copy.deepcopy(shadowRay)
        sphereRay.direction = sphereRay.direction.convert_vector3_vector4()
        sphereRay.direction = transformList[S][INV] * sphereRay.direction
        sphereRay.direction = sphereRay.direction.convert_vector4_vector3()
        sphereRay.direction = sphereRay.direction.normalize_vector()

        sphereRay.origin = sphereRay.origin.convert_point3_vector4()
        sphereRay.origin = transformList[S][INV] * sphereRay.origin
        sphereRay.origin = sphereRay.origin.convert_point4_vector3()

        # Sphere center and radius
        sphereCenter = Vector3(0, 0, 0)
        radius = 1
        
        # (Step 2) Calculate point if there is an intersection point between the sphereRay and the current object

        # Distance from ray origin to sphere center
        originToCenter = sphereCenter - sphereRay.origin

        # Dot product between distance from ray origin to sphere center and ray direction
        v = originToCenter.calculate_scalar_product(sphereRay.direction)

        # Discriminant
        discriminant = pow(radius, 2) - (originToCenter.calculate_scalar_product(originToCenter) - (v * v))

        if discriminant < 0:
            # No intersection
            return False
        else:

            # If it intersects, we calculate the intersection point
            d = np.sqrt(discriminant)
            point = sphereRay.origin + (sphereRay.direction * (v - d))
            
            # Calculate normal and apply transformation before transforming the intersection point
            normal = self.calculate_normal(point)

            # (Step 3) Convert point to homogenoeus coordinates (object coordinates), transform it, and bring it back to world coordinates (cartesian coordinates)
            #point = point.convert_point3_vector4()
            #point = transformList[S][FINAL] * point
            #point = point.convert_point4_vector3()

            # Then the distance from the camera to the intersection point
            distance = point - sphereRay.origin
            shadowHit.t = distance.calculate_distance()

            # And check if it is smaller than the lowest distance and also epsilon
            if shadowHit.t >= epsilon and shadowHit.t < shadowHit.t_min:
                # Transform normal
                normal = normal.convert_vector3_vector4()
                normal = transformList[T][TRAN] * normal
                normal = normal.convert_vector4_vector3()
                normal = normal.normalize_vector()

                shadowHit.found = True
                shadowHit.t_min = shadowHit.t
                shadowHit.point = point        
                shadowHit.material = self.material
                shadowHit.normal = normal
                return True

            return False