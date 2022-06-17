from classes import *

class Box(Object3D):
    """
    Represents a box 3D object. Inherits from base super class "Object3D"
    :param Transformation transformation: Applied object transformation
    :param Material material: Object material
    """

    def __init__(self, transformation, material):
        super().__init__(transformation, material)
        self.bounds=[Vector3(-0.5, -0.5, -0.5), Vector3(0.5,0.5,0.5)]

    
    def calculate_normal(self):  # TODO Calculate box normal
        """
        Calculates box's normal on intersection point.
        :returns: Box's normal on intersection point.
        :rtype: Vector3
        """
        #var newV1 = Vector3.Subtract(v1, v0);
        #var newV2 = Vector3.Subtract(v2, v0);
        #var newX = ((newV1.Y * newV2.Z) - (newV1.Z * newV2.Y));
        #var newY = ((newV1.Z * newV2.X) - (newV1.X * newV2.Z));
        #var newZ = ((newV1.X * newV2.Y) - (newV1.Y * newV2.X));
        #var size = Math.Sqrt(Math.Pow(newX, 2) + Math.Pow(newY, 2) + Math.Pow(newZ, 2));
        #return new Vector3
        #{
        #    X = (float)(newX / size),
        #    Y = (float)(newY / size),
        #    Z = (float)(newZ / size)
        #};
        return Vector3(1, 1, 1)

    
    def intersect(self, ray: Ray, hit: Hit, transformList: list) -> bool:
        """
        Checks if the ray hits the object or not.
        :param Ray ray: Current ray being analyzed.
        :param Hit hit: Information about the ray intersection with the current object. If no intersection, hit.found = False, rest is irrelevant.
        :returns: True if ray intersects current object, False if not
        :rtype: bool
        """
        epsilon = 1.0E-5

        # (Step 1) Ray copy for the Box. Transforming the ray
        boxRay = copy.deepcopy(ray)
        boxRay.origin = boxRay.origin.convert_point3_vector4()
        boxRay.origin = transformList[B][INV] * boxRay.origin
        boxRay.origin = boxRay.origin.convert_point4_vector3()
        boxRay.direction = boxRay.direction.convert_vector3_vector4()
        boxRay.direction = transformList[B][INV] * boxRay.direction
        boxRay.direction = boxRay.direction.convert_vector4_vector3()
        boxRay.direction = boxRay.direction.normalize_vector()
        
        tnear = -1000
        tfar = 1000

        minimumCoords = -0.5
        maximumCoords = 0.5
        
        # Ray directions and origins
        Dx = boxRay.direction.x
        Rx = boxRay.origin.x
        Dy = boxRay.direction.y
        Ry = boxRay.origin.y
        Dz = boxRay.direction.z
        Rz = boxRay.origin.z
        
        # Doesn't intersect
        if Dx == 0:
            if Rx < minimumCoords or Rx > maximumCoords:
                return False
        if Dy == 0:
            if Ry < minimumCoords or Ry > maximumCoords:
                return False
        if Dz == 0:
            if Rz < minimumCoords or Rz > maximumCoords:
                return False
        
        # Intersections
        txmin = (-0.5 - Rx)/Dx
        txmax = (0.5 - Rx)/Dx
        tymin = (-0.5 - Ry)/Dy
        tymax = (0.5 - Ry)/Dy
        tzmin = (-0.5 - Rz)/Dz
        tzmax = (0.5 - Rz)/Dz
        
        # Check x axis
        if txmin > txmax: 
            txmin, txmax = [txmax, txmin]
        if txmin > tnear:
            tnear = txmin
        if txmax < tfar:
            tfar = txmax
        if tnear > tfar:
            return False
        if tfar < 0:
            return False

        # Check y axis
        if tymin > tymax: 
            tymin, tymax = [tymax, tymin]
        if tymin > tnear:
            tnear = tymin
        if tymax < tfar:
            tfar = tymax
        if tnear > tfar:
            return False
        if tfar < 0:
            return False

        # Check z axis
        if tzmin > tzmax: 
            tzmin, tzmax = [tzmax, tzmin]
        if tzmin > tnear:
            tnear = tzmin
        if tzmax < tfar:
            tfar = tzmax
        if tnear > tfar:
            return False
        if tfar < 0:
            return False

        intersectionPoint = Vector3(Rx + Dx * tnear, Ry + Dy * tnear, Rz + Dz * tnear)

        # Calculate normal and apply transformation before transforming the intersection point
        normal =  self.calculate_normal()

        # (Step 3) Convert point to homogenoeus coordinates (object coordinates), transform it, and bring it back to world coordinates (cartesian coordinates)     
        intersectionPoint = intersectionPoint.convert_point3_vector4()
        intersectionPoint = transformList[B][FINAL] * intersectionPoint
        intersectionPoint = intersectionPoint.convert_point4_vector3()
        
        # Calculate distance
        distance = intersectionPoint - boxRay.origin
        hit.t = distance.calculate_distance()
        
        # Intersection is verified
        if hit.t >= epsilon and hit.t < hit.t_min:
        
            # Transform normal
            normal = normal.convert_vector3_vector4()
            normal = transformList[B][TRAN] * normal
            normal = normal.convert_vector4_vector3()
            normal = normal.normalize_vector()
        
            hit.found = True
            hit.t_min = hit.t
            hit.material = self.material
            hit.normal = normal
            hit.point = intersectionPoint
            hit.is_floor = False
            return True
        
        return False
    

    def intersect_shadow(self, shadowRay: Ray, shadowHit: Hit, transformList: list) -> bool:
        """
        Checks if the ray hits the object or not.
        :param Ray ray: Current ray being analyzed.
        :param Hit hit: Information about the ray intersection with the current object. If no intersection, hit.found = False, rest is irrelevant.
        :returns: True if ray intersects current object, False if not
        :rtype: bool
        """
        epsilon = 1.0E-5
        
        boxRay = None
        del boxRay
        # (Step 1) Ray copy for the box. Transforming the ray
        boxRay = copy.deepcopy(shadowRay)
        boxRay.direction = boxRay.direction.convert_vector3_vector4()
        boxRay.direction = transformList[B][INV] * boxRay.direction
        boxRay.direction = boxRay.direction.convert_vector4_vector3()
        boxRay.direction = boxRay.direction.normalize_vector()

        boxRay.origin = boxRay.origin.convert_point3_vector4()
        boxRay.origin = transformList[B][INV] * boxRay.origin
        boxRay.origin = boxRay.origin.convert_point4_vector3()
        
        tnear = -1000
        tfar = 1000

        minimumCoords = -0.5
        maximumCoords = 0.5
        
        # Ray directions and origins
        Dx = boxRay.direction.x
        Rx = boxRay.origin.x
        Dy = boxRay.direction.y
        Ry = boxRay.origin.y
        Dz = boxRay.direction.z
        Rz = boxRay.origin.z
        
        # Doesn't intersect
        if Dx == 0:
            if Rx < minimumCoords or Rx > maximumCoords:
                return False
        if Dy == 0:
            if Ry < minimumCoords or Ry > maximumCoords:
                return False
        if Dz == 0:
            if Rz < minimumCoords or Rz > maximumCoords:
                return False
        
        # Intersections
        txmin = (-0.5 - Rx)/Dx
        txmax = (0.5 - Rx)/Dx
        tymin = (-0.5 - Ry)/Dy
        tymax = (0.5 - Ry)/Dy
        tzmin = (-0.5 - Rz)/Dz
        tzmax = (0.5 - Rz)/Dz
        
        # Check x axis
        if txmin > txmax: 
            txmin, txmax = [txmax, txmin]
        if txmin > tnear:
            tnear = txmin
        if txmax < tfar:
            tfar = txmax
        if tnear > tfar:
            return False
        if tfar < 0:
            return False

        # Check y axis
        if tymin > tymax: 
            tymin, tymax = [tymax, tymin]
        if tymin > tnear:
            tnear = tymin
        if tymax < tfar:
            tfar = tymax
        if tnear > tfar:
            return False
        if tfar < 0:
            return False

        # Check z axis
        if tzmin > tzmax: 
            tzmin, tzmax = [tzmax, tzmin]
        if tzmin > tnear:
            tnear = tzmin
        if tzmax < tfar:
            tfar = tzmax
        if tnear > tfar:
            return False
        if tfar < 0:
            return False

        intersectionPoint = Vector3(Rx + Dx * tnear, Ry + Dy * tnear, Rz + Dz * tnear)

        # Calculate normal and apply transformation before transforming the intersection point
        normal =  self.calculate_normal()

        # (Step 3) Convert point to homogenoeus coordinates (object coordinates), transform it, and bring it back to world coordinates (cartesian coordinates)     
        #intersectionPoint = intersectionPoint.convert_point3_vector4()
        #intersectionPoint = transformList[B][FINAL] * intersectionPoint
        #intersectionPoint = intersectionPoint.convert_point4_vector3()
        
        # Calculate distance
        distance = intersectionPoint - boxRay.origin
        shadowHit.t = distance.calculate_distance()
        
        # Intersection is verified
        if shadowHit.t >= epsilon and shadowHit.t < shadowHit.t_min:
        
            # Transform normal
            normal = normal.convert_vector3_vector4()
            normal = transformList[B][TRAN] * normal
            normal = normal.convert_vector4_vector3()
            normal = normal.normalize_vector()
        
            shadowHit.found = True
            shadowHit.t_min = shadowHit.t
            shadowHit.material = self.material
            shadowHit.normal = normal
            shadowHit.point = intersectionPoint
            return True
        
        return False