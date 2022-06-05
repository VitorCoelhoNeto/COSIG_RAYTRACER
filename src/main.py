from app import *

FILE_PATH = "..\\test_scene_1.txt"

def testes():
    """
    Testing purposes
    """
    # Color testing
    #testColor = Color3(250, 230, 100)
    #testColor = Color3(0.5, 0.451, 0.4)
    #print(testColor.print_colors())
    #print(testColor.red)

    # Vectors and Rays testing
    #vec1 = Vector3(1.0, 2.0, 3.3)
    #vec2 = Vector3(1, 2, 3)
    #vec3 = Vector3(2, 3, 4)
    #print(vec1.normalize_vector())
    #print(vec1.calculate_distance_two(vec2))
    #print(vec1.calculate_distance())
    #x = vec1 * 3
    #print(x.print_coordinates())
    #print(vec2.calculate_scalar_product(vec3))
    #print(vec2.calculate_vectorial_product(vec3))


    #Vector3.print_coordinates(Vector3.sum_vectors(vec1, vec2))
    #rayTest = Ray(vec1, vec2)
    #print(rayTest.direction.x)
    #x = rayTest.point_at_parameter(0.5)
    #x.print_coordinates()

    # Transformation tests
    #originMatrix = [[0, 0, 0, 0],
    #                [0, 0, 0, 0],
    #                [0, 0, 0, 0],
    #                [0, 0, 0, 0]]
    #
    #matrix = Transformation(originMatrix)
    #
    #print(type(matrix))
    #print(type(matrix.get_matrix()))

    # Test parsing the items
    imageContents = parser(FILE_PATH)
    #with open("temp.json", "w", encoding="utf-8") as file2:
    #    file2.write(json.dumps(imageContents, indent=4, separators=(',', ':')))

    # Test creating the parsed triangles
    #floorTriangleList = get_mesh_triangle_list(imageContents, 0)
    #pyramidTriangleList = get_mesh_triangle_list(imageContents, 1)
    #donutTriangleList = get_mesh_triangle_list(imageContents, 2)
    #normalDict = calculate_mesh_normals(pyramidTriangleList)
    #print(normalDict[2][0])
    #print(normalDict)
    
    #testTransform = Transformation()
    #testTransform.translate(2, 1, 3)
    #testTransform.rotateX(90)
    #testTransform.scale(2,3,2)
    #print(testTransform.get_matrix())
    #print(testTransform.transpose_matrix())
    #testTransform.print_matrix_items()

    # Get sceneObjects and pixelList
    LIST_CAM = 0
    LIST_IMG = 1
    LIST_LIG = 2
    LIST_SPH = 3
    LIST_BOX = 4
    LIST_FLO = 5
    LIST_PYR = 6
    LIST_DON = 7
    sceneObjects = generate_scene_objects(imageContents)
    pixelList, rayList = preliminar_calculations(sceneObjects[LIST_CAM], sceneObjects[LIST_IMG], sceneObjects)

    # Test for mesh triangle's vertices
    #sceneObjects[LIST_PYR].print_triangles_vertices()

    # Test for transformation matrices
    #print(sceneObjects[LIST_CAM].transformation.get_matrix())

    #Test for materials
    #print(sceneObjects[LIST_SPH].material.print_material())

    # Object3D intersection
    #for ray in rayList:
    #    print(ray.print_ray_origin())
    #print(len(rayList))
    #for triangle in sceneObjects[LIST_FLO].triangleList:
    #    print(triangle.print_vertices())
    #sceneObjects[LIST_SPH].intersect(1, 2, 3)

    # Final matrices
    #print(np.matmul(sceneObjects[LIST_CAM].transformation.get_matrix(), sceneObjects[LIST_LIG].transformation.get_matrix()), "\n\n")
    #print(np.matmul(sceneObjects[LIST_CAM].transformation.get_matrix(), sceneObjects[LIST_BOX].transformation.get_matrix()), "\n\n")
    #print(np.matmul(sceneObjects[LIST_CAM].transformation.get_matrix(), sceneObjects[LIST_SPH].transformation.get_matrix()), "\n\n")
    pass



if __name__ == "__main__":
    """
    Main function
    """
    #window()
    testes()
