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

    parser(FILE_PATH)
    pass



if __name__ == "__main__":
    """
    Main function
    """
    #window()
    testes()