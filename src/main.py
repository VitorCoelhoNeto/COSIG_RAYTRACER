from app import *

def main():
    # test = Color3(250, 230, 100)
    # window()
    print("olá main")
    vec1 = Vector3(1, 2, 3)
    vec2 = Vector3(1, 2, 3)

    Vector3.print_coordinates(Vector3.sum_vectors(vec1, vec2)) 

if __name__ == "__main__":
    main()