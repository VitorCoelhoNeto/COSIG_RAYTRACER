

class Color3:
    def __init__(self, red, green, blue):
        self.red = red
        self.green = green
        self.blue = blue
    
    def print_colors(self):
        print(self.red, self.green, self.blue)


class Vector3:
    def __init__(self, x, y, z):
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







