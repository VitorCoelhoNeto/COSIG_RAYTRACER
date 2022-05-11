class Color3:
    def __init__(self, red, green, blue):
        self.red = red
        self.green = green
        self.blue = blue
    
    def print_colors(self):
        print(self.red, self.green, self.blue)

def main():
    test = Color3(250, 230, 100)

    test.print_colors()

if __name__ == "__main__":
    main()