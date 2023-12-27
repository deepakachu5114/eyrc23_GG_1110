def scale_coordinates(x, y, old_width, old_height, new_width, new_height):
    new_x = int((x / old_width) * new_width)
    new_y = int((y / old_height) * new_height)
    return new_x, new_y

def main():
    # Input coordinates for the 4 points in the 610x610 image
    old_width = 610
    old_height = 610
    new_width = 700
    new_height = 700

    # List to store the coordinates of 4 points [x, y]
    coordinates = []

    for i in range(4):
        x = float(input(f"Enter X coordinate for Point {chr(65+i)}: "))
        y = float(input(f"Enter Y coordinate for Point {chr(65+i)}: "))
        coordinates.append((x, y))

    # Scale the coordinates to the 700x700 image
    scaled_coordinates = [scale_coordinates(x, y, old_width, old_height, new_width, new_height) for x, y in coordinates]

    # Output the scaled coordinates
    print("Scaled Coordinates in the 700x700 image:")
    for i, (new_x, new_y) in enumerate(scaled_coordinates):
        print(f"Point {chr(65+i)}: ({new_x}, {new_y})")

if __name__ == "__main__":
    main()
