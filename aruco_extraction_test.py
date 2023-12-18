import cv2
import cv2.aruco as aruco
import numpy as np
from math import sqrt

# Function to detect markers in the provided image
def detect_markers(image):
    # Convert the image to grayscale for marker detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Define the ArUco marker dictionary and parameters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    # Create an ArUco detector object
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    # Detect markers in the grayscale image
    markerCorners, markerIds, rejectedIDs = detector.detectMarkers(gray)
    return markerCorners, markerIds

# Function to calculate the coordinates of the arena based on marker corners
def arena_coordinates(top_left, top_right, bottom_right, bottom_left):
    # Calculate coordinates for each corner of the arena based on marker corners
    # top left - min x -- min y
    # top right -  max x -- min y
    # bottom right - max x -- max y
    # bottom left - min x - max y
    tl_x = min(coord[0] for coord in top_left)
    tl_y = min(coord[1] for coord in top_left)
    tr_x = max(coord[0] for coord in top_right)
    tr_y = min(coord[1] for coord in top_right)
    br_x = max(coord[0] for coord in bottom_right)
    br_y = max(coord[1] for coord in bottom_right)
    bl_x = min(coord[0] for coord in bottom_left)
    bl_y = max(coord[1] for coord in bottom_left)
    # Construct arena coordinates based on the calculated corner values
    arena_coordinates = [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]]
    return arena_coordinates

# Function to perform a perspective transformation on the image
def perspective_transform(arena_coordinates, image):
    """
    NEEDS WORK, DOES NOT TRANSFORM EFFECTIVELY. BUT THE FUNCTION MIGHT NOT BE NEEDED GIVEN THE CAMERA ANGLE IS
    GOOD ENOUGH.

    ASSUMES THE WARPING IS AS SHOWN IN THE IMAGE gg_arena.png, does not work otherwise
    Args:
        arena_coordinates: coordinates of the arena on the image
        image: the warped image

    Returns: straightened image

    """
    source_pts = np.array(arena_coordinates, np.float32)
    aspect_ratio = 1  # Define aspect ratio for transformation
    # Extracting the coordinates of the arena corners
    tl_x, tl_y = arena_coordinates[0]
    tr_x, tr_y = arena_coordinates[1]
    # Calculate width and height of the transformed arena
    transformed_w = sqrt((tl_x - tr_x) ** 2 + (tl_y - tr_y) ** 2)
    transformed_h = transformed_w * aspect_ratio
    # Define destination points for the transformation
    destination_pts = [[tl_x, tl_y], [tl_x + transformed_w, tl_y],
                       [tl_x + transformed_w, tl_y + transformed_h], [tl_x, tl_y + transformed_h]]
    destination_pts = np.array(destination_pts, np.float32)
    # Calculate perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(source_pts, destination_pts)
    # Apply the perspective transformation to the image
    transformed_image = cv2.warpPerspective(image, transform_matrix, (int(transformed_w), int(transformed_h)))
    return transformed_image

# Entry point of the script
def main():
    image_path = 'sample_arenas/arena.png'
    original_image = cv2.imread(image_path)
    # Resize the original image for easier processing
    resized_ori = cv2.resize(original_image, (1000, 1000))
    # Detect markers in the resized image and get marker corner coordinates
    markerCorners, markerIds = detect_markers(resized_ori)
    coordinates = get_corner_coordinates(markerCorners, markerIds)
    print(coordinates)
    # Draw detected markers on the resized image
    cv2.aruco.drawDetectedMarkers(resized_ori, markerCorners, markerIds)
    # Extracting the region based on the marker coordinates
    tl_x, tl_y = coordinates[0]
    br_x, br_y = coordinates[2]
    # Crop the region of interest from the original image
    cropped_region = resized_ori[int(tl_y):int(br_y), int(tl_x):int(br_x)]
    # Save the cropped region as 'cropped_region.png'
    cv2.imwrite('cropped_region.png', cropped_region)
    # Display the original and cropped regions for visualization
    cv2.imshow("original", resized_ori)
    cv2.imshow("Cropped Region", cropped_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to get the corner coordinates from marker corners
def get_corner_coordinates(markerCorners, markerIds):
    corner_list = [[], [], [], []]
    for i in range(len(markerIds)):
        corner_list[0].append(markerCorners[i][0][0])
        corner_list[1].append(markerCorners[i][0][1])
        corner_list[2].append(markerCorners[i][0][2])
        corner_list[3].append(markerCorners[i][0][3])
    coordinates = arena_coordinates(corner_list[0], corner_list[1], corner_list[2], corner_list[3])
    return coordinates

# Call the main function when the script is executed
main()
