'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 4A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			GG_1110
# Author List:		Aishini Bhattacharjee, Adithya Ubaradka, Deepak C Nayak, Upasana Nayak
# Filename:			task_4a.py


####################### IMPORT MODULES #######################
import cv2
import numpy as np
import tensorflow as tf
##############################################################



################# ADD UTILITY FUNCTIONS HERE #################

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

def get_corner_coordinates(markerCorners, markerIds):
    corner_list = [[], [], [], []]
    for i in range(len(markerIds)):
        corner_list[0].append(markerCorners[i][0][0])
        corner_list[1].append(markerCorners[i][0][1])
        corner_list[2].append(markerCorners[i][0][2])
        corner_list[3].append(markerCorners[i][0][3])
    coordinates = arena_coordinates(corner_list[0], corner_list[1], corner_list[2], corner_list[3])
    return coordinates

def arena_extraction(image):
    markerCorners, markerIds = detect_markers(image)
    coordinates = get_corner_coordinates(markerCorners, markerIds)
    tl_x, tl_y = coordinates[0]
    br_x, br_y = coordinates[2]
    arena = image[int(tl_y):int(br_y), int(tl_x):int(br_x)]
    return arena

def frame_capture():
    video_capture = cv2.VideoCapture(2)  # 0 for default webcam, change if needed
    frame = None
    # Set the resolution to the maximum supported by the webcam
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height
    #
    # frame_count = 0
    # while frame_count != 20:
    #     ret, frame = video_capture.read()  # Capture a single frame
    #     frame_count += 1
    #
    # # Release the video capture
    # video_capture.release()

    # Path to your video file (or set as 0 for webcam)
    # video_path = '/eyrc23_GG_1110/sample_arenas/gg_test_vid.webm'  # Replace with your video file path, or use 0 for webcam

    # Initialize video capture from the video file or webcam
    # video_capture = cv2.VideoCapture(2)


    frame_count = 0
    while frame_count != 20:
        ret, frame = video_capture.read()  # Capture a single frame
        frame_count += 1

    # Release the video capture
    video_capture.release()

    # frame = cv2.imread("/home/deepakachu/Pictures/Screenshots/Screenshot from 2023-12-21 15-23-35.png")

    return frame

def event_extraction(arena):
    # Convert the resized image to grayscale
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)

    cv2.imshow("grey", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply thresholding to create a mask of the white area
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a directory to save the extracted images
    # output_dir = '../experimetation/extracted_images'
    # os.makedirs(output_dir, exist_ok=True)

    # Define minimum and maximum contour perimeters to filter thick borders
    min_contour_perimeter = 200  # Adjust this based on the border thickness
    max_contour_perimeter = 2000  # Adjust this based on the size of the enclosed images

    events = []
    event_coordinates = []

    # print(contours)

    # Extract and save the enclosed images with thick white borders
    for i, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        if min_contour_perimeter < perimeter < max_contour_perimeter:
            # Find bounding rectangle to get the region of interest
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the region of interest (excluding the border)
            extracted_event = arena[y + 5:y + h - 5, x + 5:x + w - 5]

            # sharpening_kernel = np.array([[-0.2, -0.2, -0.2],
            #                               [-0.2, 2.5, -0.2],
            #                               [-0.2, -0.2, -0.2]])
            #
            # sharpened_extracted_image = cv2.filter2D(extracted_event, -1, sharpening_kernel)

            # Resize the denoised extracted image to 224x224
            extracted_resized = cv2.resize(extracted_event, (224, 224))
            events.append(extracted_resized)
            event_coordinates.append([x, y, w, h])
    print(events)
    return events, event_coordinates

def event_label_prediction(model, events):
    predicted_labels = []
    for event in events:
        prediction_arr = model.predict(np.array([event]), verbose=0)
        predicted_index = np.where(prediction_arr[0] == max(prediction_arr[0]))[0][0]
        if predicted_index == 0:
            predicted_label = "combat"
        elif predicted_index == 1:
            predicted_label = 'human_aid_rehabilitation'
        elif predicted_index == 2:
            predicted_label = 'fire'
        elif predicted_index == 3:
            predicted_label = 'military_vehicles'
        else:
            predicted_label = 'destroyed_buildings'
        predicted_labels.append(predicted_label)
    return predicted_labels

def bounding_box(arena, event_coordinates, predicted_labels):
    for i, coordinate in enumerate(event_coordinates):
        x, y, w, h = coordinate
        cv2.rectangle(arena, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(arena, predicted_labels[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return arena


def event_mapping(event_coordinates, predicted_labels):
    event_mapping = {}
    # Sort by y values
    sorted_coordinates = sorted(event_coordinates, key=lambda item: item[1])
    # first element corresponds to E, and the last one to A
    event_mapping["E"] = predicted_labels[np.bincount(np.where(np.array(event_coordinates) == sorted_coordinates[0])[0]).argmax()]
    event_mapping["A"] = predicted_labels[np.bincount(np.where(np.array(event_coordinates) == sorted_coordinates[-1])[0]).argmax()]
    # Remove the first and last elements from sorted_coordinates
    sorted_coordinates = sorted_coordinates[1:-1]
    # sort by  x values
    sorted_coordinates = sorted(sorted_coordinates, key=lambda item: item[0])
    # first element corresponds to D, and the last one to B
    event_mapping["D"] = predicted_labels[np.bincount(np.where(np.array(event_coordinates) == sorted_coordinates[0])[0]).argmax()]
    event_mapping["C"] = predicted_labels[np.bincount(np.where(np.array(event_coordinates) == sorted_coordinates[-1])[0]).argmax()]
    # The remaining element corresponds to C
    event_mapping["B"] = predicted_labels[np.bincount(np.where(np.array(event_coordinates) == sorted_coordinates[1])[0]).argmax()]

    sorted_keys = sorted(event_mapping.keys())  # Sort the keys alphabetically
    sorted_dict = {key: event_mapping[key] for key in sorted_keys}  # Create a new dictionary with sorted keys
    return sorted_dict

def return_4a_helper():
    model = tf.keras.models.load_model('/home/deepakachu/Desktop/eyantra_stage_2/task2b/saved_models/model_2b_revamped')
    captured_frame = frame_capture()
    arena = arena_extraction(captured_frame)
    cv2.imshow("gg", cv2.resize(arena, (900,900)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    events, event_coordinates = event_extraction(cv2.resize(arena, (900,900)))
    print(events)
    predicted_labels = event_label_prediction(model, events)
    identified_labels = event_mapping(event_coordinates, predicted_labels)

    return identified_labels, arena, event_coordinates, predicted_labels
##############################################################


def task_4a_return():
    """
    Purpose:
    ---
    Only for returning the final dictionary variable
    ---
    Arguments: None
    Returns:
    `identified_labels` : { dictionary }
        dictionary containing the labels of the events detected
    """
##############	ADD YOUR CODE HERE	##############
    identified_labels = return_4a_helper()[0]
##################################################
    return identified_labels


###############	Main Function	#################
if __name__ == "__main__":
    _, arena, event_coordinates, predicted_labels = return_4a_helper()
    arena_labelled = bounding_box(arena, event_coordinates, predicted_labels)
    arena_labelled = cv2.resize(arena_labelled, (900, 900))
    identified_labels = task_4a_return()
    print(identified_labels)
    cv2.imshow("Arena Feed", arena_labelled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()