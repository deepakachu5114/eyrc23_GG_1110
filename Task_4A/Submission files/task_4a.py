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
import os
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

    frame_count = 0
    while frame_count != 20:
        ret, frame = video_capture.read()  # Capture a single frame
        frame_count += 1

    # Release the video capture
    video_capture.release()
    return frame

def event_extraction(arena):
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(2, 2))
    clahe_image = clahe.apply(gray)
    _, thresh = cv2.threshold(clahe_image
                              , 230, 255, cv2.THRESH_BINARY)

    # cv2.imshow("clahe", cv2.resize(clahe_image, (800,800)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a directory to save the extracted images

    output_dir = '/home/deepakachu/Desktop/eyantra_stage_2/eyrc23_GG_1110/Task_4A/Submission files/extracted_images'
    os.makedirs(output_dir, exist_ok=True)

    # Define minimum and maximum contour perimeters to filter tuse a frame captured by the hick borders
    min_area = 6000  # Adjust this based on the border thickness
    max_area = 10000  # Adjust this based on the size of the enclosed images

    images = []
    coordinates = []

    # Extract and save the enclosed images with thick white borders
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_area < area < max_area:  # Define your min and max area values here
            # Find bounding rectangle to get the region of interest
            x, y, w, h = cv2.boundingRect(contour)

            # Crop the region of interest (excluding the border)
            extracted_image = arena[y + 15:y + h - 15, x + 15:x + w - 15]

            # Apply sharpening kernel
            sharpening_kernel = np.array([[-0.1, -0.1, -0.1],
                                          [-0.1, 1.5, -0.1],
                                          [-0.1, -0.1, -0.1]])
            sharpened_extracted_image = cv2.filter2D(extracted_image, -1, sharpening_kernel)

            # Resize the sharpened extracted image to 224x224
            extracted_resized = cv2.resize(extracted_image, (224, 224))
            images.append(extracted_resized)
            coordinates.append([x, y, w, h])

    for i in range(1, len(images) + 1):
        cv2.imwrite(f'{output_dir}/extracted_image_{i}.png', images[i - 1])

    return images, coordinates

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
    global arena
    global event_coordinates
    global predicted_labels
    model = tf.keras.models.load_model('/home/deepakachu/Desktop/eyantra_stage_2/task2b/saved_models/model_2b_revamped_test')
    captured_frame = frame_capture()
    arena = arena_extraction(captured_frame)
    events, event_coordinates = event_extraction(arena)
    predicted_labels = event_label_prediction(model, events)
    identified_labels = event_mapping(event_coordinates, predicted_labels)

    return identified_labels
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
    identified_labels = return_4a_helper()
##################################################
    return identified_labels

###############	Main Function	#################
# (Previous code...)

if __name__ == "__main__":
    global arena
    global event_coordinates
    global predicted_labels

    identified_labels = task_4a_return()
    cap = cv2.VideoCapture(2)
    print(identified_labels)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Process the frame to get the arena_labelled
        arena_labelled = bounding_box(arena, event_coordinates, predicted_labels)
        arena_labelled = cv2.resize(arena_labelled, (900, 900))

        # Show the live video stream
        cv2.imshow('Live Video Feed', frame)

        # Show the processed frame (arena_labelled)
        cv2.imshow("Processed Frame", arena_labelled)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
