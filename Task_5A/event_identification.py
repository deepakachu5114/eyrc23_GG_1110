'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to detect and classify the events on the arena for Task 5A of Geo Guide (GG) Theme (eYRC 2023-24).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			GG_1110
# Author List:		Aishini Bhattacharjee, Adithya S Ubaradka, Deepak C Nayak, Upasana Nayak
# Filename:			event_identification.py


####################### IMPORT MODULES #######################
import cv2
import numpy as np
import tensorflow as tf
import os
import json
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
    # print(arena_coordinates)
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
    # frame = cv2.imread("/home/deepakachu/Desktop/eyantra_stage_2/experimentation/captured_images/frame_3.jpg")
    return frame



def event_extraction(arena):
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=9, tileGridSize=(2, 2))
    clahe_image = clahe.apply(gray)
    # cv2.imshow("clahe", cv2.resize(clahe_image, (720,540)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find contours

    contours_ = []
    for i in range(200, 250, 10):
        _, thresh = cv2.threshold(clahe_image, i, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_.append(list(contours))

    contours = (ele for sublist in contours_ for ele in sublist)
    # Create a directory to save the extracted images
    output_dir = '/home/deepakachu/Desktop/eyantra_stage_2/eyrc23_GG_1110/Task_4A/Submission files/extracted_images'
    os.makedirs(output_dir, exist_ok=True)

    # Define minimum and maximum contour perimeters to filter thick borders
    min_area = 8800  # Adjust this based on the border thickness
    max_area = 9250  # Adjust this based on the size of the enclosed images

    all_images = []
    all_coordinates = []
    present_images = []
    present_coordinates = []
    processed_coordinates = set()  # To keep track of processed coordinates

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 90 or h > 100:
                continue
            current_coordinates = (x, y, x + w, y + h)

            # Check if the current coordinates overlap with any processed coordinates
            if any(coord_overlap(current_coordinates, processed_coord) for processed_coord in processed_coordinates):
                continue  # Skip if there is an overlap

            processed_coordinates.add(current_coordinates)  # Add current coordinates to processed set

            extracted_image = arena[y + 10:y + h - 10, x + 10:x + w - 10]

            # Calculate the standard deviation of pixel values
            pixelated_extracted_image = pixelate_image(extracted_image, pixel_size=2.5)
            blurred_image = cv2.GaussianBlur(cv2.resize(extracted_image, (224,224)), (21, 21), 0)
            blurred_image2 = cv2.GaussianBlur(cv2.resize(extracted_image, (224,224)), (3, 3), 0)
            brightened_image = cv2.convertScaleAbs(blurred_image2, alpha=0.8, beta=15)
            saturated_image = increase_saturation(brightened_image)

            # Calculate the sum of squared differences (SSD)
            ssd = np.sum((cv2.resize(extracted_image, (224, 224)) - blurred_image) ** 2)
            # print(ssd)

            # Apply sharpening kernel
            sharpening_kernel = np.array([[-0.1, -0.1, -0.1],
                                          [-0.1, 1.5, -0.1],
                                          [-0.1, -0.1, -0.1]])
            sharpened_extracted_image = cv2.filter2D(saturated_image, -1, sharpening_kernel)

            # Resize the sharpened extracted image to 224x224

            extracted_resized = cv2.resize(saturated_image, (224, 224))
            denoised_image = bilateral_denoising(extracted_resized)

            all_images.append(denoised_image)
            all_coordinates.append([x, y, w, h])


            # Check if the std_dev is above the minimum threshold
            if ssd >= 5400000:
                present_images.append(denoised_image)
                present_coordinates.append([x, y, w, h])
            else:
                present_coordinates.append(0)


    for i in range(1, len(all_images) + 1):
        cv2.imwrite(f'{output_dir}/extracted_image_{i}.png', all_images[i - 1])

    return all_images, all_coordinates, present_images, present_coordinates

def coord_overlap(coord1, coord2):
    """Check if two sets of coordinates overlap."""
    x1, y1, x2, y2 = coord1
    x3, y3, x4, y4 = coord2

    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

def pixelate_image(image, pixel_size):
    small = cv2.resize(image, None, fx=1.0/pixel_size, fy=1.0/pixel_size, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

def bilateral_denoising(image, d=20, sigma_color=50, sigma_space=50):
    # Apply bilateral filter
    denoised_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return denoised_image

def increase_saturation(image, saturation_factor=0.7):
    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Increase the saturation channel
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)

    # Convert the image back to BGR
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return saturated_image
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
        if coordinate != 0:
            x, y, w, h = coordinate
            cv2.rectangle(arena, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(arena, predicted_labels[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return arena



def event_mapping(all_event_coordinates, predicted_labels, present_coordinates):
    # print(predicted_labels)
    event_mapping = {}

    # Sort by y values
    sorted_coordinates = sorted(all_event_coordinates, key=lambda item: item[1])
    # print(sorted_coordinates)
    # print(present_coordinates)
    # print(all_event_coordinates)
    # first element corresponds to E, and the last one to A
    if (sorted_coordinates[0] in present_coordinates):
        event_mapping["E"] = predicted_labels[np.bincount(np.where(np.array(all_event_coordinates) == sorted_coordinates[0])[0]).argmax()]
    # print(sorted_coordinates[0])
    if sorted_coordinates[-1] in present_coordinates:
        event_mapping["A"] = predicted_labels[np.bincount(np.where(np.array(all_event_coordinates) == sorted_coordinates[-1])[0]).argmax()]
    # Remove the first and last elements from sorted_coordinates
    sorted_coordinates = sorted_coordinates[1:-1]
    # sort by x values
    sorted_coordinates = sorted(sorted_coordinates, key=lambda item: item[0])
    # first element corresponds to D, and the last one to B
    if sorted_coordinates[0] in present_coordinates:
        event_mapping["D"] = predicted_labels[np.bincount(np.where(np.array(all_event_coordinates) == sorted_coordinates[0])[0]).argmax()]
    if sorted_coordinates[-1] in present_coordinates:
        event_mapping["C"] = predicted_labels[np.bincount(np.where(np.array(all_event_coordinates) == sorted_coordinates[-1])[0]).argmax()]
    # The remaining element corresponds to C
    if sorted_coordinates[1] in present_coordinates:
        event_mapping["B"] = predicted_labels[np.bincount(np.where(np.array(all_event_coordinates) == sorted_coordinates[1])[0]).argmax()]


    return dict(sorted(event_mapping.items()))


priority_edges = {'A':('A', 'H'), 'E':('D', 'E'), 'D':('C', 'J'), 'C':('F', 'J'), 'B':('G', 'I')}

def map_priority_to_events(priority_edges, events):
    mapped_dict = {}
    for key in events.keys():
        if events[key] == "fire":
            mapped_dict[key] = (1,priority_edges[key])
        elif events[key] == "destroyed_buildings":
            mapped_dict[key] = (2,priority_edges[key])
        elif events[key] == "human_aid_rehabilitation":
            mapped_dict[key] = (3,priority_edges[key])
        elif events[key] == "military_vehicles":
            mapped_dict[key] = (4,priority_edges[key])
        elif events[key] == "combat":
            mapped_dict[key] = (5,priority_edges[key])

    sorted_dict = dict(sorted(mapped_dict.items(), key=lambda item: item[1][0]))

    l = []
    for key in sorted_dict.keys():
        l.append(sorted_dict[key][1])

    return l


def event_bot_mapping(present_edge_list,
    priority_edges):

    event_bot_numbers = {'A':5, 'B':6, 'C':7, 'D':8, 'E':9}
    event_bot_mapping_list = []

    for edge in present_edge_list:
        for key, value in priority_edges.items():
            if value == edge:
                event_bot_mapping_list.append(event_bot_numbers[key])
    # print(event_bot_mapping_list)
    return event_bot_mapping_list





def return_4a_helper():
    global arena
    global present_coordinates
    global predicted_labels
    model = tf.keras.models.load_model('/home/deepakachu/Desktop/eyantra_stage_2/task2b/saved_models/model_2b_revamped_test')
    captured_frame = frame_capture()
    arena = arena_extraction(captured_frame)
    all_images, all_coordinates, present_images, present_coordinates = event_extraction(arena)
    predicted_labels = event_label_prediction(model, all_images)
    identified_labels = event_mapping(all_coordinates, predicted_labels, present_coordinates)

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
if __name__ == "__main__":
    global arena
    global present_coordinates
    global predicted_labels
    identified_labels = task_4a_return()
    arena_labelled = bounding_box(arena, present_coordinates, predicted_labels)
    # arena_labelled = cv2.resize(arena_labelled, (720, 540))
    print(identified_labels)
    m = map_priority_to_events(priority_edges, identified_labels)
    n = event_bot_mapping(m, priority_edges)

    # Dumping m into a JSON file
    with open('priority_edge_order.json', 'w') as json_file:
        json.dump(m, json_file)
    # print(m)
    with open('bot_stop_numbers.json', 'w') as json_file:
        json.dump(n, json_file)
    cv2.imshow("Captured frame", arena_labelled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()