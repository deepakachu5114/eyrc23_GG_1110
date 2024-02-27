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
# Theme: 			Geo Guide
# Functions:			detect_markers, arena_coordinates, get_corner_coordinates, arena_extraction, frame_capture,
#                       event_extraction, coord_overlap, pixelate_image, bilateral_denoising, increase_saturation,
#                       event_label_prediction, bounding_box,event_mapping, map_priority_to_events, event_bot_mapping,
#                       return_4A_helper
# Global Variables:		priority_edges

'''
The model used for event identification is below:
Team ID = GG_1110
Trained weights drive link = "https://drive.google.com/drive/folders/1HdPrfJNJyGFt6b7isnr9CwieWOdpZuRW?usp=sharing"

NOTE - While loading the model, load it by using the below lines (path to the whole directory, not the .pb file):
    model = tensorflow.keras.models.load_model('path/to/the/folder/model1')
'''

####################### IMPORT MODULES #######################
import cv2
import numpy as np
import tensorflow as tf
import os
import json
##############################################################



################# ADD UTILITY FUNCTIONS HERE #################

'''
Function Name: detect_markers
Input: image (image of the arena)
Output: markerCorners - multidimensional list containing the coordinates of the corners of the detected aruco markers,
        markerIds - list containing the ids of the detected aruco markers
Logic: This function detects the markers on the arena from the arena image 
        and returns their location in terms of coordinates and their ids
        
        The image is first converted to grayscale for detection.
        The dictionary and parameters for detection are defined using getPredefinedDictionary which is part of the 
        OpenCV library
        
        The required return values are found using the detectMarkers function which is also part of OpenCV
Example Call: corners,ids = detect_markers(image)
'''

def detect_markers(image):
 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    markerCorners, markerIds, rejectedIDs = detector.detectMarkers(gray)
    return markerCorners, markerIds

#########################################################################################################################################

'''
Function Name: arena_coordinates
Input:	top_left - list of top left coordinates of all aruco markers,
        top_right - list of top right coordinates of all aruco markers,
        bottom_right - list of bottom right coordinates of all aruco markers,
        bottom_left - list of bottom left coordinates of all aruco markers
Output:	arena_coordinates - list of the coordinates of the four corners of the arena
Logic: This function calculates the coordinates of the arena based on marker corners
        Calculates coordinates for each corner of the arena based on marker corners:
        
            top left = min x, min y
            top right =  max x, min y
            bottom right = max x, max y
            bottom left = min x, max y
            
        In each, x and y denote the 0th and 1st index respectively of the inputs necessary to calculate the left hand side
            
        
Example Call: arena_coordinates = arena_coordinates(top_left, top_right, bottom_right, bottom_left)
'''

def arena_coordinates(top_left, top_right, bottom_right, bottom_left):
    
    tl_x = min(coord[0] for coord in top_left)
    tl_y = min(coord[1] for coord in top_left)
    tr_x = max(coord[0] for coord in top_right)
    tr_y = min(coord[1] for coord in top_right)
    br_x = max(coord[0] for coord in bottom_right)
    br_y = max(coord[1] for coord in bottom_right)
    bl_x = min(coord[0] for coord in bottom_left)
    bl_y = max(coord[1] for coord in bottom_left)
  
    arena_coordinates = [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]]

    return arena_coordinates
    
###############################################################################################################################################################################

'''
Function Name: get_corner_coordinates
Input:	markerCorners - multidimensional list containing the coordinates of the corners of the detected aruco markers,
        markerIds - list containing the ids of the detected aruco markers
Output:	coordinates - list of coordinates of the four corners of the arena
Logic: This function creates the lists required for the arena_coordinates function and then calculates the coordinates of the corners of the arena
    
        The corner list contains individual lists corresponding to the top left, top right, bottom right and
        bottom right coordinates of all the arucos. These are first filled in the individual lists and then 
        passed as arguments to the arena_coordinates function which finds the coordinates of the arena 
        corners.
        This list is then returned by the function.

Example Call: coordinates = get_corner_coordinates(markerCorners, markerIds)
'''

def get_corner_coordinates(markerCorners, markerIds):
    corner_list = [[], [], [], []]
    for i in range(len(markerIds)):
        corner_list[0].append(markerCorners[i][0][0])
        corner_list[1].append(markerCorners[i][0][1])
        corner_list[2].append(markerCorners[i][0][2])
        corner_list[3].append(markerCorners[i][0][3])
    coordinates = arena_coordinates(corner_list[0], corner_list[1], corner_list[2], corner_list[3])
    return coordinates

##############################################################################################################################################################################

'''
Function Name: arena_extraction
Input:	image - The image of the arena as seen by the camera
Output:	arena - The image of the extracted arena
Logic: This function extracts the image of the arena from the image of the arena from the camera
        The aruco markers' id and coordinates are found by calling the detect_markers function.
        Based on the arena coordinates found by calling the get_corner_coordinates function,
        we crop the image to extract the arena.

Example Call: arena = arena_extraction(image)
'''

def arena_extraction(image):
    markerCorners, markerIds = detect_markers(image)
    coordinates = get_corner_coordinates(markerCorners, markerIds)
    tl_x, tl_y = coordinates[0]
    br_x, br_y = coordinates[2]
    arena = image[int(tl_y):int(br_y), int(tl_x):int(br_x)]
    return arena

##############################################################################################################################################################################

'''
Function Name: frame_capture
Input:	---
Output:	frame
Logic: This function return every 20th frame captured from the camera.
Example Call: frame = frame_capture()
'''

def frame_capture():
    video_capture = cv2.VideoCapture(0)  # 0 for default webcam, change if needed
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

###############################################################################################################################################################################

'''
Function Name: event_extraction
Input:	arena - The extracted arena image
Output:	all_images - list of all the contours that may contain an event
        all_coordinates - list of the coordinates of all_images
        present_images - actual event images detected
        present_coordinates - coordinates of the actual event images detected
Logic: This function extracts the images from the arena images and their coordinates

        The minimum and maximum contour area are defined. These are dependent on the image
        captured by the camera and have to be adjusted.
        
        The arena image is grayscaled (for thresholding) and the contours are detected using a threshold which is incremented.
        This is done to obtain all possible images based on varying degrees of "white" since the camera is
        not perfect in its detection of "pure white". This depends on the lighting and the positioned angle 
        of the camera.
        
        The contours detected are stored in a list called "contours". This list contains many redundant contours.
        
        The redundant contours are removed by first checking the dimensions and seeing if they match that of the 
        events dimensions (based on the predefined areas). This also compensates for thick white borders. Valid contours
        have their dimensions found by adding the height and width of their x and y coordinates: (x,y,x+w,y+h)
        
        After this, the contours are checked for overlapping with any of the already found contours. 
        If there is, then such contours are redundant.
        
        The remaining contours have their coordinates added to a "processed_contours" set, and based on the coordinates,
        the arena is sliced to extract the image of the event.
        
        Now, there exists the possibility of a "no event" contour being detected, since the dimensions will match that
        of the events themselves.
        
        Thus we calculate the standard deviation of the image pixel values.  This can be used to measure the variance 
        (square of standard deviation) of the image pixels. This is done by applying many filters on the extracted image 
        and then finding the sum of squares of differences between the pixel values. If this value is above a certain 
        threshold then we can say that the image contains an event,
        
        The logic is that no matter how many filters you apply to a solid colour patch that doesnt fundamentally change the
        colour in local areas (such as a Gaussian Blur), the homogeneity of the image remains the same.
        
        The event image is then denoised and then sharpened by applying a sharpening kernel and added to the "present_images" 
        array along with the coordinates in the "present_coordinates" array.
        
        Each step has been documented below to show which step does which function mentioned above.
        
Example Call: extract = event_extraction(arena)
'''

def event_extraction(arena):
    gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=9, tileGridSize=(2, 2))
    clahe_image = clahe.apply(gray)


    contours_ = []
    for i in range(190, 250, 10):
        _, thresh = cv2.threshold(clahe_image, i, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_.append(list(contours))

    contours = (ele for sublist in contours_ for ele in sublist)
    # Create a directory to save the extracted images
    output_dir = '/home/deepakachu/Desktop/eyantra_stage_2/eyrc23_GG_1110/Task_4A/Submission files/extracted_images'
    os.makedirs(output_dir, exist_ok=True)

    # Define minimum and maximum contour area to filter thick borders
    min_area = 8300  # Adjust this based on the border thickness
    max_area = 9600  # Adjust this based on the size of the enclosed images

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
            blurred_image = cv2.GaussianBlur(cv2.resize(extracted_image, (224,224)), (21, 21), 0)
            blurred_image2 = cv2.GaussianBlur(cv2.resize(extracted_image, (224,224)), (3, 3), 0)
            brightened_image = cv2.convertScaleAbs(blurred_image2, alpha=0.6, beta=50)
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

            extracted_resized = cv2.resize(sharpened_extracted_image, (224, 224))
            denoised_image = bilateral_denoising(extracted_resized)

            all_images.append(extracted_resized)
            all_coordinates.append([x, y, w, h])


            # Check if the std_dev is above the minimum threshold
            if ssd >= 4000000:
                present_images.append(denoised_image)
                present_coordinates.append([x, y, w, h])
            else:
                present_coordinates.append(0)


    for i in range(1, len(all_images) + 1):
        # We save all the extracted images into a directory
        cv2.imwrite(f'{output_dir}/extracted_image_{i}.png', all_images[i - 1])

    return all_images, all_coordinates, present_images, present_coordinates

##############################################################################################################################################################

'''
Function Name: coord_overlap
Input: coord1 - first set of coordinates (x1,y1,x1+w1,y1+h1)
        coord2 - second set of coordinates (x2,y2,x2+w2,y2+h2)
Output:	True or False (boolean)
Logic: This function checks if two sets of coordinates overlap
        The coordinates are checked to see if one lies within the other.
Example Call: val = coord_overlap(coord1, coord2)
'''

def coord_overlap(coord1, coord2):
    x1, y1, x2, y2 = coord1
    x3, y3, x4, y4 = coord2

    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

###############################################################################################################################################################

'''
Function Name: bilateral_denoising
Input: image - input image
Output:	denoised image
Logic: This function applies the bilateral filter on the input image, this makes the image appear smoother 
        and legible. 
        
Example Call: img = bilateral_denoising(image)
'''

def bilateral_denoising(image, d=20, sigma_color=50, sigma_space=50):
    # Apply bilateral filter
    denoised_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return denoised_image

###############################################################################################################################################################

'''
Function Name: increase_saturation
Input: image - input image
Output:	saturated_image
Logic: This function saturates the input image, the higher the saturation factor, the more intense the colors appear
        This was done to make sure we do not lose out on any information that the image has due to its diverse 
        colors under an unfortunate lighting condition.
Example Call: img = increase_saturation(image)
'''

def increase_saturation(image, saturation_factor=0.7):
    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Increase the saturation channel
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)

    # Convert the image back to BGR
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return saturated_image

###############################################################################################################################################################

'''
Function Name: event_label_prediction
Input: model - the trained model for classification of the extracted events
        events - the extracted event images
Output:	predicted_labels - the list of labels corresponding to the events in
        "events".
Logic: This function classifies the events extracted. The events are passed into the
        model for prediction. This outputs an array of softmax probabilities.

        The model has been trained according to the mentioned integer mapping 
        to the string labels.
        0 - Combat
        1 - Human Aid and Rehabilitation
        2 - Fire
        3 - Military Vehicles
        4 - Destroyed Buildings
        
        The index of maximum probability corresponds to the above integer mapping.
        
Example Call: labels = event_label_prediction(model, events)
'''

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

###############################################################################################################################################################

'''

Function Name: bounding_box_prediction
Input: arena - the image of the arena
        event_coordinates - the coordinates of the events detected
        predicted_labels - the detected events' classifications / labels
Output:	arena - the arena with the labeled bounding boxes
Logic: This function constructs the bounding boxes and labels them around the detected events
        in the arena.
        The coordinates are used to draw the boxes in green (0,255,0). The labels are used to
        label the boxes.
Example Call: arena = bounding_box(arena,event_coordinates,labels)
'''

def bounding_box(arena, event_coordinates, predicted_labels):
    for i, coordinate in enumerate(event_coordinates):
        if coordinate != 0:
            x, y, w, h = coordinate
            cv2.rectangle(arena, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(arena, predicted_labels[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return arena

###############################################################################################################################################################

'''
Function Name: event_mapping
Input: all_event_coordinates - the coordinates of the contours that may contain an event
        predicted_labels - the predicted labels of the detected events
        present_coordinates - coordinates of the actual event images detected
Output:	a dictionary containing the events mapped to their alphabets
Logic: This function maps the events to their alphabets on the arena. 
        This is done by comparing the coordinates to the relative positions of 
        each alphabet on the arena.
Example Call: dict = event_mapping(all_event_coordinates, predicted_labels, present_coordinates)
'''

def event_mapping(all_event_coordinates, predicted_labels, present_coordinates):

    event_mapping = {}

    # Sort by y values
    sorted_coordinates = sorted(all_event_coordinates, key=lambda item: item[1])

    # first element corresponds to E, and the last one to A
    if (sorted_coordinates[0] in present_coordinates):
        event_mapping["E"] = predicted_labels[np.bincount(np.where(np.array(all_event_coordinates) == sorted_coordinates[0])[0]).argmax()]
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

###############################################################################################################################################################

'''
priority_edges : a dictionary containing the edges representing each event's defining edge.  This is in accordance
                with a graph representation of the whole arena.  The tuple associated with each event contains the
                starting and ending vertex of the priority edge
'''
priority_edges = {'A':('A', 'H'), 'E':('D', 'E'), 'D':('C', 'J'), 'C':('F', 'J'), 'B':('G', 'I')}


###############################################################################################################################################################

'''
Function Name: map_priority_to_events
Input: priority_edges - dictionary containing the events defining edge
        events - the detected events
Output:	list containing the mapping between the priority edges and the detected events which is sorted
        based on the priority of visiting
Logic: This function maps the priority edges to the detected events and arranges the list according to the priority
        of visiting the events.
        The alphabet is used to get the priority edges from the priority_edges dictionary, along with the priority
        number in a tuple.
        The list is then sorted according to this priority number.
Example Call: l = map_priority_to_events(priority_edges, events)
'''

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

    present_edge_list = []
    for key in sorted_dict.keys():
        present_edge_list.append(sorted_dict[key][1])

    return present_edge_list


###############################################################################################################################################################

'''
Function Name: event_bot_mapping
Input: present_edge_list - list containing the mapping between the priority edges and the detected events which is sorted
        based on the priority of visiting
        priority_edges - dictionary containing the event defining edges
Output:	A list mapping the events to the numbers associated with them in the order of their priority
Logic: This function maps the events to their numbers in order of their priority.
        The priorities are reflected in the order of the present_edge_list list (output of the previous function).
        If the edges in the priorities are present in the priority_edges then it is assigned the event number and 
        appended to the output list.
Example Call: event_bot_mapping_list = event_bot_mapping(present_edge_list,priority_edges)

'''

def event_bot_mapping(present_edge_list,
    priority_edges):

    event_bot_numbers = {'A':5, 'B':6, 'C':7, 'D':8, 'E':9}
    event_bot_mapping_list = []

    for edge in present_edge_list:
        for key, value in priority_edges.items():
            if value == edge:
                event_bot_mapping_list.append(event_bot_numbers[key])

    return event_bot_mapping_list


###############################################################################################################################################################

'''
Function Name: return_4a_helper
Input: ---
Output:	identified_labels - a dictionary containing the events mapped to their alphabets
Logic: This function loads the model and calls the functions (as shown below in the function) to
        extract the events from the arena and map them to the alphabets.
Example Call: identified_labels = return_4a_helper()

'''


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

###########################################################################################################################################################################

'''
Function Name: task_4a_return
Input: ---
Output:	identified_labels - { dictionary } dictionary containing the labels of the events detected
Logic: This function loads the model and calls the functions (as shown below in the function) to
        extract the events from the arena and map them to the alphabets.
Example Call: identified_labels = task_4a_return()

'''

def task_4a_return():

    identified_labels = return_4a_helper()
    return identified_labels

#############################################################################################################################################################################

'''
Function Name: printing_function
Input: identified_labels - { dictionary } dictionary containing the labels of the events detected
Output: new_dict - { dictionary } dictionary with changed names for printing in the terminal
Logic: This function creates a new dictionary which is the same as the input except for the 
        event label formats being changed to the required format for outputting to the terminal.
Example Call: new_dict = printing_function(identified_labels)
'''
def printing_function(identified_labels):

    new_dict = {}

    for key in identified_labels.keys():
        if identified_labels[key] ==  "combat":
            new_dict[key] = "Combat"
        elif identified_labels[key] ==  "human_aid_rehabilitation":
            new_dict[key] = "Humanitarian Aid and rehabilitation"
        elif identified_labels[key] ==  "fire":
            new_dict[key] = "Fire"
        elif identified_labels[key] ==  "military_vehicles":
            new_dict[key] = "Military Vehicles"
        elif identified_labels[key] ==  "destroyed_buildings":
            new_dict[key] = "Destroyed buildings"

    return new_dict

################################################################################################################################################################################

###############	Main Function	#################
if __name__ == "__main__":
    global arena
    global present_coordinates
    global predicted_labels
    identified_labels = task_4a_return()
    arena_labelled = bounding_box(arena, present_coordinates, predicted_labels)
    print_classes = printing_function(identified_labels)
    print(print_classes)
    m = map_priority_to_events(priority_edges, identified_labels)
    n = event_bot_mapping(m, priority_edges)

    # Dumping m into a JSON file
    # This will later be used by the shortest_path.py file to come up with the path for the bot.
    with open('priority_edge_order.json', 'w') as json_file:
        json.dump(m, json_file)
    # Dumping n into a JSON file
    # This will later be used by the communication.py file to send the bot the information about
    # which is the event the bot has to stop at currently.
    with open('bot_stop_numbers.json', 'w') as json_file:
        json.dump(n, json_file)
    cv2.imshow("Captured frame", arena_labelled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

########################################################