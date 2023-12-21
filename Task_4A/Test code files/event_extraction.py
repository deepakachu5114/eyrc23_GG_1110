import sys

import cv2
import numpy as np
import os
import tensorflow as tf


# this function maps the coordinates to their events

def map_coordinates_to_events(coordinates):
    mapping = {}
    # Sort by y values
    sorted_coordinates = sorted(coordinates, key=lambda item: item[1])

    # first element corresponds to E, and the last one to A
    mapping["E"] = sorted_coordinates[0]
    mapping["A"] = sorted_coordinates[-1]

    # Remove the first and last elements from sorted_coordinates
    sorted_coordinates = sorted_coordinates[1:-1]

    # sort by  x values
    sorted_coordinates = sorted(sorted_coordinates, key=lambda item: item[0])

    # first element corresponds to D, and the last one to B
    mapping["D"] = sorted_coordinates[0]
    mapping["C"] = sorted_coordinates[-1]

    # The remaining element corresponds to C
    mapping["B"] = sorted_coordinates[1]
    return mapping


# this function maps the labels to their events

def map_labels_to_events(coordinates, predictions):
    mapping = {}
    # Sort by y values
    sorted_coordinates = sorted(coordinates, key=lambda item: item[1])

    # first element corresponds to E, and the last one to A

    mapping["E"] = predictions[np.bincount(np.where(np.array(coordinates) == sorted_coordinates[0])[0]).argmax()]
    mapping["A"] = predictions[np.bincount(np.where(np.array(coordinates) == sorted_coordinates[-1])[0]).argmax()]

    # Remove the first and last elements from sorted_coordinates
    sorted_coordinates = sorted_coordinates[1:-1]

    # sort by  x values
    sorted_coordinates = sorted(sorted_coordinates, key=lambda item: item[0])

    # first element corresponds to D, and the last one to B
    mapping["D"] = predictions[np.bincount(np.where(np.array(coordinates) == sorted_coordinates[0])[0]).argmax()]
    mapping["C"] = predictions[np.bincount(np.where(np.array(coordinates) == sorted_coordinates[-1])[0]).argmax()]

    # The remaining element corresponds to C
    mapping["B"] = predictions[np.bincount(np.where(np.array(coordinates) == sorted_coordinates[1])[0]).argmax()]
    return mapping


'''image = cv2.imread('sample_arenas/arena.png')'''
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

image = frame


# Read the image

# Apply Bilateral filter
# image = cv2.bilateralFilter(image, 9, 75, 75)  # Parameters can be adjusted

# Resize the image to 700x700
# resized_image = cv2.resize(image, (800, 800))

# Convert the resized image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a mask of the white area
_, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a directory to save the extracted images

output_dir = '../../../experimetation/extracted_images'
os.makedirs(output_dir, exist_ok=True)

# Define minimum and maximum contour perimeters to filter thick borders
min_contour_perimeter = 100  # Adjust this based on the border thickness
max_contour_perimeter = 1000  # Adjust this based on the size of the enclosed images

images = []
coordinates = []

# Extract and save the enclosed images with thick white borders
for i, contour in enumerate(contours):
    perimeter = cv2.arcLength(contour, True)
    if min_contour_perimeter < perimeter < max_contour_perimeter:
        # Find bounding rectangle to get the region of interest
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the region of interest (excluding the border)
        extracted_image = image[y + 5:y + h - 5, x + 5:x + w - 5]

        sharpening_kernel = np.array([[-0.2, -0.2, -0.2],
                                      [-0.2, 2.5, -0.2],
                                      [-0.2, -0.2, -0.2]])

        sharpened_extracted_image = cv2.filter2D(extracted_image, -1, sharpening_kernel)

        # Resize the denoised extracted image to 224x224
        extracted_resized = cv2.resize(sharpened_extracted_image, (224, 224))
        images.append(extracted_resized)
        coordinates.append([x, y, w, h])

for i in range(1, len(images) + 1):
    cv2.imwrite(f'{output_dir}/extracted_image_{i}.png', images[i - 1])

model = tf.keras.models.load_model("/home/deepakachu/Desktop/eyantra_stage_2/task2b/saved_models/model_2b_revamped")

predictions = []
for img in images:
    pred_arr = model.predict(np.array([img]))
    pred_val = np.where(pred_arr[0] == max(pred_arr[0]))[0][0]
    if pred_val == 0:
        pred = "combat"
    elif pred_val == 1:
        pred = 'human_aid_rehabilitation'
    elif pred_val == 2:
        pred = 'fire'
    elif pred_val == 3:
        pred = 'military_vehicles'
    else:
        pred = 'destroyed_buildings'
    predictions.append(pred)

mapping = map_labels_to_events(coordinates, predictions)

event_letters = list(mapping.keys())
event_letters.sort()
sorted_mapping = {i: mapping[i] for i in event_letters}

# Now you have a dictionary where the keys are the event letters and the values are the labels
# You can use this dictionary to identify unique properties of each event's labels

print(sorted_mapping)

# adding the bounding box (green) with class description (green) to the image
# coordinates are stored in the coordinates array (done earlier)
img_with_box = None
for i, coordinate in enumerate(coordinates):
    x = coordinate[0]
    y = coordinate[1]
    w = coordinate[2]
    h = coordinate[3]
    img_with_box = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(image, predictions[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("imagebox", img_with_box)
cv2.waitKey(0)
cv2.destroyAllWindows()