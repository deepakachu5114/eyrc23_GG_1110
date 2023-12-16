import cv2
import cv2.aruco as aruco
import numpy as np
import time
def extract_rectangular_roi(image_path, scale_factor=2.0):
    # Read the original image
    original_image = cv2.imread(image_path)

    # Resize the image while preserving the aspect ratio
    height, width = original_image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    scaled_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Scaled Image', original_image)
    time.sleep(10)

    # Convert the scaled image to grayscale
    gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)


    corners, ids, rejectedCandidates = detector.detectMarkers(original_image)

    # Draw the detected markers on the scaled image
    aruco.drawDetectedMarkers(scaled_image, corners, ids)

    # Extract the bounding box around all detected markers
    if ids is not None and len(ids) > 0:
        all_corners = [corner for marker_corners in corners for corner in marker_corners]
        x, y, w, h = cv2.boundingRect(np.array(all_corners))

        # Perspective transformation to obtain a top-down view
        src_points = np.array(all_corners, dtype=np.float32)
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        roi = cv2.warpPerspective(original_image, transform_matrix, (w, h))

        # Display the original and transformed images
        cv2.imshow('Original Image', original_image)
        cv2.imshow('Transformed ROI', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return roi

if __name__ == "__main__":
    image_path = r'C:\Users\AISHINI\Desktop\SS.png'
    extracted_roi = extract_rectangular_roi(image_path)

    # Print information about the extracted ROI
    if extracted_roi is not None:
        print("ROI successfully extracted.")
        print("Shape of the extracted ROI:", extracted_roi.shape)
    else:
        print("No ArUco markers detected or an error occurred during extraction.")

