import cv2
import cv2.aruco as aruco
import numpy as np
import time


def detect_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)
    return markerCorners, markerIds

def extract_roi(image, corners):
    if corners is not None and len(corners) > 0:
        all_corners = np.concatenate(corners, axis=0)
        x, y, w, h = cv2.boundingRect(all_corners)

        src_points = np.array(all_corners, dtype=np.float32)
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        roi = cv2.warpPerspective(image, transform_matrix, (w, h))
        return roi
    else:
        print("No ArUco markers detected or an error occurred during extraction.")
        return None

if __name__ == "__main__":
    image_path = '/experimetation/arena.png'
    original_image = cv2.imread(image_path)

    markerCorners, markerIds = detect_markers(original_image)
    if markerIds is not None and len(markerIds) > 0:
        cv2.aruco.drawDetectedMarkers(original_image, markerCorners, markerIds)
        cv2.imshow('Detected Markers', original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        extracted_roi = extract_roi(original_image, markerCorners)
        if extracted_roi is not None:
            cv2.imshow('Transformed ROI', extracted_roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("ROI successfully extracted.")
            print("Shape of the extracted ROI:", extracted_roi.shape)
