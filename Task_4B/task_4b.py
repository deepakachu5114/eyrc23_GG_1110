import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import math

# Load the video file
video_path = r'C:\Users\AISHINI\Downloads\output.mp4'
cap = cv2.VideoCapture(video_path)

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

all_corners = r"C:\Users\AISHINI\PycharmProjects\lat_long.csv"
live_location = r"C:\Users\AISHINI\PycharmProjects\live_data.csv"
lat_lon = {}

def read_csv(csv_name):

    # open csv file (lat_lon.csv)
    # read "lat_lon.csv" file
    # store csv data in lat_lon dictionary as {id:[lat, lon].....}
    # return lat_lon

    with open(csv_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            ar_id = row[0]
            lat = row[1]
            lon = row[2]
            lat_lon[ar_id] = [lat, lon]

def write_csv(loc, csv_name):

    # open csv (csv_name)
    # write column names "lat", "lon"
    # write loc ([lat, lon]) in respective columns

    with open(csv_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["lat", "lon"])
        # Write the coordinates in the respective columns
        csvwriter.writerow([loc[0], loc[1]])

def tracker(ar_id, lat_lon):

    # find the lat, lon associated with ar_id (aruco id)
    # write these lat, lon to "live_data.csv"
    ar_id = str(ar_id)
    if ar_id in lat_lon:
        # Get the coordinates associated with the ar_id
        coordinate = lat_lon[ar_id]

        # Write the coordinates to the live_data.csv file
        write_csv(coordinate, live_location)

'''def detect_ArUco_details(image):
    ArUco_details_dict = {}

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    (corners, ids, rejected) = detector.detectMarkers(image)
    corners_list = []
    id_list = []

    for (markerCorner, markerID) in zip(corners, ids):
        corners = markerCorner.reshape((4, 2))
        corners_list.append(corners)
        corners1 = corners.tolist()

        topLeft = [float(corners1[0][0]), float(corners1[0][1])]
        topRight = [float(corners1[1][0]), float(corners1[1][1])]
        bottomRight = [float(corners1[2][0]), float(corners1[2][1])]
        bottomLeft = [float(corners1[3][0]), float(corners1[3][1])]
        cX = int((topLeft[0] + bottomLeft[0] + topRight[0] + bottomRight[0]) / 4.0)
        cY = int((topLeft[1] + bottomLeft[1] + topRight[1] + bottomRight[1]) / 4.0)
        dx = topLeft[0] - topRight[0]
        dy = topLeft[1] - topRight[1]
        angle = int(np.arctan2(dy, dx))
        markerID = markerID.item()
        id_list.append(markerID)
        ArUco_details_dict[markerID] = [[cX, cY], angle]

    ArUco_corners = dict(zip(id_list, corners_list))

    return ArUco_details_dict, ArUco_corners

def calculate_distance_between_markers(image, marker1_id, marker2_id):
    # Detect ArUco markers and get their coordinates
    ArUco_details_dict, _ = detect_ArUco_details(image)

    # Check if both markers are found
    if marker1_id in ArUco_details_dict and marker2_id in ArUco_details_dict:
        # Get the coordinates of each marker
        marker1_coords = np.array(ArUco_details_dict[marker1_id][0])
        print(marker1_coords)
        marker2_coords = np.array(ArUco_details_dict[marker2_id][0])
        print(marker2_coords)

        # Calculate the distance between marker centroids
        distance = np.linalg.norm(marker2_coords - marker1_coords)

        return distance
    else:
        return None
'''
read_csv(all_corners)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Detect markers
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)


    # Check if ArUco marker with ID 1 is found
    if ids is not None and 1 in ids:
        ids = ids.flatten().tolist()
        dist = 99999
        loc = 900
        for id in lat_lon.keys():
            #if id in ids:
                if id != 1:
                    corners1 = []
                    corners2 = []
                    for (markerCorner, markerID) in zip(corners, ids):
                        if markerID == 1:
                            corners1 = markerCorner.reshape((4, 2)).tolist()
                        if markerID == id:
                            corners2 = markerCorner.reshape((4, 2)).tolist()
                        print(corners1)
                        print(corners2)
                        topLeft = [float(corners1[0][0]), float(corners1[0][1])]
                        topRight = [float(corners1[1][0]), float(corners1[1][1])]
                        bottomRight = [float(corners1[2][0]), float(corners1[2][1])]
                        bottomLeft = [float(corners1[3][0]), float(corners1[3][1])]
                        cX = int((topLeft[0] + bottomLeft[0] + topRight[0] + bottomRight[0]) / 4.0)
                        cY = int((topLeft[1] + bottomLeft[1] + topRight[1] + bottomRight[1]) / 4.0)
                        topLeft1 = [float(corners2[0][0]), float(corners2[0][1])]
                        topRight1 = [float(corners2[1][0]), float(corners2[1][1])]
                        bottomRight1 = [float(corners2[2][0]), float(corners2[2][1])]
                        bottomLeft1 = [float(corners2[3][0]), float(corners2[3][1])]
                        cX1 = int((topLeft1[0] + bottomLeft1[0] + topRight1[0] + bottomRight1[0]) / 4.0)
                        cY1 = int((topLeft1[1] + bottomLeft1[1] + topRight1[1] + bottomRight1[1]) / 4.0)
                    distance = math.sqrt((cX-cX1)**2+(cY-cY1)**2)
                    print(distance)
                    if distance < dist:
                        dist = distance
                        loc = id

        tracker(loc, lat_lon)


    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
