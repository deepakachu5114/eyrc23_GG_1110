import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import math

cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980*0.6)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080*0.6)  # Height

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

all_corners = "/home/deepakachu/Desktop/eyantra_stage_2/experimetation/lat_long(2).csv"
live_location = "/home/deepakachu/Desktop/eyantra_stage_2/experimetation/GG_1110_task_4b.csv"
lat_lon = {}
last_5_unique_points = []

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

    # return lat_lon

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


read_csv(all_corners)
print(lat_lon)
while cap.isOpened():
    ret, frame = cap.read()

    # cv2.imshow("live feed", frame)

    if not ret:
        break

    # Detect markers
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # aruco.drawDetectedMarkers(frame, corners, ids)

    # resized_frame = cv2.resize(frame, None, fx=0.75, fy=0.75)
    cv2.imshow("Live Feed", frame)
    # print(corners)


    # Check if ArUco marker with ID 1 is found
    if ids is not None and 1 in ids:
        # print(ids)
    # print(corners)
        ids = ids.flatten().tolist()
        corners = [arr[0].tolist() for arr in corners]
        # Create a list to store indices of elements to remove
        indices_to_remove = []

        # Find indices of elements in ids list to remove from corners list
        for index, value in enumerate(ids):
            if value in [4, 5, 6, 7]:
                indices_to_remove.append(index)

        # Remove elements from corners list using indices_to_remove
        for index in sorted(indices_to_remove, reverse=True):
            del corners[index]
            del ids[index]

        # print(ids)
        # print(corners)
        dist = 99999
        loc = 900
        threshold = 60

        corners1 = corners[ids.index(1)]
        # print(corners1)

        for id in ids:
            if id != 1:
                corners2 = corners[ids.index(id)]

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
                distance = math.sqrt((cX - cX1) ** 2 + (cY - cY1) ** 2)
                # print(distance)

                if distance < dist:
                    dist = distance
                    loc = id
        print(loc, dist)
        if dist <= threshold:
            # continue
            if loc not in (last_5_unique_points):
                #continue since no unique marker is detected
                # continue
                # print(last_5_unique_points)
            # else:
                tracker(loc, lat_lon)
                # Add the detected point to the list of last 20 detected points
                last_5_unique_points.append(loc)

                # Keep only the last 20 elements in the list
                last_5_unique_points = last_5_unique_points[-5:]
                # print(last_5_unique_points)




    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()