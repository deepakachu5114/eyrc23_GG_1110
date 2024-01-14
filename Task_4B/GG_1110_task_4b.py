'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 4B of Geo Guide (GG) Theme (eYRC 2023-24).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:           GG_1110
# Author List:       Aishini Bhattacharjee, Adithya Ubaradka, Deepak C Nayak, Upasana Nayak
# Filename:          task_4b.py

####################### IMPORT MODULES #######################
import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import math

##############################################################

all_corners = "/home/deepakachu/Desktop/eyantra_stage_2/experimetation/lat_long(2).csv"
live_location = "/home/deepakachu/Desktop/eyantra_stage_2/experimetation/live_data_1.csv"
lat_lon = {}


def read_csv(csv_name):
    # Function to read CSV file and store data in the lat_lon dictionary
    with open(csv_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            ar_id = row[0]
            lat = row[1]
            lon = row[2]
            lat_lon[ar_id] = [lat, lon]


def write_csv(loc, csv_name):
    # Function to write coordinates to a CSV file
    with open(csv_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["lat", "lon"])
        csvwriter.writerow([loc[0], loc[1]])


def tracker(ar_id, lat_lon):
    # Function to track ArUco markers and write their coordinates to a CSV file
    ar_id = str(ar_id)
    if ar_id in lat_lon:
        coordinate = lat_lon[ar_id]
        write_csv(coordinate, live_location)


def main():
    last_5_unique_points = []
    read_csv(all_corners)
    print(lat_lon)

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980 * 0.6)  # Width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 * 0.6)  # Height

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        cv2.imshow("Live Feed", frame)

        if ids is not None and 1 in ids:
            ids = ids.flatten().tolist()
            corners = [arr[0].tolist() for arr in corners]

            indices_to_remove = []

            for index, value in enumerate(ids):
                if value in [4, 5, 6, 7]:
                    indices_to_remove.append(index)

            for index in sorted(indices_to_remove, reverse=True):
                del corners[index]
                del ids[index]

            dist = 99999
            loc = 900
            threshold = 60

            corners1 = corners[ids.index(1)]

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

                    if distance < dist:
                        dist = distance
                        loc = id

            print(loc, dist)

            if dist <= threshold:
                if loc not in last_5_unique_points:
                    tracker(loc, lat_lon)
                    last_5_unique_points.append(loc)
                    last_5_unique_points = last_5_unique_points[-5:]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
