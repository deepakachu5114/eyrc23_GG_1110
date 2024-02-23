'''
* Team Id : GG_1110
* Author List : Aishini Bhattacharjee, Adithya S Ubaradka, Deepak C Nayak, Upasana Nayak
* Filename: sample_track.py
* Theme: GeoGuide
* Functions: read_csv, write_csv,track,tracking
* Global Variables: all_corners, live_location, lat_lon, last_5_unique_points
'''


####################### IMPORT MODULES #######################
import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import math

#all_corners: the path to a csv file lat_long(2).csv containing arena aruco id's along with the respective latitude, longitude of the streets
all_corners = "/home/deepakachu/Desktop/eyantra_stage_2/experimetation/lat_long(2).csv"

#live_location: the path to a csv file live_data_1.csv which gets updated with coordinates of ArUco markers
live_location = "/home/deepakachu/Desktop/eyantra_stage_2/experimetation/live_data_1.csv"

lat_lon = {}

# last_5_unique_points : List to store the last 5 unique points detected
last_5_unique_points = []

##############################################################

'''
    * Function Name: read_csv
    * Input: csv_name (string) - Name of the CSV file to  read
    * Output: None
    * Logic: Reads CSV file and stores data in the lat_lon dictionary
    * Example Call: read_csv("filename.csv")
    '''
def read_csv(csv_name): 
    # Read the csv_name file
    with open(csv_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row

        # Add the Latitude and Longitude from each row of the CSV file to the lat_lon dictionary. Use the ArUco ID as the index/key for each entry in the dictionary
        for row in csvreader:
            ar_id = row[0]
            lat = row[1]
            lon = row[2]
            lat_lon[ar_id] = [lat, lon]


'''
    * Function Name: write_csv
    * Input: 
        - loc (list) : List containing latitude and longitude
        - csv_name (string) : Name of the CSV file to write
    * Output: None
    * Logic: Writes coordinates to a CSV file
    * Example Call: write_csv([lat, lon], "filename.csv")
'''
def write_csv(loc, csv_name):
    # Write data to the csv file csv_name
    with open(csv_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # It writes a header row with column names "lat" and "lon" to indicate latitude and longitude.
        csvwriter.writerow(["lat", "lon"])
        # It writes a new row containing the latitude and longitude values provided in the 'loc' list.
        csvwriter.writerow([loc[0], loc[1]])

'''
    * Function Name: tracker
    * Input: 
        - ar_id (int) : ArUco marker ID
        - lat_lon (dict) : Dictionary containing ArUco marker IDs and corresponding coordinates
    * Output: None
    * Logic: Tracks ArUco markers and writes their coordinates to a CSV file
    * Example Call: tracker(100, lat_lon)
'''
def tracker(ar_id, lat_lon):
    # Convert ar_id to a string
    ar_id = str(ar_id)
    #Check if the given ArUco marker ID (ar_id) is present in the lat_lon dictionary. If it is, fetch the corresponding coordinate from the dictionary using the ArUco marker ID as the index. Subsequently, write this coordinate to a CSV file utilizing the write_csv function.
    if ar_id in lat_lon:
        coordinate = lat_lon[ar_id]
        write_csv(coordinate, live_location)

'''
    * Function Name: tracking
    * Input: 
        - frame (numpy array) : Image frame from the camera
        - ret (bool) : Flag indicating if a frame has been read successfully
    * Output: None
    * Logic: Tracks ArUco markers in the frame and writes their coordinates to a CSV file
    * Example Call: tracking(frame1, ret)
    '''
def tracking(frame, ret):

    global last_5_unique_points
    #Read ArUco marker coordinates from the csv file
    read_csv(all_corners)

    # aruco_dict : Get predefined ArUco dictionary with a 4x4 grid and marker size of 250 pixels
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    # Define parameters for ArUco marker detection
    parameters = aruco.DetectorParameters()

    # Detect ArUco markers in the frame.
    # corners: contains a list of lists, where each inner list represents the four corners of an ArUco marker detected in the frame
    # ids: a list of ArUco marker IDs corresponding to the detected markers
    # _ : to store rejected candidates or markers that do not meet certain criteria during the detection process
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # Show the frame with ArUco markers
    cv2.imshow("Live Feed", cv2.resize(frame, None, fx=1.5, fy=1.5))

    # Check if ArUco markers are detected and if ArUco marker ID 100 is present 
    if ids is not None and 100 in ids:
        # Flatten the IDs array
        ids = ids.flatten().tolist()
        # Convert corners array to list
        corners = [arr[0].tolist() for arr in corners]

        # indices_to_remove: list of specific indices which need to be removed from the IDs and corners list
        indices_to_remove = []

        # Iterate through the list of ArUco marker IDs and their corresponding indices using enumerate
        for index, value in enumerate(ids):
            if value in [4, 5, 6, 7]:
                # If the value matches, add the index to the list of indices to remove
                indices_to_remove.append(index)

        # Sort the list of indices to remove in reverse order
        for index in sorted(indices_to_remove, reverse=True):
            # Remove the corresponding corner and ID from the list
            del corners[index]
            del ids[index]

        # dist : Maximum  distance between two ArUco markers
        dist = 99999
        # loc :  stores the ArUco marker ID (location) that is currently considered closest to the reference marker (ID 100)
        loc = 900
        # threshold : maximum acceptable distance between two ArUco markers for them to be considered as close
        threshold = 31.5

        # Get corners of ArUco marker with ID 100
        corners1 = corners[ids.index(100)]

        # Loop through detected ArUco markers
        for id in ids:
            if id != 100:
                corners2 = corners[ids.index(id)]

                # Calculate center coordinates of the detected ArUco marker. 
                # The four corners of detected ArUco marker are represented as liste of (x,y) coordinates. Each corner corresponds to one of the four corners of the marker
                topLeft = [float(corners1[0][0]), float(corners1[0][1])]
                topRight = [float(corners1[1][0]), float(corners1[1][1])]
                bottomRight = [float(corners1[2][0]), float(corners1[2][1])]
                bottomLeft = [float(corners1[3][0]), float(corners1[3][1])]

                # The centroid is calculated by taking the average of the x-coordinates and y-coordinates of the marker's corners.
                cX = int((topLeft[0] + bottomLeft[0] + topRight[0] + bottomRight[0]) / 4.0)
                cY = int((topLeft[1] + bottomLeft[1] + topRight[1] + bottomRight[1]) / 4.0)


                # Calculate center coordinates of the detected ArUco marker. 
                # The four corners of detected ArUco marker are represented as liste of (x,y) coordinates. Each corner corresponds to one of the four corners of the marker
                topLeft1 = [float(corners2[0][0]), float(corners2[0][1])]
                topRight1 = [float(corners2[1][0]), float(corners2[1][1])]
                bottomRight1 = [float(corners2[2][0]), float(corners2[2][1])]
                bottomLeft1 = [float(corners2[3][0]), float(corners2[3][1])]

                # The centroid is calculated by taking the average of the x-coordinates and y-coordinates of the marker's corners.
                cX1 = int((topLeft1[0] + bottomLeft1[0] + topRight1[0] + bottomRight1[0]) / 4.0)
                cY1 = int((topLeft1[1] + bottomLeft1[1] + topRight1[1] + bottomRight1[1]) / 4.0)

                # Calculate distance between centers of ArUco markers using distance formula
                distance = math.sqrt((cX - cX1) ** 2 + (cY - cY1) ** 2)

                # Update closest marker ID and distance if distance is smaller than previous
                if distance < dist:
                    dist = distance
                    loc = id

        # Check if closest marker is within threshold distance
        if dist <= threshold:
            # If the closest marker is not in the last 5 unique points list, track it using the tracker function.
            if loc not in last_5_unique_points:
                tracker(loc, lat_lon)
                # Add the current ArUco marker ID 'loc' to the list last_5_unique_points only if it is not already present in the list.
                last_5_unique_points.append(loc)
                last_5_unique_points = last_5_unique_points[-5:]
