'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to establish wifi connection between the esp32 and laptop Task 5A of Geo Guide (GG) Theme (eYRC 2023-24).
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
# Filename:			communication.py


####################### IMPORT MODULES #######################
import socket
import json
import numpy as np
import cv2
import cv2.aruco as aruco
import time
import threading
import sample_track
import experimetation.tracking as tracking
ESP_IP = '192.168.223.165'
ESP_PORT = 8266

#
# TOP_LEFT = [[52, 54], 100]
# TOP_RIGHT = [[10, 43], 100]
BOTTOM_RIGHT = [[14], 27]
EVENT_E = [[48, 47, 40], 57, 77, 50] # one away one close scheme
EVENT_D = [[34, 33], 53, 70] # one away one close scheme
EVENT_C = [[30,31], 48, 47]
EVENT_B = [[28, 29], 58, 50]
EVENT_A = [[21, 20], 47, 65] # one away one close scheme
BOT = 100


def calculate_distance(corners, ids, marker_id1, marker_id2):
    # Check if both markers are present
    if marker_id1 in ids and marker_id2 in ids:
        # Find the indices of the markers in the list
        index1 = np.where(ids == marker_id1)[0][0]
        index2 = np.where(ids == marker_id2)[0][0]

        # Get the corner coordinates of the markers
        corners1 = corners[index1][0]
        corners2 = corners[index2][0]

        # Calculate the distance between the centers of the markers
        center1 = np.mean(corners1, axis=0)
        center2 = np.mean(corners2, axis=0)

        distance = np.linalg.norm(center2 - center1)

        return distance
    else:
        return None


# Function to reset previous_ack_received to True after a delay
def reset_acknowledgment():
    global previous_ack_received
    time.sleep(5)  # Set the delay in secondsq
    previous_ack_received = True
    # print("Previous acknowledgment reset to True")

# Create a thread for resetting acknowledgment
reset_thread = threading.Thread(target=reset_acknowledgment)

# Start the reset thread
reset_thread.start()


def bot_status(corners, ids):
    # tl_dist = [calculate_distance(corners, ids, TOP_LEFT[0][0], BOT), calculate_distance(corners, ids, TOP_LEFT[0][1], BOT)]
    # tr_dist = [calculate_distance(corners, ids, TOP_RIGHT[0][0], BOT), calculate_distance(corners, ids, TOP_RIGHT[0][1], BOT)]
    br_dist = [calculate_distance(corners, ids, BOTTOM_RIGHT[0][0], BOT)]
    e_dist = [calculate_distance(corners, ids, EVENT_E[0][0], BOT), calculate_distance(corners, ids, EVENT_E[0][1], BOT), calculate_distance(corners, ids, EVENT_E[0][2], BOT)]
    d_dist = [calculate_distance(corners, ids, EVENT_D[0][0], BOT), calculate_distance(corners, ids, EVENT_D[0][1], BOT)]
    c_dist = [calculate_distance(corners, ids, EVENT_C[0][0], BOT), calculate_distance(corners, ids, EVENT_C[0][1], BOT)]
    b_dist = [calculate_distance(corners, ids, EVENT_B[0][0], BOT), calculate_distance(corners, ids, EVENT_B[0][1], BOT)]
    a_dist = [calculate_distance(corners, ids, EVENT_A[0][0], BOT), calculate_distance(corners, ids, EVENT_A[0][1], BOT)]


    # print("Top Left Distances:", tl_dist)
    # print("Top Right Distances:", tr_dist)
    # print("Bottom Right Distances:", br_dist)
    # print("Event E Distances:", e_dist)
    # print("Event D Distances:", d_dist)
    # print("Event C Distances:", c_dist)
    # print("Event B Distances:", b_dist)
    # print("Event A Distances:", a_dist)


    # if None not in tl_dist and (tl_dist[0] < TOP_LEFT[1] or tl_dist[1] < TOP_LEFT[1]):
    #     print(f"Extreme turn maneuver, top left: {tl_dist}")
    #     return 0
    #
    # if None not in tr_dist and (tr_dist[0] < TOP_RIGHT[1] or tr_dist[1] < TOP_RIGHT[1]):
    #     print(f"Extreme turn maneuver, top right: {tr_dist}")
    #     return 1
    #
    # if None not in br_dist and (br_dist[0] < BOTTOM_RIGHT[1] or br_dist[1] < BOTTOM_RIGHT[1]):
    #     print(f"Extreme turn maneuver, bottom right: {br_dist}")
    #     return 2

    if None not in e_dist and (e_dist[0] < EVENT_E[1] and e_dist[1] > EVENT_E[2] and e_dist[2] > EVENT_E[3]):
        print(f"Event E: {e_dist}")
        return 9

    if None not in d_dist and (d_dist[0] < EVENT_D[1] and d_dist[1] > EVENT_D[2]):
        print(f"Event D: {d_dist}")
        return 8

    if None not in c_dist and (c_dist[0] < EVENT_C[1] and c_dist[1] < EVENT_C[2]):
        print(f"Event C: {c_dist}")
        return 7

    if None not in b_dist and (b_dist[0] < EVENT_B[1] and b_dist[1] < EVENT_B[1]):
        print(f"Event B: {b_dist}")
        return 6

    if None not in a_dist and (a_dist[0] < EVENT_A[1] and a_dist[1] > EVENT_A[2]):
        print(f"Event A: {a_dist}")
        return 5

    if None not in br_dist and (br_dist[0] < BOTTOM_RIGHT[1]):
        print(f"Bottom Right: {br_dist}")
        return 1

    return 4



with open('/home/deepakachu/Desktop/eyantra_stage_2/experimetation/scam_config.json', 'r') as json_file:
    bot_path = list(json.load(json_file))

with open('bot_stop_numbers.json', 'r') as json_file:
    bot_stop_nos = list(json.load(json_file))
# Your array data
intersectionActions = bot_path

# Cree a dictionary with array size and data
data_dict = {"size": len(intersectionActions), "data": intersectionActions}

# Convert the dictionary to a JSON string
json_str = json.dumps(data_dict)

# Create a socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the ESP32
s.connect((ESP_IP, ESP_PORT))

# Send the JSON string without appending a newline character
s.sendall(json_str.encode('utf-8'))

# Receive acknowledgment from the ESP32
# acknowledgment = s.recv(1024).decode('utf-8')
# print(acknowledgment)


#
# Load the video file
video_path = '/home/deepakachu/Desktop/eyantra_stage_2/experimetation/fire_dataset/output_2.mp4'
cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

# Set the time interval for sending data to ESP32 (in seconds)
# send_interval =

frame_no = 0
previous_ack_received = True
bot_event_index = 0
sendMessage = True



# s.settimeout(0.25)q
while cap.isOpened():
    ret, frame = cap.read()
    if frame_no<20:
        frame_no+=1
        continue
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(2, 2))
    # clahe_image = clahe.apply(gray)
    tracking.tracking(frame, ret)


    if not ret:
        break

    # Detect markers
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)

    # Display the frame
    # cv2.imshow('ArUco Markers', frame)

    message = bot_status(corners, ids)
    # print(message)

    # time.sleep(send_interval)
    # Check if it's time to send the data

    # message = bot_status(corners, ids)
    # print(message)

    # Check if it's time to send the data
    # print(message)
    if (message is not None and message != 4 and sendMessage == True) or (message==1):
        print(message)

        if message == bot_stop_nos[bot_event_index] or message == 1:

            print(previous_ack_received)
            # Send the message only if the previous acknowledgment was received
            if previous_ack_received:
                s.sendall(f"{message}\n".encode('utf-8'))
                previous_ack_received = False
                print(message)

                # Receive acknowledgment from the ESP32


                #
                # acknowledgment = s.recv(1024).decode('utf-8')
                # print(acknowledgment)

                # Update the previous acknowledgment status
                # previous_ack_received = True if acknowledgment == "ACK" else False

                # Restart the reset thread for acknowledgment after 5 seconds
                reset_thread = threading.Thread(target=reset_acknowledgment)
                reset_thread.start()

            if (bot_event_index < len(bot_stop_nos)-1):
                bot_event_index+=1
            else:
                sendMessage = False

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Close the socket when you are done
s.close()
