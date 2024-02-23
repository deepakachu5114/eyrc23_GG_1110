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

# Team ID:		GG_1110
# Author List:		Aishini Bhattacharjee, Adithya S Ubaradka, Deepak C Nayak, Upasana Nayak
# Filename:		communication.py
# Theme:                Geo Guide
# Functions:            calculate_distance, reset_acknowledgment, bot_status
# Global Variables:                                                       ESP_IP,ESP_PORT,EVENT_E,EVENT_D,EVENT_C,EVENT_B,EVENT,A,BOT,reset_thread,intersectionActions,data_dict,json_str,s,video_path,cap,aruco_dict,parameters,frame_no,previous_ack_received,bot_event_index,sendMessage   


####################### IMPORT MODULES #######################################################################################################################
import socket
import json
import numpy as np
import cv2
import cv2.aruco as aruco
import time
import threading
import sample_track
##############################################################################################################################################################


ESP_IP = '192.168.125.165' # ip address of the esp32
ESP_PORT = 8266 # esp32 port number 

''' 
EVENT_(event letter) : List that contains three elements. 
			The first is a list containing 2 aruco ids that mark the two "ends"
			of that event and can be used for identifying that event letter.
			The second element contains the threshold for the distances between the FIRST aruco id and the bot
			The third element contains the threshold for the distance between the SECOND aruco id and the bot
			These thresholds are a property of the arena image that has been detected by the camera.
			
BOT : the aruco id of the bot
'''

EVENT_E = [[48, 47], 57, 65] 
EVENT_D = [[34, 33], 53, 70] 
EVENT_C = [[30,31], 50]
EVENT_B = [[28, 29], 58]
EVENT_A = [[21, 20],50, 65] 
BOT = 100 

###############################################################################################################################################################

'''
Function Name: calculate_distance
Input: corners - multidimensional list containing the coordinates of the corners of the detected aruco markers, 
	ids - list containing the ids of the detected aruco markers, 
	marker_id1 - id of the first aruco marker, 
	marker_id2 - id of the second aruco marker
Output: distance (if markers are in id list),
	None (if markers are not in id list)
Logic: This function calculates the euclidean distance between two aruco markers.

	The "corners" list contains the coordinates of the corners of the detected aruco markers.
	The ids of the aruco markers pertaining to the events and the bot are included in the global variables defined above.
	
	During the running, it is possible that certain frames captured by the camera in real time may not be such that all
	arucos markers are successfully detected. It may be necessary to calculate distances involving these undetected markers.
	Thus, to avoid an error, we handle the undetected case separately by checking if the ids are in the id list.
	
	If the ids are in the list, we then obtain the index of occurrence of these arucos in the same list. This index corresponds 
	to the location of the said ids coordinates in the "corners" list.
	
	The centre of the aruco marker is then found by finding the centroid of the coordinates
	
	Lastly, the frobenius norm (l2 norm or euclidean distance) of the difference vector between the two centres is found.
	This gives the required distance.
	
	If the ids are not in the list, we return "None"
	
Example Call: sample_distance = calculate_distance(corners, ids, EVENT_E[0][0], BOT)
'''

def calculate_distance(corners, ids, marker_id1, marker_id2):
 
    if marker_id1 in ids and marker_id2 in ids:
       
        index1 = np.where(ids == marker_id1)[0][0]
        index2 = np.where(ids == marker_id2)[0][0]

        corners1 = corners[index1][0]
        corners2 = corners[index2][0]

        center1 = np.mean(corners1, axis=0)
        center2 = np.mean(corners2, axis=0)

        distance = np.linalg.norm(center2 - center1)

        return distance
    else:
        return None

################################################################################################################################################################

'''
Function Name: reset_acknowledgment
Input: ---
Output: ---
Logic: This function resets previous_ack_received to "True" after a delay
	
	The global variable "previous_ack_received" is set to "True" after a 5 second delay.
	
Example Call: reset_acknowledgment()
'''

def reset_acknowledgment():
    global previous_ack_received
    time.sleep(5)  
    previous_ack_received = True

#################################################################################################################################################################

# Create a thread for resetting acknowledgment
reset_thread = threading.Thread(target=reset_acknowledgment)

# Start the reset thread
reset_thread.start()

'''
Function Name: bot_status
Input: corners - multidimensional list containing the coordinates of the corners of the detected aruco markers,
	ids - list containing the ids of the detected aruco markers
Output: an integer representing the presence of a particular event or no event
Logic: This function returns the status of the bot with respect to the events.

	The distances between the bots aruco marker and the events aruco markers are calculated using the calculate_distance function.
	Now if the distances fall within a certain threshold (which has been defined in the events global variables and is based on the arena detection 
	from the camera), an integer corresponding to that event is returned (9-E, 8-D, 7-C, 6-B, 5-A). If a None is returned or the threshold criteria is
	not met, then a 4 is returned, signifying that there is no event nearby.
	
Example Call: message = bot_status(corners, ids)
'''

def bot_status(corners, ids):
    
    e_dist = [calculate_distance(corners, ids, EVENT_E[0][0], BOT), calculate_distance(corners, ids, EVENT_E[0][1], BOT)]
    d_dist = [calculate_distance(corners, ids, EVENT_D[0][0], BOT), calculate_distance(corners, ids, EVENT_D[0][1], BOT)]
    c_dist = [calculate_distance(corners, ids, EVENT_C[0][0], BOT), calculate_distance(corners, ids, EVENT_C[0][1], BOT)]
    b_dist = [calculate_distance(corners, ids, EVENT_B[0][0], BOT), calculate_distance(corners, ids, EVENT_B[0][1], BOT)]
    a_dist = [calculate_distance(corners, ids, EVENT_A[0][0], BOT), calculate_distance(corners, ids, EVENT_A[0][1], BOT)]

    if None not in e_dist and (e_dist[0] < EVENT_E[1] and e_dist[1] > EVENT_E[2]):
        print(f"Event E: {e_dist}")
        return 9

    if None not in d_dist and (d_dist[0] < EVENT_D[1] and d_dist[1] > EVENT_D[2]):
        print(f"Event D: {d_dist}")
        return 8

    if None not in c_dist and (c_dist[0] < EVENT_C[1] and c_dist[1] < EVENT_C[1]):
        print(f"Event C: {c_dist}")
        return 7

    if None not in b_dist and (b_dist[0] < EVENT_B[1] and b_dist[1] < EVENT_B[1]):
        print(f"Event B: {b_dist}")
        return 6

    if None not in a_dist and (a_dist[0] < EVENT_A[1] and a_dist[1] > EVENT_A[2]):
        print(f"Event A: {a_dist}")
        return 5

    return 4

#####################################################################################################################################################################

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


frame_no = 0
previous_ack_received = True
bot_event_index = 0
sendMessage = True


while cap.isOpened():
    ret, frame = cap.read()
    if frame_no<20:
        frame_no+=1
        continue
    sample_track.tracking(frame, ret)


    if not ret:
        break

    # Detect markers
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)


    message = bot_status(corners, ids)

    if message is not None and message != 4 and sendMessage == True:
        print(message)

        if message == bot_stop_nos[bot_event_index]:

            print(previous_ack_received)
            # Send the message only if the previous acknowledgment was received
            if previous_ack_received:
                s.sendall(f"{message}\n".encode('utf-8'))
                previous_ack_received = False
                print(message)

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
