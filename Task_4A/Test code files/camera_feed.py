import cv2

# Initialize webcam capture
video_capture = cv2.VideoCapture(2)  # 0 for default webcam, change if needed

# Set the resolution to the maximum supported by the webcam
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height

frame_count = 0
while frame_count != 20:
    ret, frame = video_capture.read()  # Capture a single frame
    frame_count += 1

# Release the video capture
video_capture.release()

cv2.imshow("Captured Frame", frame)
print(frame.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Check if a frame was successfully captured
if ret:
    print("Frame captured successfully.")
    # The 'frame' variable contains the captured frame
else:
    print("Failed to capture frame.")
