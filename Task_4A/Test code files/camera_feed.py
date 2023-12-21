import cv2
import os

# Initialize webcam capture
video_capture = cv2.VideoCapture(2)  # 0 for default webcam, change if needed

# Set the resolution to the maximum supported by the webcam
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Width
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height

frame_count = 0
captured_frames_3_5 = []  # List to store captured frames 3-5
captured_frames_10_20 = []  # List to store captured frames 10-20

while frame_count < 20:
    ret, frame = video_capture.read()  # Capture a single frame
    if ret:
        frame_count += 1

        if 3 <= frame_count <= 5:
            captured_frames_3_5.append(frame)  # Store frames 3-5

        if 10 <= frame_count <= 20:
            captured_frames_10_20.append(frame)  # Store frames 10-20

# Release the video capture
video_capture.release()

# Create directories to save frames
os.makedirs("frames_3_5_version2", exist_ok=True)
os.makedirs("frames_10_20_version2", exist_ok=True)

# Save frames 3-5
for i, frame in enumerate(captured_frames_3_5):
    filename = f"frames_3_5/frame_{i+1}.jpg"  # Naming each frame
    cv2.imwrite(filename, frame)

print(f"{len(captured_frames_3_5)} frames (3-5) saved successfully.")

# Save frames 10-20
for i, frame in enumerate(captured_frames_10_20):
    filename = f"frames_10_20/frame_{i+10}.jpg"  # Naming each frame starting from 10
    cv2.imwrite(filename, frame)

print(f"{len(captured_frames_10_20)} frames (10-20) saved successfully.")
