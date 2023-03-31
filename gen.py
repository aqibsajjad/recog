import cv2
from deepface import DeepFace

# Create a video capture object
cap = cv2.VideoCapture(r"C:\Users\aqib-\OneDrive\Skrivebord\MVI_2002.MP4")

# Loop through the video frames and analyze each frame
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if frame was read successfully
    if not ret:
        break

    # Analyze the frame using DeepFace with enforce_detection set to False
    result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

    # Print the emotion analysis results
    print(result)

# Release the video capture object and destroy any open windows
cap.release()
cv2.destroyAllWindows()
