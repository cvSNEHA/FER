import cv2
from fer import FER
from tensorflow import *

# Load pre-trained facial emotion recognition model
detector = FER()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = detector.detect_emotions(frame)

    # Draw rectangles around the faces and display emotions
    for face in faces:
        x, y, w, h = face['box']
        emotions = face['emotions']

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100,200,100), 2)

        # Display emotions
        emotion_text = max(emotions, key=emotions.get)
        cv2.putText(frame, emotion_text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100,200,100), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'esc' key is pressed
    if cv2.waitKey(1) & 0xFF==27:
        break

# Release the capture object and close the window
cap.release()
cv2.destroyAllWindows()