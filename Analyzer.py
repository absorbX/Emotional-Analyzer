import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Map emotion labels to corresponding indices
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_model = './Emotion_little_vgg.h5'

# Load the facial emotion analysis model
model = load_model(emotion_model)

# Take user input for selecting webcam or video file
user_input = input("Enter 1 for webcam or 2 for video file: ")

if user_input == '1':
    # Open the webcam
    cap = cv2.VideoCapture(0)
elif user_input == '2':
    # Read video file
    video_file_path = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(video_file_path)
else:
    print("Invalid input. Please enter 1 for webcam or 2 for video file.")
    exit()

while True:
    # Read a frame from the webcam or video file
    ret, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, (640, 480))

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Perform emotion analysis if faces are detected
    for (x, y, w, h) in faces:
        # Extract the face region
        face_img = gray[y:y+h, x:x+w]
        # Resize and normalize the face image
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)

        # Predict emotions
        emotion_pred = model.predict(face_img)
        emotion_idx = np.argmax(emotion_pred)
        emotion_label = emotion_labels[emotion_idx]

        # Display the predicted emotion text
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (4, 0, 255), 1)
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (110, 49, 255), 2)

    # Display the frame on the screen
    cv2.imshow('Emotion Analysis', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam or video file and close the window
cap.release()
cv2.destroyAllWindows()
