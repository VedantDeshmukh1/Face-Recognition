import cv2
import face_recognition
import numpy as np
import os
from pathlib import Path
import time

# Function to collect a dataset of face images
def capture_face_dataset(dataset_path, num_images=100):
    # Create the dataset directory if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize a counter for the image filenames
    image_counter = 1

    while image_counter <= num_images:
        # Capture a single frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            face_image = frame[y:y+h, x:x+w]

            # Save the face image to the dataset directory
            image_filename = f"face_{image_counter}.jpg"
            cv2.imwrite(os.path.join(dataset_path, image_filename), face_image)

            # Increment the image counter
            image_counter += 1

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame with detected faces
        cv2.imshow('Capturing Face Dataset', frame)

        # Break the loop if the desired number of images is reached or 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q') or image_counter > num_images:
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

    print(f"Face dataset captured successfully. {num_images} images saved to {dataset_path}.")

dataset_path = "C:/Users/admin/Desktop/img"
num_images = 100
capture_face_dataset(dataset_path, num_images)

# Function to compare detected face with the dataset
def is_face_matched(known_face_encodings, face_to_check):
    matches = face_recognition.compare_faces(known_face_encodings, face_to_check)
    return True in matches

# Load known face images and encode them
def load_and_encode_known_faces(dataset_path):
    known_face_encodings = []
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        image = face_recognition.load_image_file(image_path)
        # Get face encodings for the image
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            # Take the first face encoding found in the image
            face_encoding = face_encodings[0]
            known_face_encodings.append(face_encoding)
        else:
            print(f"No faces found in {image_name}. Skipping.")
    return known_face_encodings


# Load and encode the known face images
known_face_encodings = load_and_encode_known_faces(dataset_path)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Capture frames continuously
while True:
    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert captured image to RGB (face_recognition uses RGB images)
    rgb_frame = frame[:, :, ::-1]

    # Get face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop over each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = is_face_matched(known_face_encodings, face_encoding)
        if matches:
            print("Face matched! Opening the folder...")
            # Open the protected folder
            os.startfile(dataset_path)
            break
        else:
            print("Face not matched! Access denied.")

    # Show the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()