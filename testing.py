# pylint:disable=no-member
import os
import cv2 as cv
import numpy as np
import time

# Load trained face recognizer model
model_path = 'face_trained.yml'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model file '{model_path}' not found!")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(model_path)

# Load Haar Cascade for face detection
haar_cascade_path = 'haar_face.xml'
if not os.path.exists(haar_cascade_path):
    raise FileNotFoundError(f"Haar cascade file '{haar_cascade_path}' not found!")

haar_cascade = cv.CascadeClassifier(haar_cascade_path)

if haar_cascade.empty():
    raise RuntimeError("⚠️ Error loading Haar cascade! Check 'haar_face.xml'.")

# Load label names
people_path = 'people.npy'
if not os.path.exists(people_path):
    raise FileNotFoundError(f"People labels file '{people_path}' not found!")

people = np.load(people_path, allow_pickle=True)

# Start video capture from webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam!")

print("🎥 Starting real-time face recognition. Press 'q' to exit.")

confidence_threshold = 60  # Adjust based on testing

seen_people = {}  # Dictionary to track when a person was first seen

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Warning: Failed to capture frame, retrying...")
        continue

    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    current_seen = []  # Track people detected in the current frame

    for (x, y, w, h) in faces_rect:
        face_roi = gray[y:y+h, x:x+w]

        # Predict label
        label, confidence = face_recognizer.predict(face_roi)

        if confidence < confidence_threshold and 0 <= label < len(people):
            person_name = people[label]
        else:
            person_name = "Unknown"

        # Track known persons with timestamps
        if person_name != "Unknown":
            current_seen.append(person_name)  # Add to current frame detections
            if person_name not in seen_people:
                seen_people[person_name] = time.time()  # Store first seen time
            else:
                elapsed_time = time.time() - seen_people[person_name]
                if elapsed_time >= 5:  # Person has been in frame for at least 5 seconds
                    print(f"✅ {person_name} has been here for 5 seconds!")
                    seen_people.pop(person_name)  # Remove to avoid repeated messages

        # Display label & confidence
        text = f"{person_name} ({confidence:.2f})"
        cv.putText(frame, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Remove people who are no longer in the frame
    for person in list(seen_people.keys()):
        if person not in current_seen:
            del seen_people[person]  # Reset if the person disappears

    # Show the video feed
    cv.imshow("Face Recognition", frame)

    # Exit when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
print("🛑 Face recognition stopped.")

