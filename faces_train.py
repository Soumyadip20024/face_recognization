import os
import cv2 as cv
import numpy as np
from tqdm import tqdm  # For progress bar

# Define people categories (ensure these match folder names)
people = ['viratkohli', 'srk', 'soumya', 'saptak', 'rohit']

# Define training data directory
#DIR = os.path.abspath('C:\face_rec\train_dataset')
DIR = r"C:\face_rec\train_dataset"

# Load Haar Cascade for face detection
haar_cascade_path = 'haar_face.xml'
if not os.path.exists(haar_cascade_path):
    raise FileNotFoundError(f"Haar cascade file '{haar_cascade_path}' not found!")

haar_cascade = cv.CascadeClassifier(haar_cascade_path)

# Storage for features and labels
features = []
labels = []

def create_train():
    """
    Function to read images from the dataset, detect faces, and store features and labels.
    """
    if not os.path.exists(DIR):
        raise FileNotFoundError(f"Training data directory '{DIR}' not found!")

    print("🔄 Training in progress...")

    for person in tqdm(people, desc="Processing Faces"):
        path = os.path.join(DIR, person)
        label = people.index(person)

        if not os.path.exists(path):
            print(f"⚠️ Warning: Directory '{path}' not found, skipping...")
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            # Read image
            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"⚠️ Warning: Could not read {img_path}, skipping...")
                continue 
                
            # Convert to grayscale
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Detect faces
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]  # Extract face region
                features.append(faces_roi)
                labels.append(label)

create_train()
print("\n✅ Training completed successfully!")

# Convert to numpy arrays
features = np.array(features, dtype=object)
labels = np.array(labels)

# Check if training data exists before proceeding
if len(features) == 0 or len(labels) == 0:
    raise ValueError("No faces were found! Check dataset or Haar cascade settings.")

# Create Face Recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer
print("🧠 Training the model...")
face_recognizer.train(features, labels)

# Save the trained model and metadata
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
np.save('people.npy', np.array(people))  # Save people list to ensure consistency

print("🎉 Model training complete! Files saved:")
print(" - face_trained.yml")
print(" - features.npy")
print(" - labels.npy")
print(" - people.npy")
