# Automatic-Attendance-Marker By Using Camera 
This Face Recognition Attendance System allows for automatic attendance tracking using real-time facial recognition. The system uses a deep learning-based CNN model alongside traditional Euclidean distance methods to recognize faces and log attendance.
It also includes features for handling duplicate images and generating attendance reports in an Excel file.

# Features

Real-time Face Recognition using webcam.

Attendance Tracking in an Excel sheet.

Face Detection & Recognition via deep learning (CNN + Euclidean distance).

Duplicate Image Removal from the dataset.

Automatic Excel Export of attendance data with detailed tracking (first appearance, last appearance, total duration).

Face Landmark Detection using dlib.

# Prerequisites

Make sure you have the following installed:

Python 3.x or higher

OpenCV (for computer vision tasks)

dlib (for face landmark detection)

face_recognition (for recognizing faces)

TensorFlow & Keras (for training and using the CNN)

Numpy (for numerical operations)

openpyxl (for handling Excel files)

SciPy (for Euclidean distance calculations)

# Installation

To set up the environment and install the required libraries, run the following commands in your terminal:

```
bash

# Install OpenCV for computer vision tasks
pip install opencv-python

# Install dlib for face detection and landmark estimation
pip install dlib

# Install face_recognition for face recognition
pip install face_recognition

# Install TensorFlow for CNN model creation and predictions
pip install tensorflow

# Install Keras (included with TensorFlow)
pip install keras

# Install Numpy for numerical operations
pip install numpy

# Install openpyxl for working with Excel files
pip install openpyxl

# Install SciPy for Euclidean distance calculations
pip install scipy

```

# Download Dlib's Face Landmark Model

You will need the shape_predictor_68_face_landmarks.dat file to perform face landmark detection. You can download it from here.
```
https://www.kaggle.com/datasets/sajikim/shape-predictor-68-face-landmarks
```
After downloading, extract it and place the file in your project directory.

# Project Setup

# Prepare the Dataset:
Create a folder named dataset in your project directory. Inside this folder, create subdirectories for each person and add their face images (e.g., .jpg, .png files).


# Example directory structure:

```
dataset/
  ├── person1/
  │   ├── img1.jpg
  │   ├── img2.jpg
  ├── person2/
  │   ├── img1.jpg

  ```
The system will scan these folders and recognize faces from the images for future comparison during live video capture.

# Run the Face Recognition Script: 

Once your dataset is ready, you can start the face recognition system using the following command:

```
bash

python face_recognition_attendance.py

```
# How the System Works:

The script will open your webcam and start real-time face detection.

When a face is detected, it attempts to match it with the faces in the dataset using a combination of Euclidean distance and a CNN model (if trained weights are provided).


The attendance is logged in an Excel file (attendance.xlsx or final_attendance.xlsx).

# Attendance Tracking:

When a recognized face is detected, the system will track the first appearance, last appearance, and the duration of the individual’s presence in front of the camera.


The attendance data is recorded in an Excel file, which includes:

Name

First Appearance Time

Last Appearance Time

Duration (time spent in front of the camera)

# Key Components of the Code:


# 1. Face Detection and Encoding
The system uses dlib and face_recognition to detect and encode faces. The encodings are used to compare faces for recognition.

# Python
```

import face_recognition

# Load image and detect face locations
image = face_recognition.load_image_file("person_image.jpg")
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

# Recognize face by encoding
name = recognize_face_euclidean(face_encodings[0], known_face_encodings, known_face_names)
2. CNN Model for Face Recognition
A Convolutional Neural Network (CNN) is used to improve accuracy for recognizing faces. The model is trained on images in the dataset directory. If available, trained weights can be loaded into the model.

from tensorflow import keras

def create_cnn_model(input_shape):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(set(known_face_names)), activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
3. Attendance Data Logging in Excel
The attendance data (e.g., first appearance, last appearance, total duration) is logged into an Excel file using the openpyxl library.

import openpyxl

# Create or update Excel file with attendance data
def create_or_update_excel(excel_file, name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not os.path.isfile(excel_file):
        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        worksheet.title = "Attendance"
        worksheet.append(["Name", "First Appearance", "Last Appearance", "Duration"])
    else:
        workbook = openpyxl.load_workbook(excel_file)
        worksheet = workbook["Attendance"]
    
    worksheet.append([name, timestamp, timestamp, "0:00:00"])
    workbook.save(excel_file)
```

# How the System Works

# Real-time Face Recognition

Video Capture: The script captures video from the webcam and detects faces in real-time.

# Face Recognition: 

It attempts to recognize faces by comparing them with the encoded faces stored in the dataset using both the Euclidean distance method and a trained CNN model.

# Attendance Tracking:

The system tracks when the person appears in front of the camera and logs the first appearance, last appearance, and time spent in front of the camera into the Excel sheet.

#Duplicate Image Removal

The system checks for duplicate images in the dataset and automatically removes them based on their hash.

# python
```
def remove_duplicate_images(dataset_dir):
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            image_files = [f for f in os.listdir(person_dir) if f.endswith(".jpg") or f.endswith(".png")]
            seen_images = set()
            for image_file in image_files:
                image_path = os.path.join(person_dir, image_file)
                image_hash = hash(open(image_path, "rb").read())
                if image_hash in seen_images:
                    os.remove(image_path)
                    print(f"Removed duplicate image: {image_path}")
                else:
                    seen_images.add(image_hash)
```

# File Structure

```
.
├── dataset/                         # Face images of known individuals
│   ├── person1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   ├── person2/
│   │   ├── img1.jpg
├── face_recognition_attendance.py   # Main script for face recognition and attendance tracking
├── shape_predictor_68_face_landmarks.dat  # Dlib's face landmark model
├── cnn_model_weights.h5             # Trained CNN model weights (if available)
├── attendance.xlsx                  # Raw attendance data (if you save manually)
└── final_attendance.xlsx            # Final attendance record
```

# Final Notes
Make sure the shape_predictor_68_face_landmarks.dat is correctly placed in your project directory.

Press q to stop the webcam and exit the program.

You can further customize the dataset and model as per your requirements.