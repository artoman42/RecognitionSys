import dlib
import cv2
import numpy as np
#load the face detector, landmark predictor, and
# face recognition model

face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor(
                                         "models/shape_predictor_68_face_landmarks.dat"
)

face_encoder = dlib.face_recognition_model_v1(
                                             "models/dlib_face_recognition_resnet_model_v1.dat"
)

# Import libraries
import os
from glob import glob

# Define valid image extensions
VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def get_image_paths(root_dir, class_names):
  """
  This function grabs the paths to the images in the dataset.

  Args:
      root_dir: The root directory of the dataset.
      class_names: A list of the class names in the dataset.

  Returns:
      A list of image paths.
  """

  # Initialize an empty list to store image paths
  image_paths = []

  # Loop through the class names
  for class_name in class_names:

    # Get the path to the current class directory
    class_dir = os.path.sep.join([root_dir, class_name])

    # Get all file paths in the class directory using glob
    class_file_paths = glob(os.path.sep.join([class_dir, '*.*']))  

    # Loop through the file paths in the class directory
    for file_path in class_file_paths:

      # Extract the file extension from the current file path
      ext = os.path.splitext(file_path)[1]

      # Check if the file extension is valid
      if ext.lower() not in VALID_EXTENSIONS:
        print("Skipping file: {}".format(file_path))
        continue  # Skip the file if the extension is not valid

      # Add the path to the current image to the list of image paths
      image_paths.append(file_path)

  # Return the list of image paths
  return image_paths

def face_rects(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  rects = face_detector(gray, 1)
  return rects

def face_landmarks(image):
  return [shape_predictor(image, face_rect) for face_rect in face_rects(image)]

def face_encodings(image):
  return [
    np.array(face_encoder.compute_face_descriptor(image, face_landmark)) for face_landmark in face_landmarks(image)
  ]

def nb_of_matches(known_encodings, unknown_encodings):
    distances = np.linalg.norm(known_encodings - unknown_encodings, axis=1)
    small_distances = distances <= 0.6
    return sum(small_distances)