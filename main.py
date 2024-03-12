from utils import get_image_paths, face_encodings
import cv2

# Define class names
class_names = ['Angelina_Jolie']

# Get image paths (function call likely from another script)
image_paths = get_image_paths("dataset", class_names)

image = cv2.imread(image_paths[0])
print(face_encodings(image))
print(face_encodings(image)[0].shape)
