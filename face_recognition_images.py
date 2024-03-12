import pickle
import cv2
import numpy as np
from utils import face_encodings, nb_of_matches, face_rects

with open('encodings.pickle', 'rb') as f:
    name_encodings_dict = pickle.load(f)

image = cv2.imread("examples/1.jpg")
image = cv2.resize(image, (250, 250))
encodings = face_encodings(image)

names = []

for encoding in encodings:
    counts = {}

    for (name, encodings) in name_encodings_dict.items():
        counts[name] = nb_of_matches(encodings, encoding)

    if all(count == 0 for count in counts.values()):
        name = "Unkown"

    else:
        name = max(counts, key=counts.get)

    names.append(name)

for rect, name in zip(face_rects(image), names):

    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(image, name, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
cv2.imshow("image", image)
cv2.waitKey(0)