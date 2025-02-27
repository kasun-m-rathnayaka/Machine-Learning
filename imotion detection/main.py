import os
import cv2
from utils import get_face_landmarks

# clean the data
# balance data set, remove outliers

# prepare tha data
data_dir = './data'

output = []
for emotion in os.listdir(data_dir):
    for image_path in os.listdir(os.path.join(data_dir, emotion)):
        image = cv2.imread(os.path.join(data_dir, emotion, image_path))
        face_landmarks = get_face_landmarks(image)
        print(emotion)
        # if len(face_landmarks):

# train model

# test model
