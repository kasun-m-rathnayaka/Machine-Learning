import os
import cv2
import numpy as np

from utils import get_face_landmarks

# clean the data
# balance data set, remove outliers

# prepare tha data
data_dir = './data'

output = []
for emotion_index ,emotion in enumerate(sorted(os.listdir(data_dir))):
    for image_path in os.listdir(os.path.join(data_dir, emotion)):
        image = cv2.imread(os.path.join(data_dir, emotion, image_path))
        face_landmarks = get_face_landmarks(image)

        if len(face_landmarks) == 1404:
            output.append(face_landmarks)
            face_landmarks.append(int(emotion_index))

np.savetxt('data.txt', np.asarray(output))



# train model

# test model
