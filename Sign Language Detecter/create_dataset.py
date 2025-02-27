import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle
DATA_DIR = "./data"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        cv2.imshow('img', img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data_aux = []
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_aux.append(x)
                    data_aux.append(y)

        data.append(data_aux)
        labels.append(dir_)
#         plt.figure()
#         plt.imshow(img_rgb)
# plt.show()

f = open('data.pkl', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

