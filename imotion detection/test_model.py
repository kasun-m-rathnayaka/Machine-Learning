import cv2
import pickle
from utils import get_face_landmarks

cap = cv2.VideoCapture(0)

emotions = ['happy', 'sad', 'surprised']

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

while True:
    ret, frame = cap.read()

    face_landmarks = get_face_landmarks(frame, static_image_mode=False , draw=True)
    output = model.predict([face_landmarks])

    cv2.putText(frame, emotions[int(output[0])], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()