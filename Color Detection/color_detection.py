import cv2
import numpy as np
from utils import get_limits
from PIL import Image

# read webcam
cap = cv2.VideoCapture(0)
yellow = [0, 255, 255]  # yellow in BGR color space

while True:
    ret, frame = cap.read()
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # pick color range
    lower_limit, upper_limit = get_limits(yellow)
    mask = cv2.inRange(hsv_img, lower_limit, upper_limit)
    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
