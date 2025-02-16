import cv2, os
import numpy as np

img = cv2.imread(os.path.join('.', 'data', 'edge.jpg'))

#Canny edge detecter
img_edge = cv2.Canny(img, 100, 400)
img_edge_dilate = cv2.dilate(img_edge, np.ones((2,2), dtype=np.int8))
img_edge_erode = cv2.erode(img_edge_dilate, np.ones((2,2), dtype=np.int8))

# show images
cv2.imshow('img',img_edge)
cv2.imshow('img_dilate',img_edge_dilate)
cv2.imshow('img_edge_erode',img_edge_erode)
cv2.waitKey(0)
