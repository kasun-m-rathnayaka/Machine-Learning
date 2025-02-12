import cv2
import os

image = cv2.imread('data/img.jpg')
print(image.shape)
print(type(image))

# read image
image_path = os.path.join('.','data','img.jpg')
img = cv2.imread(image_path)

# imgage write
cv2.imwrite(os.path.join('.','data','img_out.png'), img)

# visualize image
cv2.imshow('image',img)
cv2.waitKey(0)